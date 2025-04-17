import torch
import torch.nn as nn
from SWOW_prediction.models import *
from SWOW_prediction.data_preprocessing import *
from scipy.stats import spearmanr
import yaml
import tqdm
import wandb
import gc
from sklearn.metrics import r2_score


def get_likelihood_loss(reduction = 'sum', device = 'cuda'):
    def get_function(input, output):
        var = torch.ones(input.shape, requires_grad=False)
        var = var.to(device)
        loss = loss_function(input, output, var)
        return loss
    loss_function = torch.nn.GaussianNLLLoss(reduction = reduction)
    return get_function

def get_l1_regularization(model, lambda_value: 0.0):
    l1_parameters = []
    if lambda_value == 0.0:
        return 0.0
    for name, param in model.named_parameters():
        if 'bias' not in name:
            l1_parameters.append(param.view(-1))
    
    l1_regularization = lambda_value * torch.abs(torch.cat(l1_parameters)).sum()

    return l1_regularization

def get_row_difference_from_1(row_sum: torch.Tensor):
  
    return torch.sum(torch.abs(row_sum - 1))

def get_config(config_path ='src/SWOW_prediction/config_features.yml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def train(config_file):
    model_name = config_file['model_name']
    data_name = config_file['data_name']
    data_path = config_file['data_path'][data_name]
    device_name = config_file['device_name']
    eval = config_file['eval']
    lr = config_file['learning_rate']
    max_length = config_file['max_length']
    token_strategy = config_file['token_strategy']
    graph_strategy = config_file['graph_strategy']
    graph_version = config_file['graph_version']
    node_neighbors = config_file['node_neighbors']
    swow_version = config_file['swow_version']
    fill = config_file['fill']
    add_self_loops = config_file['add_self_loops']
    num_epochs = config_file['num_epochs']
    num_super_epochs = config_file['num_super_epochs']
    batch_size = config_file['batch_size']
    normalize_swow = config_file['normalize_swow']
    negative_sampling = config_file['negative_sampling']
    negative_sample_num = config_file['negative_sample_num']
    property_name = config_file['property']
    take_log = config_file['take_log']
    k = config_file['k']
    walking_type = config_file['walking_type']
    reduce = config_file['reduce']
    negative_sample_num = negative_sample_num if negative_sample_num > 0 else None
    device = torch.device(device_name)
    two_gram = config_file['two_gram']
    train_section = config_file['train_section']

    swow_data_sets, textual_data, vocab_mapping = construct_property_dataset(model_name, 
                                                                    max_length, 
                                                                    data_name, 
                                                                    data_path, 
                                                                    token_strategy,
                                                                    graph_strategy, 
                                                                    device_name, 
                                                                    graph_version, 
                                                                    swow_version, 
                                                                    fill,
                                                                    add_self_loops, 
                                                                    node_neighbors,
                                                                    normalize_swow,
                                                                    take_log,
                                                                    negative_sampling,
                                                                    negative_sample_num,
                                                                    property_name,
                                                                    two_gram,
                                                                    k = k, walking_type = walking_type,
                                                                    data_features = config_file['data_features'])
    
    torch.cuda.empty_cache() 
    n = len(vocab_mapping)
    in_channels = textual_data['embedding'].shape[1]
    hidden_channels = config_file['graph_encoder_out_size']
    out_channels = config_file['graph_encoder_out_size']
    num_layers = config_file['graph_encoder_num_layers']
    num_heads = config_file['graph_encoder_num_heads']
    dropout = config_file['graph_encoder_dropout']
    encoder_model_name = config_file['graph_encoder']
    gamma = config_file['gamma']
    activation_function = config_file['activation_function']
    eps = config_file['epsilon']
    l1_lambda = config_file['l1_regularization']
    loss_function_name = config_file['loss_function']
    add_linear = config_file['add_linear']

    if config_file['baseline'] == True:
        model = BasicPropertyPredictor(in_channels,dropout) #Baseline model
    
    else:
        model = ParameterPropertyPredictor(n, 
                                    in_channels,
                                    hidden_channels, 
                                    out_channels, 
                                    num_layers, 
                                    num_heads, 
                                    dropout, 
                                    encoder_model_name, 
                                    reduce,
                                    add_linear)
    if torch.cuda.is_available():
        model.cuda(device)
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500], gamma=gamma)
    
    loss_function = get_likelihood_loss(reduction = 'sum',device=device)
    clip_value = 1
    total_loss = 0
    
    embeddings = textual_data['embedding'].float().to(device)

    text_edge_index, text_edge_weight = textual_data['train']
    text_edge_index = text_edge_index.long().to(device)
    text_edge_weight = text_edge_weight.float().to(device)
    
    swow_data_sets = swow_data_sets[train_section]
    train_index, train_target, train_words = swow_data_sets['train']
    train_index = train_index.long().to(device)
    train_target = train_target.float().to(device)
    

    dev_index, dev_target,dev_words = swow_data_sets['dev']
    dev_index = dev_index.long().to(device)
    dev_target = dev_target.float().to(device)
    

    
    best_r2 = 0
    for super_epoch in tqdm.tqdm(range(num_super_epochs)):
        
        for epoch in  tqdm.tqdm(range(num_epochs)):
            model.train()
            optimizer.zero_grad()
            if config_file['baseline'] == False:
                output = model(embeddings, text_edge_index,  text_edge_weight)
            else:
                output = model(embeddings)
            output_train = output[train_index]
                
            
            loss = loss_function(output_train, train_target) + get_l1_regularization(model, l1_lambda) 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            loss.backward()            
            optimizer.step()
            scheduler.step()
            loss_value = loss.item()
            total_loss += loss_value
            
            tqdm.tqdm.write(f'Epoch: {epoch} | Loss: {loss.item()}')


            model.eval() 
            with torch.no_grad():
                if config_file['baseline'] == False:
                    output = model(embeddings, text_edge_index,  text_edge_weight)
                else:
                    output = model(embeddings)
                output_dev = output[dev_index]
                
                loss = loss_function(output_dev, dev_target) + \
                    get_l1_regularization(model, l1_lambda) 
                
                outputs = output_dev.cpu().numpy()
                targets = dev_target.cpu().numpy()
                
                
                spearman_corr = spearmanr(targets.reshape(-1), outputs.reshape(-1))
                r2 = r2_score(targets.reshape(-1), outputs.reshape(-1))
            
                
                tqdm.tqdm.write(f'Epoch: {epoch} | Loss: {loss.item()},\
                                 Spearman correlation: {spearman_corr[0]}, \
                                 p-value: {spearman_corr[1]},\
                                 R2: {r2}')

                
                if r2 > best_r2 and  (epoch > 10 or super_epoch > 0):
                    best_loss = loss
                    
                    
                    best_r2 = r2
                    
                    if config_file['baseline'] == False:
                        torch.save(model.state_dict(), 
                        os.path.join(config_file['model_saving_dir'],\
                            f'{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt'))
                    else:
                        torch.save(model.state_dict(), 
                        os.path.join(config_file['model_saving_dir'],\
                            f'{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt'))
    
    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()


def evaluate(config_file, section = 'dev'):
    
    if section not in ['test','dev']: #Evaluating on the historical sections
        return evaluate_on_new(config_file, section)

    model_name = config_file['model_name']
    data_name = config_file['data_name']
    data_path = config_file['data_path'][data_name]
    device_name = config_file['device_name']
    eval = config_file['eval']
    lr = config_file['learning_rate']
    max_length = config_file['max_length']
    token_strategy = config_file['token_strategy']
    graph_strategy = config_file['graph_strategy']
    graph_version = config_file['graph_version']
    node_neighbors = config_file['node_neighbors']
    
    swow_version = config_file['swow_version']
    fill = config_file['fill']
    add_self_loops = config_file['add_self_loops']
    
    normalize_swow = config_file['normalize_swow']
    negative_sampling = config_file['negative_sampling']
    negative_sample_num = config_file['negative_sample_num']
    property = config_file['property']
    k = config_file['k']
    take_log = config_file['take_log']
    walking_type = config_file['walking_type']
    reduce = config_file['reduce']
    negative_sample_num = negative_sample_num if negative_sample_num > 0 else None
    device = torch.device(device_name)
    two_gram = config_file['two_gram']
    data_features = config_file['data_features']
    train_section = config_file['train_section']
    
    swow_data_sets, textual_data, vocab_mapping = construct_property_dataset(model_name, 
                                                                    max_length, 
                                                                    data_name, 
                                                                    data_path, 
                                                                    token_strategy,
                                                                    graph_strategy, 
                                                                    device_name, 
                                                                    graph_version, 
                                                                    swow_version, 
                                                                    fill,
                                                                    add_self_loops, 
                                                                    node_neighbors,
                                                                    normalize_swow,
                                                                    take_log,
                                                                    negative_sampling,
                                                                    negative_sample_num,
                                                                    property,
                                                                    two_gram,
                                                                    k = k, walking_type = walking_type,
                                                                    data_features = data_features,
                                                                    store_dir = config_file['store_dir'])
    swow_data_sets = swow_data_sets[train_section]
    
    n = len(vocab_mapping)
    in_channels = textual_data['embedding'].shape[1]
    hidden_channels = config_file['graph_encoder_out_size']
    out_channels = config_file['graph_encoder_out_size']
    num_layers = config_file['graph_encoder_num_layers']
    num_heads = config_file['graph_encoder_num_heads']
    dropout = config_file['graph_encoder_dropout']
    encoder_model_name = config_file['graph_encoder']
    eps = config_file['epsilon']
    l1_lambda = config_file['l1_regularization']
    loss_function_name = config_file['loss_function']
    
    add_linear = config_file['add_linear']
    
    if config_file['baseline'] == True:
        model = BasicPropertyPredictor(in_channels,dropout)
    else:
        model = ParameterPropertyPredictor(n, 
                                    in_channels,
                                    hidden_channels, 
                                    out_channels, 
                                    num_layers, 
                                    num_heads, 
                                    dropout, 
                                    encoder_model_name, 
                                    reduce,
                                    add_linear)
    torch.cuda.empty_cache() 
    if torch.cuda.is_available():
        model.cuda(device)
    torch.autograd.set_detect_anomaly(True)
    
    loss_function = get_likelihood_loss(reduction = 'sum', device=device)
    
    train_index, train_target, train_words = swow_data_sets['train']
    train_index = train_index.long().to(device)
    train_target = train_target.float().to(device)
    
    dev_index, dev_target, dev_words = swow_data_sets[section]
    dev_index = dev_index.long().to(device)
    dev_target = dev_target.float().to(device)
    
    embeddings = textual_data['embedding'].float().to(device)
    text_edge_index, text_edge_weight = textual_data['train']
    text_edge_index = text_edge_index.long().to(device)
    text_edge_weight = text_edge_weight.float().to(device)

    if config_file['baseline'] == False:

        model.load_state_dict(torch.load(
            os.path.join(
            config_file['model_saving_dir'],\
            f'{property}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt'))
            )
    else:
        model.load_state_dict(
            torch.load(
                os.path.join(
                config_file['model_saving_dir'],\
                f'{property}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt'))
                )
    model.eval()
    model.to(device)

    
    with torch.no_grad():
        if config_file['baseline'] == False:
            output = model(embeddings, text_edge_index,  text_edge_weight)
        else:
            output = model(embeddings)
                           
        output_dev = output[dev_index]
        
        loss = loss_function(output_dev, dev_target) + \
                    get_l1_regularization(model, l1_lambda) 
        

        outputs = output_dev.cpu().numpy()
        targets = dev_target.cpu().numpy()
        
        spearman_corr = spearmanr(targets.reshape(-1), outputs.reshape(-1))
        r2 = r2_score(targets.reshape(-1), outputs.reshape(-1))

        tqdm.tqdm.write(f'Spearman correlation: {spearman_corr[0]}, p-value: {spearman_corr[1]}, R2: {r2}, Loss: {loss.item()}')
    
    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()
    
    df = pd.DataFrame({'targets':targets, 'outputs':outputs, 'words':dev_words})
    if config_file['baseline'] == False:
        df_dir = os.path.join(config_file['test_results_path'] ,
                               f'{property}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{section}.csv')
    else:
        df_dir = os.path.join(config_file['test_results_path'], 
                              f'{property}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{section}.csv')
    df.to_csv(df_dir, index = False)

    
    spearman_corr = spearmanr(df['targets'], df['outputs'])
    print(f'Spearman correlation: {spearman_corr[0]}')
    print(f'p-value: {spearman_corr[1]}')
    
    return df

def evaluate_on_new(config_file, section = 1):
    model_name = config_file['model_name']
    data_name = config_file['data_name']

    device_name = config_file['device_name']
    token_strategy = config_file['token_strategy']
    graph_strategy = config_file['graph_strategy']
    graph_version = config_file['graph_version']
    
    swow_version = config_file['swow_version']
    fill = config_file['fill']
    add_self_loops = config_file['add_self_loops']
    negative_sample_num = config_file['negative_sample_num']
    
    property = config_file['property']
    k = config_file['k']
    take_log = config_file['take_log']
    
    reduce = config_file['reduce']
    negative_sample_num = negative_sample_num if negative_sample_num > 0 else None
    device = torch.device(device_name)
    
    
    train_section = config_file['train_section']
    
    d = pickle.load(open(f'data/SWOW_prediction/data_{data_name}_{section}_{model_name}.pkl','rb'))
    embedding_data = d['embedding_data']
    vocab_mapping = d['vocab_mapping']
    edge_index = d['edge_index']
    edge_weight = d['edge_weight']
    year = d['year']
    vocab_mapping_reverse = {v:k for k,v in vocab_mapping.items()}
    embedding_tensor = np.array([embedding_data[vocab_mapping_reverse[i]] for i in range(len(vocab_mapping))])
    embedding_tensor = torch.tensor(embedding_tensor)


    torch.cuda.empty_cache() 
    n = len(vocab_mapping)
    in_channels = embedding_tensor.shape[1]
    hidden_channels = config_file['graph_encoder_out_size']
    out_channels = config_file['graph_encoder_out_size']
    num_layers = config_file['graph_encoder_num_layers']
    num_heads = config_file['graph_encoder_num_heads']
    dropout = config_file['graph_encoder_dropout']
    encoder_model_name = config_file['graph_encoder']
    loss_function_name = config_file['loss_function']
    add_linear = config_file['add_linear']
    
    if config_file['baseline'] == True:
        model = BasicPropertyPredictor(in_channels,dropout)
    else:
        model = ParameterPropertyPredictor(n, 
                                    in_channels,
                                    hidden_channels, 
                                    out_channels, 
                                    num_layers, 
                                    num_heads, 
                                    dropout, 
                                    encoder_model_name, 
                                    reduce,
                                    add_linear)
    
    if torch.cuda.is_available():
        model.cuda(device)
    torch.autograd.set_detect_anomaly(True)
    
    embeddings = embedding_tensor.float().to(device)

    text_edge_index = edge_index.long().to(device)
    text_edge_weight = edge_weight.float().to(device)

    
    if config_file['baseline'] == False:
        model.load_state_dict(torch.load(
            os.path.join(
            config_file['model_saving_dir'],\
            f'{property}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt'))
        )
    else:
        model.load_state_dict(torch.load(
            os.path.join(
            config_file['model_saving_dir'],\
            f'{property}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt'))
        )
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        if config_file['baseline'] == False:
            output = model(embeddings, text_edge_index,  text_edge_weight)
        else:
            output = model(embeddings)
        
        outputs = output.cpu().numpy()
        
    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()
    
    df = pd.DataFrame({ 'outputs':outputs, 
                       'words':[vocab_mapping_reverse[i] for i in range(len(embedding_tensor))],
                       'year': len(outputs)*[year]})


    if config_file['baseline'] == False:
        df_dir = os.path.join(config_file['test_results_path'] , '/time_series/', f'testing_data_{data_name}_{model_name}_{section}_{property}_train_section_{train_section}.csv')
    else:
        df_dir = os.path.join(config_file['test_results_path'] , '/time_series/' , f'testing_data_{data_name}_{model_name}_{section}_{property}_baseline_train_section_{train_section}.csv')
    
    df.to_csv(df_dir, index = False)


def main(config_path = 'SWOW_prediction/config_features.yml',**kwargs):
    
    config = get_config(config_path)
    eval_section = kwargs['eval_section']
    section = kwargs['section']
    baseline = kwargs['baseline']
    reduce = kwargs['reduce']
    config['baseline'] = baseline
    config['reduce'] = reduce
    os.makedirs(config['test_results_path'], exist_ok = True)
    os.makedirs(os.path.join(config['test_results_path'] , '/time_series/'), exist_ok = True)
    model_saving_dir = config['model_saving_dir']
    os.makedirs(model_saving_dir, exist_ok = True)
    

    
    if section != -1:
        config['train_section'] = section
    if eval_section != -1:
        config['eval_section'] = eval_section
    eval = config['eval']
    print('baseline:',config['baseline'], 'reduce:', config['reduce'], 'eval:', eval,
          'train_section:', config['train_section'], 'eval_section:', config['eval_section'])
   
    if eval == False:
        train(config)
    else:
        print(config['eval_section'])
        evaluate(config,config['eval_section'])

    
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='SWOW_prediction/config_features.yml')
    parser.add_argument('--section', type=int, default=-1)
    parser.add_argument('--eval_section', default=-1)
    parser.add_argument('--baseline', dest='baseline', action='store_true')
    parser.add_argument('--reduce', default='forward')
    parser.set_defaults(baseline=False)
    args = parser.parse_args()
    main(args.config_path, section = args.section, 
         eval_section = args.eval_section,
         baseline = args.baseline,
         reduce = args.reduce)
    
    

