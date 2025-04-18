import tqdm
import os
import gc
import pickle
import pandas as pd
import numpy as np
import yaml

import torch
from torch.optim.lr_scheduler import MultiStepLR
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


from SWOW_prediction.models import BasicPropertyPredictor, ParameterPropertyPredictor
from SWOW_prediction.data_preprocessing import construct_property_dataset

def get_likelihood_loss(reduction='sum', device='cuda'):
    """Create a Gaussian negative log likelihood loss function."""
    loss_function = torch.nn.GaussianNLLLoss(reduction=reduction)
    
    def loss_wrapper(input, output):
        var = torch.ones_like(input, requires_grad=False).to(device)
        return loss_function(input, output, var)
    
    return loss_wrapper

def get_l1_regularization(model, lambda_value=0.0):
    """Calculate L1 regularization for model parameters."""
    if lambda_value == 0.0:
        return 0.0
        
    l1_parameters = [param.view(-1) for name, param in model.named_parameters() 
                     if 'bias' not in name]
    
    return lambda_value * torch.abs(torch.cat(l1_parameters)).sum()



def get_config(config_path='src/SWOW_prediction/config_features.yml'):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def train(config_file):
    """Train the model based on configuration parameters."""
    

    model_name = config_file['model_name']
    data_name = config_file['data_name']
    data_path = config_file['data_path'][data_name]
    device_name = config_file['device_name']
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
    normalize_swow = config_file['normalize_swow']
    negative_sampling = config_file['negative_sampling']
    negative_sample_num = config_file['negative_sample_num']
    property_name = config_file['property']
    take_log = config_file['take_log']
    k = config_file['k']
    walking_type = config_file['walking_type']
    reduce = config_file['reduce']
    two_gram = config_file['two_gram']
    train_section = config_file['train_section']
    
    
    negative_sample_num = negative_sample_num if negative_sample_num > 0 else None
    
    
    device = torch.device(device_name)
    
    #Construct training dataset
    swow_data_sets, textual_data, vocab_mapping = construct_property_dataset(
        model_name, max_length, data_name, data_path, token_strategy,
        graph_strategy, device_name, graph_version, swow_version, fill,
        add_self_loops, node_neighbors, normalize_swow, take_log,
        negative_sampling, negative_sample_num, property_name, two_gram,
        k=k, walking_type=walking_type, data_features=config_file['data_features'],
        store_dir=config_file['store_dir']
    )
    
    
    torch.cuda.empty_cache() 
    
    # Extract model parameters
    n = len(vocab_mapping)
    in_channels = textual_data['embedding'].shape[1]
    hidden_channels = config_file['graph_encoder_out_size']
    out_channels = config_file['graph_encoder_out_size']
    num_layers = config_file['graph_encoder_num_layers']
    num_heads = config_file['graph_encoder_num_heads']
    dropout = config_file['graph_encoder_dropout']
    encoder_model_name = config_file['graph_encoder']
    gamma = config_file['gamma']
    eps = config_file['epsilon']
    l1_lambda = config_file['l1_regularization']
    loss_function_name = config_file['loss_function']
    add_linear = config_file['add_linear']
    
    
    if config_file['baseline']:
        model = BasicPropertyPredictor(in_channels, dropout) #Baseline BERT based model
    else:
        model = ParameterPropertyPredictor( #Custom GNN model
            n, in_channels, hidden_channels, out_channels, num_layers,
            num_heads, dropout, encoder_model_name, reduce, add_linear
        )
    
    
    if torch.cuda.is_available():
        model.cuda(device)
    
    
    torch.autograd.set_detect_anomaly(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[100, 500], gamma=gamma)
    loss_function = get_likelihood_loss(reduction='sum', device=device)
    clip_value = 1
    
    # Prepare data
    embeddings = textual_data['embedding'].float().to(device)
    text_edge_index, text_edge_weight = textual_data['train']
    text_edge_index = text_edge_index.long().to(device)
    text_edge_weight = text_edge_weight.float().to(device)
    
    swow_data_sets = swow_data_sets[train_section]
    train_index, train_target, train_words = swow_data_sets['train']
    train_index = train_index.long().to(device)
    train_target = train_target.float().to(device)
    
    dev_index, dev_target, dev_words = swow_data_sets['dev']
    dev_index = dev_index.long().to(device)
    dev_target = dev_target.float().to(device)
    
    # Training loop
    best_r2 = 0
    for super_epoch in tqdm.tqdm(range(num_super_epochs), desc="Super Epochs"):
        for epoch in tqdm.tqdm(range(num_epochs), desc=f"Epochs (Super Epoch {super_epoch})"):
            # Training step
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            if not config_file['baseline']:
                output = model(embeddings, text_edge_index, text_edge_weight)
            else:
                output = model(embeddings)
                
            output_train = output[train_index]
            
            # Calculate loss and update
            loss = loss_function(output_train, train_target) + get_l1_regularization(model, l1_lambda)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            tqdm.tqdm.write(f'Epoch: {epoch} | Loss: {loss.item():.6f}')
            
            # Evaluation step
            model.eval()
            with torch.no_grad():
                if not config_file['baseline']:
                    output = model(embeddings, text_edge_index, text_edge_weight)
                else:
                    output = model(embeddings) #Baseline model only words with embeddings, and not the graph
                    
                output_dev = output[dev_index]
                
                # Calculate metrics
                dev_loss = loss_function(output_dev, dev_target) + get_l1_regularization(model, l1_lambda)
                outputs = output_dev.cpu().numpy()
                targets = dev_target.cpu().numpy()
                
                spearman_corr = spearmanr(targets.reshape(-1), outputs.reshape(-1))
                r2 = r2_score(targets.reshape(-1), outputs.reshape(-1))
                
                tqdm.tqdm.write(
                    f'Epoch: {epoch} | Dev Loss: {dev_loss.item():.6f}, '
                    f'Spearman correlation: {spearman_corr[0]:.4f}, '
                    f'p-value: {spearman_corr[1]:.4f}, '
                    f'R2: {r2:.4f}'
                )
                
                # Save best model
                if r2 > best_r2 and (epoch > 10 or super_epoch > 0):
                    best_r2 = r2
                    
                    # Define model path
                    if not config_file['baseline']:
                        model_path = os.path.join(
                            config_file['model_saving_dir'],
                            f'{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_'
                            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
                            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt'
                        )
                    else:
                        model_path = os.path.join(
                            config_file['model_saving_dir'],
                            f'{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_'
                            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
                            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt'
                        )
                    
                    torch.save(model.state_dict(), model_path)
    
    # Clean up
    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()


def evaluate(config_file, section='dev'):
    """Evaluate the model on specified section."""
    # Handle historical sections
    if section not in ['test', 'dev']:
        return evaluate_on_new(config_file, section)
    
    # Extract configuration parameters
    model_name = config_file['model_name']
    data_name = config_file['data_name']
    data_path = config_file['data_path'][data_name]
    device_name = config_file['device_name']
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
    property_name = config_file['property']
    k = config_file['k']
    take_log = config_file['take_log']
    walking_type = config_file['walking_type']
    reduce = config_file['reduce']
    two_gram = config_file['two_gram']
    train_section = config_file['train_section']
    
   
    negative_sample_num = negative_sample_num if negative_sample_num > 0 else None
    
    
    device = torch.device(device_name)
    
    
    swow_data_sets, textual_data, vocab_mapping = construct_property_dataset(
        model_name, max_length, data_name, data_path, token_strategy,
        graph_strategy, device_name, graph_version, swow_version, fill,
        add_self_loops, node_neighbors, normalize_swow, take_log,
        negative_sampling, negative_sample_num, property_name, two_gram,
        k=k, walking_type=walking_type, data_features=config_file['data_features'],
        store_dir=config_file['store_dir']
    )
    
    swow_data_sets = swow_data_sets[train_section]
    
    
    n = len(vocab_mapping)
    in_channels = textual_data['embedding'].shape[1]
    hidden_channels = config_file['graph_encoder_out_size']
    out_channels = config_file['graph_encoder_out_size']
    num_layers = config_file['graph_encoder_num_layers']
    num_heads = config_file['graph_encoder_num_heads']
    dropout = config_file['graph_encoder_dropout']
    encoder_model_name = config_file['graph_encoder']
    l1_lambda = config_file['l1_regularization']
    loss_function_name = config_file['loss_function']
    add_linear = config_file['add_linear']
    
    
    if config_file['baseline']:
        model = BasicPropertyPredictor(in_channels, dropout)
    else:
        model = ParameterPropertyPredictor(
            n, in_channels, hidden_channels, out_channels, num_layers,
            num_heads, dropout, encoder_model_name, reduce, add_linear
        )
    
    
    torch.cuda.empty_cache()
    
    
    if torch.cuda.is_available():
        model.cuda(device)
    
    torch.autograd.set_detect_anomaly(True)
    loss_function = get_likelihood_loss(reduction='sum', device=device)
    
    
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
    
    # Loading pretrained model
    if not config_file['baseline']:
        model_path = os.path.join(
            config_file['model_saving_dir'],
            f'{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_'
            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt'
        )
    else:
        model_path = os.path.join(
            config_file['model_saving_dir'],
            f'{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_'
            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt'
        )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    
    
    with torch.no_grad():
        if not config_file['baseline']:
            output = model(embeddings, text_edge_index, text_edge_weight)
        else:
            output = model(embeddings)
            
        output_dev = output[dev_index]
        
        
        loss = loss_function(output_dev, dev_target) + get_l1_regularization(model, l1_lambda)
        outputs = output_dev.cpu().numpy()
        targets = dev_target.cpu().numpy()
        
        spearman_corr = spearmanr(targets.reshape(-1), outputs.reshape(-1))
        r2 = r2_score(targets.reshape(-1), outputs.reshape(-1))
        
        print(f'Evaluation results on {section} set:')
        print(f'Spearman correlation: {spearman_corr[0]:.4f}')
        print(f'p-value: {spearman_corr[1]:.6f}')
        print(f'R2: {r2:.4f}')
        print(f'Loss: {loss.item():.6f}')
    
    # Clean up
    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()
    
    # Save results
    df = pd.DataFrame({
        'targets': targets,
        'outputs': outputs,
        'words': dev_words
    })
    
    if not config_file['baseline']:
        df_dir = os.path.join(
            config_file['test_results_path'],
            f'{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_'
            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{section}.csv'
        )
    else:
        df_dir = os.path.join(
            config_file['test_results_path'],
            f'{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_'
            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{section}.csv'
        )
    
    df.to_csv(df_dir, index=False)
    
    # Final correlation check
    spearman_corr = spearmanr(df['targets'], df['outputs'])
    print(f'Final Spearman correlation: {spearman_corr[0]:.4f}')
    print(f'Final p-value: {spearman_corr[1]:.6f}')
    
    return df


def evaluate_on_new(config_file, section=1):
    """Evaluate model on new historical data."""
   
    model_name = config_file['model_name']
    data_name = config_file['data_name']
    device_name = config_file['device_name']
    token_strategy = config_file['token_strategy']
    graph_strategy = config_file['graph_strategy']
    graph_version = config_file['graph_version']
    swow_version = config_file['swow_version']
    fill = config_file['fill']
    add_self_loops = config_file['add_self_loops']
    property_name = config_file['property']
    take_log = config_file['take_log']
    reduce = config_file['reduce']
    train_section = config_file['train_section']
    store_dir = config_file['store_dir']
    
    
    device = torch.device(device_name)
    data_file = os.path.join(store_dir, f'data_{data_name}_{section}_{model_name}.pkl')

    with open(data_file, 'rb') as f:
        d = pickle.load(f)
    
    embedding_data = d['embedding_data']
    vocab_mapping = d['vocab_mapping']
    edge_index = d['edge_index']
    edge_weight = d['edge_weight']
    year = d['year']
    
    
    vocab_mapping_reverse = {v: k for k, v in vocab_mapping.items()}
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
    
    
    if config_file['baseline']:
        model = BasicPropertyPredictor(in_channels, dropout)
    else:
        model = ParameterPropertyPredictor(
            n, in_channels, hidden_channels, out_channels, num_layers,
            num_heads, dropout, encoder_model_name, reduce, add_linear
        )
    
    
    if torch.cuda.is_available():
        model.cuda(device)
    
    torch.autograd.set_detect_anomaly(True)
    
    
    embeddings = embedding_tensor.float().to(device)
    text_edge_index = edge_index.long().to(device)
    text_edge_weight = edge_weight.float().to(device)
    
    
    if not config_file['baseline']:
        model_path = os.path.join(
            config_file['model_saving_dir'],
            f'{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_'
            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt'
        )
    else:
        model_path = os.path.join(
            config_file['model_saving_dir'],
            f'{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_'
            f'graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_'
            f'fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt'
        )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    
    
    with torch.no_grad():
        if not config_file['baseline']:
            output = model(embeddings, text_edge_index, text_edge_weight)
        else:
            output = model(embeddings)
        
        outputs = output.cpu().numpy()
    
    
    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()
    
    
    df = pd.DataFrame({
        'outputs': outputs,
        'words': [vocab_mapping_reverse[i] for i in range(len(embedding_tensor))],
        'year': [year] * len(outputs)
    })
    
    if not config_file['baseline']:
        df_dir = os.path.join(
            config_file['test_results_path'],
            'time_series',
            f'testing_data_{data_name}_{model_name}_{section}_{property_name}_train_section_{train_section}.csv'
        )
    else:
        df_dir = os.path.join(
            config_file['test_results_path'],
            'time_series',
            f'testing_data_{data_name}_{model_name}_{section}_{property_name}_baseline_train_section_{train_section}.csv'
        )
    
    df.to_csv(df_dir, index=False)

def main(config_path='SWOW_prediction/config_features.yml', **kwargs):
    """Main function to execute training or evaluation."""
    
    config = get_config(config_path)
    
    
    eval_section = kwargs['eval_section']
    section = kwargs['section']
    baseline = kwargs['baseline']
    reduce = kwargs['reduce']
    
    config['baseline'] = baseline
    config['reduce'] = reduce
    
    
    os.makedirs(config['test_results_path'], exist_ok=True)
    os.makedirs(os.path.join(config['test_results_path'], 'time_series'), exist_ok=True)
    os.makedirs(config['model_saving_dir'], exist_ok=True)
    
    
    if section != -1:
        config['train_section'] = section
    if eval_section != -1:
        config['eval_section'] = eval_section
    
    
    print(f"Configuration:")
    print(f"- Baseline: {config['baseline']}")
    print(f"- Reduce: {config['reduce']}")
    print(f"- Eval: {config['eval']}")
    print(f"- Train section: {config['train_section']}")
    print(f"- Eval section: {config['eval_section']}")
    
    
    if not config['eval']:
        train(config)
    else:
        evaluate(config, config['eval_section'])


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='SWOW Prediction')
    parser.add_argument('--config_path', type=str, default='SWOW_prediction/config_features.yml',
                        help='Path to configuration file')
    parser.add_argument('--section', type=int, default=-1,
                        help='Training section')
    parser.add_argument('--eval_section', type=int, default=-1,
                        help='Evaluation section')
    parser.add_argument('--baseline', action='store_true',
                        help='Use baseline model')
    parser.add_argument('--reduce', type=str, default='forward',
                        help='Reduction method')
    
    args = parser.parse_args()
    
    main(
        args.config_path,
        section=args.section,
        eval_section=args.eval_section,
        baseline=args.baseline,
        reduce=args.reduce
    )


