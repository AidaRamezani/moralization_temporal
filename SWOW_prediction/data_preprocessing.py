import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from torch_geometric.utils import add_self_loops, negative_sampling, remove_self_loops
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from SWOW_prediction.custom_tokenizers import *
from SWOW_prediction.utils import *
from typing import List
import pickle
import os
import networkx as nx
import io
from zipfile import ZipFile


def get_coha_path():
    coha_path = 'data/COHA.zip'
    return coha_path


def get_coha_articles(year):
    coha_path = get_coha_path()
    encoding = 'ISO-8859-1'
    articles = []
    article_text = []
    article_geners = []
    with ZipFile(coha_path) as cf_zip:
        files = cf_zip.namelist()
        file = [f for f in files if str(year) in f][0]
        zfiledata = io.BytesIO(cf_zip.read(file))
        with ZipFile(zfiledata) as fzip:

            for article_name in fzip.namelist():
                with fzip.open(article_name) as f:

                    article = f.readlines()[1:]
                    article = [s.decode(encoding).split('\t')[1] for s in article]
                    articles.append(article)
                    article_text.append(" ".join(article))
                    genre = article_name[:article_name.index('_')]
                    article_geners.append(genre)

    return articles, article_text,article_geners


def get_swow_data(version = 1) -> pd.DataFrame:
    df = pd.read_csv(f'data/SWOWEN/responses_R{version}.csv')
    df = df.groupby(['cue','response'])['total'].mean().reset_index()
    return df

class CueDataset(Dataset):
    def __init__(self, cue_data):
        super(CueDataset, self).__init__()
        self.cue_data = cue_data

    def __len__(self):
        return len(self.cue_data)

    def __getitem__(self, index):

        ids,mask, token_ids, _,_ = self.cue_data[index]

        positions = ids.index(token_ids[0])
        positions = torch.arange(positions, positions + len(token_ids))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype= torch.long), positions


def get_edge_index_from_edges(edges, vocab_mapping = None,
                              fill_value = 'add',
                              add_self_loop = True,
                              nodes_total: dict = None,
                              normalize = False,
                              sample_negatives = False,
                              negative_sample_num: int = None):


    edge_index = []
    edge_weight = []
    if vocab_mapping == None:
        vocab_mapping = {}
    for n1, node_links in edges.items():
        total = 1 if not normalize else nodes_total[n1]
        node_edges = []
        node_weights = []
        for n2 in node_links:
            if n1 not in vocab_mapping:
                vocab_mapping[n1] = len(vocab_mapping)
            if n2 not in vocab_mapping:
                vocab_mapping[n2] = len(vocab_mapping)
            i1 = vocab_mapping[n1]
            i2 = vocab_mapping[n2]
            node_edges.append([i1, i2])
            node_weights.append(node_links[n2])
        if len(node_weights) == 0:
            continue
        if add_self_loop:
            i1 = vocab_mapping[n1]
            node_edges.append([i1, i1])
            if fill_value == 'add':
                node_weights.append(sum(node_weights))
            elif fill_value == 'mean':
                node_weights.append(sum(node_weights)/len(node_weights))
            elif fill_value == 'max':
                node_weights.append(max(node_weights))
            else:
                raise ValueError('fill_value must be add, mean or max')

        node_weights = [w/total for w in node_weights]
        edge_index += node_edges
        edge_weight += node_weights

    edge_index = torch.tensor(edge_index).T
    edge_weight = torch.tensor(edge_weight)
    if sample_negatives:
        negative_edge_index = negative_sampling(edge_index, num_neg_samples = negative_sample_num)
        edge_index = torch.cat([edge_index, negative_edge_index], dim = 1)
        edge_weight = torch.cat([edge_weight, torch.zeros(negative_edge_index.shape[1])])
        if not add_self_loop:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    n = len(edge_weight)
    return edge_index, edge_weight, vocab_mapping, n

def get_coha_data(year):
    from SWOW_prediction.get_data import get_coha_articles
    articles, article_text, article_genres = get_coha_articles(year)
    return article_text

def get_nyt_data(year,data_path):
    year_data_path = os.path.join(data_path, f'{year}-lemmatized.pkl')
    d = pickle.load(open(year_data_path,'rb'))
    articles = [x['article'] for x in d]
    return articles 


def get_data(data_name, data_path, **kwargs) :
    assert data_name in ['coha','nyt'], 'Data name must be either coha or nyt'
    
    if data_name == 'coha':
        years = []
        data_features = kwargs['data_features'][data_name]
        if 'year' in data_features:
            years = data_features['year']
        year = years[0] #The training year   
        return get_coha_data(year)
    elif data_name == 'nyt':
        years = []
        data_features = kwargs['data_features'][data_name]
        if 'year' in data_features:
            years = data_features['year']
        year = years[0]
        return get_nyt_data(year, data_path)

@time_function
def sentensize_articles(articles_text):
    all_sentences = []
    for article in articles_text:
        if pd.isna(article) or len(article) < 10:
            continue
        sentences = sent_tokenize(article)
        all_sentences += sentences
    
    return all_sentences

@time_function
def get_word_count(tokenized_data, 
                   sentences, 
                   swow_cues = None,
                   add_two_gram = True, 
                   strategy = 'frequency', 
                   k = 10000):
    word_count = dict()
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    for i , tokenized_sentence in enumerate(tokenized_data):
        for j, token in enumerate(tokenized_sentence):
            if token in stop or len(token) < 3:
                continue
            if token in word_count:
                word_count[token] += 1
            else:
                word_count[token] = 1
            if add_two_gram and j < len(tokenized_sentence) - 1:
                two_gram = token + ' ' + tokenized_sentence[j + 1]
                if two_gram not in sentences[i]:
                    continue
                if two_gram in word_count:
                    word_count[two_gram] += 1
                else:
                    word_count[two_gram] = 1
    if strategy == 'frequency':
        cues = sorted(word_count.keys(), key = lambda x : -word_count[x])[:k]  
        if swow_cues != None:
            cues += [w for w in swow_cues if w not in cues]
    else:
        pass
    return word_count, cues



@time_function
def co_occurrence(articles,cues, two_gram = False, two_grams : List = None):
    window_size = 1
    total = 0
    col_totals = dict()
    vocab = sorted(cues)
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for k, tokenized_data in enumerate(articles):
        
        n = len(tokenized_data)
        tokens = [t for t in tokenized_data]
        if two_gram and two_grams != None:
            tokens = []
            i = 0
            while i < n:
                token = tokenized_data[i]
                if i == n - 1 or (token + ' ' + tokenized_data[i + 1] not in two_grams):
                    tokens.append(token)
                    i += 1
                else:
                    tokens.append(token + ' ' + tokenized_data[i + 1])
                    i += 2
        for i , token in enumerate(tokens):
            if token == '.':
                continue
            next_token = tokens[i + 1 : min(i + 1 + window_size, len(tokens))]
            for t in next_token:
                if t == '.':
                    continue
                if t in cues and token in cues:
                    key = tuple(sorted([t, token]))
                    df.loc[key[0], key[1]] = df.loc[key[0], key[1]] + 1
                    df.loc[key[1], key[0]] = df.loc[key[1], key[0]] + 1
                col_totals[t] = col_totals.setdefault(t, 0) + 1
                col_totals[token] = col_totals.setdefault(token, 0) + 1
                total += 1    
    return df, vocab, total, col_totals





def get_data_graph(tokenized_data,
                   cues, strategy,
                   graph_version = 1,
                   vocab_mapping = None,
                   fill = 'add',
                   add_self_loop = True,
                   n = 25,
                   two_gram = False,

                   two_grams = None):
    
    def get_pmi(word, df, col_totals,total, cues, n = 25):
        words = np.array(df.columns)
        word_occurrence = df.loc[word, :].values
        denominator = np.array([col_totals[k] * col_totals[word] / total for k in words])
        word_occurrence = np.log(word_occurrence / denominator)
        indices = np.where(word_occurrence > 0 )[0]
        sorted_occurrences = -np.sort(-word_occurrence[indices])[:n]
        total_value = sum(sorted_occurrences)
        sorted_indices = np.argsort(-word_occurrence[indices])[:n]
        sorted_occurrences = sorted_occurrences / total_value
        wanted_words = words[indices][sorted_indices]
        word_occurrence = dict(zip(wanted_words, sorted_occurrences))
        return word_occurrence


    edges = {}
    nodes_total = {}
    assert strategy == 'ppmi', 'Only ppmi is supported'
    
    
    df, vocab, total, col_totals = co_occurrence(tokenized_data,cues, two_gram, two_grams)
    for w in cues:
        if w not in col_totals:
            col_totals[w] = 0
            edges[w] = {}
            nodes_total[w] = 0
    
    for i, w in enumerate(cues):
        
        if col_totals[w] == 0:
            continue
        edges[w] = get_pmi(w, df, col_totals, total, cues, n)
        nodes_total[w] = sum([v for k, v in edges[w].items()])
    
    edge_index, edge_weight, vocab_mapping, _ = \
            get_edge_index_from_edges(edges, vocab_mapping, fill, add_self_loop,nodes_total, normalize=True, sample_negatives=False)
    return edge_index, edge_weight, vocab_mapping



def get_two_grams_from_swow(version, lemmatizer):
    df = get_swow_data(version)
    df['response'] = [lemmatizer.lemmatize(r) for r in df.response]
    df['cue'] =  [lemmatizer.lemmatize(c) for c in df.cue]
    two_grams = [w for w in df['cue'] if len(w.split()) == 2]
    return two_grams

def get_swow_words(version, lemmatizer):
    df = get_swow_data(version)
    df['cue'] =  [lemmatizer.lemmatize(str(c).lower()) for c in df.cue]
    words = list(df.cue.unique())   
    return words


def get_encodings(tokenized_data, token_list, lemmatized_tokens, sentences,cues,two_grams, tokenizer,model_name,max_length= 200):
    from SWOW_prediction.data_preprocessing_utils import get_sentence_encodings, get_word_position
    special_token = tokenizer.special_tokens_map['sep_token']
    special_token_id = tokenizer.encode(special_token, add_special_tokens = False)[0]
    
    sentence_positions = []
    all_encodings = []
    for i , tokenized_sentence in enumerate(tokenized_data):
        
        sentence_encodings = None
        
        for j, token in enumerate(tokenized_sentence):
            if token not in cues:
                continue
            if sentence_encodings == None:
                sentence_positions.append([])
                sentence_encodings = get_sentence_encodings(tokenizer, sentences[i], max_length = max_length)
                all_encodings.append(sentence_encodings)
            try:
                i_th_positions = get_word_position(token, 
                    sentence_encodings,
                    sentences[i], 
                    lemmatized_tokens[i], 
                    token_list[i], 
                    special_token_id, 
                    tokenizer,
                    model_name = model_name)
                sentence_positions[-1].append(i_th_positions)
            except NameError:
                
                print("position not retrieved")
            if  j < len(tokenized_sentence) - 1:
                two_gram = token + ' ' + tokenized_sentence[j + 1]
                if two_gram not in two_grams:
                    continue
                if two_gram not in sentences[i]:
                    continue
                try: 
                    i_th_positions = get_word_position(two_gram, 
                    sentence_encodings,
                    sentences[i], 
                    lemmatized_tokens[i], 
                    token_list[i], 
                    special_token_id, 
                    tokenizer,
                    model_name = model_name)
                    sentence_positions[-1].append(i_th_positions)
                except NameError:
                    print("position not retrieved")
        if len(sentence_positions) == 0:
            continue
        if len(sentence_positions[-1]) == 0:
            sentence_positions = sentence_positions[:-1]
            all_encodings = all_encodings[:-1]
    
    return sentence_positions, all_encodings



def store_encoding_data(model_name,
                                year = 2000,
                           max_length = 200,
                           data_name = 'coha',
                           **kwargs):
    

    data_path = kwargs['data_path']
    swow_version = kwargs['swow_version']
    token_strategy = kwargs['token_strategy']
    two_gram = kwargs['two_gram']



    lemmatizer = WordNetLemmatizer()
    two_grams = get_two_grams_from_swow(swow_version, lemmatizer) if two_gram else []
    swow_cues = get_swow_words(swow_version, lemmatizer)
    new_one_grams, new_two_grams = get_two_grams()
    two_grams += new_two_grams
    two_grams = list(set(two_grams))
    data_config = {'data_features':{data_name:{'year':[year]}}}

    data = get_data(data_name, data_path,data_features = data_config['data_features'])
    sentences = sentensize_articles(data)
    tokenizer = MyTokenizer()
    token_list, lemmatized_tokens, tokenized_data = tokenizer.tokenize_batch(sentences, True, 'simple')
    word_count, cues = get_word_count(tokenized_data,sentences,swow_cues,two_gram,token_strategy, k = 10000)
    cues = [w for w in cues if len(w) >= 3 and w != 'nt' and w in word_count and  word_count[w] >= 50] +\
            [w for w in swow_cues if len(w) == 2 or w not in word_count] 
    cues += new_one_grams + two_grams
    cues = list(set(cues))

    tokenizer = get_tokenizer(model_name)
    
    sentence_positions, all_encodings = get_encodings(tokenized_data,
                                                    token_list,
                                                    lemmatized_tokens,
                                                    sentences,
                                                    cues,
                                                    two_grams,
                                                    tokenizer,
                                                    model_name,
                                                    max_length)
    return sentence_positions, all_encodings
                                                   
def store_embedding_data(model_name,
                                year = 2000,
                           max_length = 200,
                           data_name = 'coha',
                           **kwargs):

                          
    
    from SWOW_prediction.data_preprocessing_utils import get_word_embedding

    data_path = kwargs['data_path']
    swow_version = kwargs['swow_version']
    token_strategy = kwargs['token_strategy']
    two_gram = kwargs['two_gram']
    device_name = kwargs['device_name']
    
    lemmatizer = WordNetLemmatizer()
    two_grams = get_two_grams_from_swow(swow_version, lemmatizer) if two_gram else []
    swow_cues = get_swow_words(swow_version, lemmatizer)
    new_one_grams, new_two_grams = get_two_grams()
    two_grams += new_two_grams
    two_grams = list(set(two_grams))
    data_features = {'data_features':{data_name:{'year':[year]}}}
    data = get_data(data_name, data_path,data_features = data_features['data_features'])
    sentences = sentensize_articles(data)
    tokenizer = MyTokenizer()
    _, _, tokenized_data = tokenizer.tokenize_batch(sentences, True, 'simple') 
    word_count, cues = get_word_count(tokenized_data,sentences,swow_cues,two_gram,token_strategy,
                                        k = 10000)
    cues = [w for w in cues if len(w) >= 3 and w != 'nt' and w in word_count and  word_count[w] >= 50] +\
            [w for w in swow_cues if len(w) == 2 or w not in word_count] 
    cues += new_one_grams + two_grams
    cues = list(set(cues))
    tokenizer = get_tokenizer(model_name)
    
    sentence_positions, all_encodings = pickle.load(
                 open(f'data/SWOW_prediction/{data_name}_{year}_encodings_{model_name}.pkl','rb'))
                                             
    final_cue_embeddings, final_word_counts = get_word_embedding(all_encodings, 
            sentence_positions, 
            model_name, 
            cues, 
            device = device_name,
            word_embeddings = {},
            word_counts = {})

    embedding_data = {w: final_cue_embeddings[w] / final_word_counts[w] for w in final_cue_embeddings.keys()}
    return final_cue_embeddings, final_word_counts, embedding_data

    
def store_textual_data(model_name,
                                year = 2000,
                           max_length = 200,
                           data_name = 'coha',
                            **kwargs):
                          
    
    token_strategy = kwargs['token_strategy']
    two_gram = kwargs['two_gram']
    years = year_range[data_name]
    i = years.index(year)
    
    lemmatizer = WordNetLemmatizer()
    two_grams = get_two_grams_from_swow(swow_version, lemmatizer) if two_gram else []
    swow_cues = get_swow_words(swow_version, lemmatizer)
    new_one_grams, new_two_grams = get_two_grams()
    two_grams += new_two_grams
    two_grams = list(set(two_grams))
    data_features = {'data_features':{data_name:{'year':[year]}}}
    data = get_data(data_name, data_path,data_features = data_features['data_features'])

    sentences = sentensize_articles(data)
    tokenizer = MyTokenizer()
    _,_, tokenized_data = tokenizer.tokenize_batch(sentences, True, 'simple') #tokenized data is lemmatized
    word_count, cues = get_word_count(tokenized_data,sentences,swow_cues,two_gram,token_strategy,
                                        k = 10000)
    cues = [w for w in cues if len(w) >= 3 and w != 'nt' and w in word_count and  word_count[w] >= 50] +\
            [w for w in swow_cues if len(w) == 2 or w not in word_count] 
    cues += new_one_grams + two_grams
    cues = list(set(cues))

    tokenizer = get_tokenizer(model_name)
    vocab_mapping = None
    _, _, embedding_data = pickle.load(open(f'data/SWOW_prediction/{year}_{data_name}_emb_{model_name}.pkl','rb'))
    cues = [c for c in cues if c in embedding_data] 
    
    edge_index, edge_weight, vocab_mapping = get_data_graph(tokenized_data,
                                                                cues,
                                                                graph_strategy,
                                                                graph_version,
                                                                fill = fill,
                                                                add_self_loop=add_self_loops,
                                                                n = node_neighbors,
                                                                two_gram=two_gram, two_grams=two_grams,
                                                                )
    
    cues = [c for c in cues if c in vocab_mapping]
    embedding_data = {w: e for w, e in embedding_data.items() if w in cues}
    return {
        'embedding_data': embedding_data,
        'vocab_mapping': vocab_mapping,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'cues': cues,
        'index':i
    }
    
def save_word_count(data_name = 'coha', model_name = 'bert-base-uncased'):
    word_count_df = pd.DataFrame()
    words = []
    counts = []
    years = []
    year_range = range(1850, 2010, 10) if data_name == 'coha' else range(1987, 2008)
    for year in year_range:
        (final_cue_embeddings, final_word_counts, embedding_data) =\
        pickle.load(open(f'data/SWOW_prediction/{year}_{data_name}_emb_{model_name}.pkl','rb'))
        words += list(final_word_counts.keys())
        counts += list(final_word_counts.values())
        years += [year] * len(final_word_counts)
    word_count_df['word'] = words
    word_count_df['count'] = counts
    word_count_df['year'] = years
    word_count_df.to_csv(f'data/SWOW_prediction/{data_name}_word_count.csv', index = False)
    

def get_textual_data_input_with_sections(model_name,
                           max_length = 200,
                           data_name = '',
                           **kwargs):
    
    years = kwargs['data_features'][data_name]['year']
    all_data = {}
    for i, year in enumerate(years):
        
        store_dir = f'data/SWOW_prediction/data_{data_name}_{i}_{model_name}.pkl'
        assert os.path.exists(store_dir), f'{store_dir} does not exist. Run data preprocessing first.'
        d = pickle.load(open(store_dir, 'rb'))
        embedding_data = d['embedding_data']
        vocab_mapping = d['vocab_mapping']
        edge_index = d['edge_index']
        edge_weight = d['edge_weight']
        cues = d['cues']
        all_data[i] = [embedding_data, edge_index, edge_weight, year, vocab_mapping,cues]

    return all_data

def construct_property_dataset(model_name,
                               max_length = 200,
                               data_name_store = '',
                               data_path = '',
                               token_strategy = 'frequency',
                               graph_strategy = 'ppmi',
                               device_name = 'cpu',
                               graph_version = 2,
                               swow_version = 1,
                               fill = 'add',
                               add_self_loops_cmd = True,
                               node_neighbors = 25,
                               normalize = True,
                               take_log = False,
                               sample_negative = False,
                               negative_sample_num = None,
                               feature = 'link',
                               two_gram = True,
                               **kwargs
                               ):
    
    assert feature in ['previous_link','polarity'], 'Feature must be either previous_link (moral relevance) or polarity (moral polarity)'
    data_name = data_name_store
    year = kwargs['data_features'][data_name]['year'][0]
    stored_data_dir = f'./data/SWOW_prediction/property_{feature}_{data_name_store}_{year}_{model_name}_{token_strategy}_{graph_strategy}_{graph_version}_{swow_version}_{fill}_{add_self_loops_cmd}_{two_gram}_{node_neighbors}_{max_length}_{take_log}.pkl'
    

    response_mfd = pd.read_csv(f'data/SWOWEN/moralized_v{swow_version}_mfd.csv')
    

    if feature == 'previous_link': #Moral relevance
        property_data = response_mfd[[feature,'cue']]
        property_data = property_data.loc[~pd.isna(property_data[feature])]
        zeros = property_data.loc[property_data[feature] == 0]
        property_data = property_data.loc[property_data[feature] != 0] 
        if take_log:
            property_data[feature] = np.log(property_data[feature] * 100 + 1) #We avoid negative values
        if sample_negative:
            zeros = zeros.sample(negative_sample_num, random_state=42)
            property_data= pd.concat([property_data, zeros], ignore_index=True)
        
    elif feature == 'polarity': #Moral polarity
       
        property_data = response_mfd[[feature,'cue','pos_score','neg_score']]
        property_data = property_data.loc[~pd.isna(property_data[feature])]
        zeros = property_data.loc[property_data[feature] == 0.5] #pos == neg
        property_data = property_data.loc[property_data[feature] != 0.5]  #pos != neg
        if take_log:
            property_data[feature] = np.log(property_data[feature] * 100 + 1) #We avoid negative values
            zeros[feature] = np.log(zeros[feature] * 100 + 1) #This is because our zeros are 0.5, so we should convert them to log form.
        if sample_negative:
            zeros = zeros.sample(negative_sample_num, random_state=42)
            property_data = pd.concat([property_data, zeros], ignore_index=True)
    
    cues = list(property_data.cue)
    
    all_data = \
        get_textual_data_input_with_sections(model_name,
                               max_length,
                               data_name,
                               data_path,
                               token_strategy,
                               graph_strategy,
                               device_name,
                               graph_version,
                               fill,
                               add_self_loops_cmd,
                               node_neighbors,
                               swow_version,
                               two_gram,
                               data_features = kwargs['data_features'])
    
    vocab_mapping_train = all_data[0][-2] #Modern time point
    cues = [c for c in cues if c in vocab_mapping_train]
    vocab_mapping_reverse = dict(zip(vocab_mapping_train.values(), vocab_mapping_train.keys()))
    targets = {w: property_data.loc[property_data.cue == w][feature].iloc[0] for w in cues}
    target_tensor = torch.tensor([property_data.loc[property_data.cue == w][feature].iloc[0] for w in cues])
    target_node_index = torch.tensor([vocab_mapping_train[w] for w in cues])
    indices = np.arange(len(targets))
    random_states = [42, 1231, 523, 432, 21]
    all_train, test = train_test_split(indices, test_size = 0.2,random_state=42, shuffle=True)
    swow_data_sets_all = []
    for random_state in random_states:
        train, dev = train_test_split(all_train, test_size = 0.25,random_state=random_state, shuffle=True)
        
        all_words = np.array(cues)
        train_words = all_words[train]
        dev_words = all_words[dev]
        test_words = all_words[test]
        train_target = target_tensor[train]
        dev_target = target_tensor[dev]
        test_target = target_tensor[test]

        swow_data_sets = {'train': [target_node_index[train], train_target,train_words ],
                        'dev': [target_node_index[dev], dev_target, dev_words],
                        'test': [target_node_index[test], test_target, test_words]}
        swow_data_sets_all.append(swow_data_sets)

    embedding_data, text_edge_index, text_edge_weight, year, vocab_mapping, data_cues = all_data[0]
    stored_data_dir = f'./data/SWOW_prediction/property_{feature}_{data_name_store}_{year}_{model_name}_{token_strategy}_{graph_strategy}_{graph_version}_{swow_version}_{fill}_{add_self_loops_cmd}_{two_gram}_{node_neighbors}_{max_length}_{take_log}.pkl'
    
    embedding_tensor = torch.tensor([embedding_data[vocab_mapping_reverse[i]] 
                                        if vocab_mapping_reverse[i] in embedding_data 
                                        else torch.zeros(dimension)
                                        for i in range(len(vocab_mapping_reverse))
                                        ])
    dimension = list(embedding_data.values())[0].shape[0]
   
    edge_indices = [k for k in range(text_edge_index.shape[1]) if 
                        vocab_mapping_reverse[int(text_edge_index[0,k])] in cues and
                        vocab_mapping_reverse[int(text_edge_index[1,k])] in cues
                        ]
    text_edge_index = text_edge_index[:, edge_indices]
    text_edge_weight = text_edge_weight[edge_indices]
    textual_data = {'embedding': embedding_tensor,
                    'train': [text_edge_index, text_edge_weight]}
    pickle.dump([swow_data_sets_all, textual_data, vocab_mapping], open(stored_data_dir, 'wb'))
    
    train_data = (swow_data_sets_all, textual_data, vocab_mapping)
    

    return train_data




def get_two_grams(): #A curated set of words and bigrams 
    tech_df = pd.read_csv('data/moralization_terms/technologies_inventions_brands.csv')
    tech_cues = list(set(tech_df['Terms']))
    president_df = pd.read_csv('data/president.csv')
    president_df['pre_name'] = [n.split()[-1].lower() for n in president_df['PRESIDENT']]
    president_df['vice_name'] = [n.split()[-1].lower()  if not pd.isna(n) else n for n in president_df['VICE PRESIDENT']]
    president_df['first_name'] = [n.split()[0].lower() for n in president_df['PRESIDENT']]
    president_df['year1'] = [y.split('-')[0] if '-' in y else y for y in president_df['YEAR']]
    president_df['year2'] = [y.split('-')[1]  if '-' in y else y for y in president_df['YEAR']]

    president_two_grams = [first_name + ' ' + last_name for first_name, last_name in zip(president_df.first_name, 
                                                                                         president_df.pre_name)]
    president_two_grams += ['president' + ' ' + last_name for last_name in president_df.pre_name]
    president_cues =   list(set(president_two_grams))
    
    civil_unrest_df = pd.read_csv('data/moralization_terms/civil_unrest.csv')
    civil_terms = []
    for i, row in civil_unrest_df.iterrows():
        row_terms = row['Terms'].split(',')
        new_rows = [s.lower().strip() for s in row_terms]
        civil_terms += new_rows
    
    disease_df = pd.read_csv('data/moralization_terms/diseases.csv')
    disease_cues = []
    for i, row in disease_df.iterrows():
        row_terms = row['Terms'].split(',')
        new_rows = [s.lower().strip() for s in row_terms]
        disease_cues += new_rows
    
    epidemic_df = pd.read_csv('data/moralization_terms/epidemics.csv')
    epidemic_cues = []
    for i, row in epidemic_df.iterrows():
        row_terms = row['Terms'].split(',')
        new_rows = [s.lower().strip() for s in row_terms]
        epidemic_cues += new_rows
    
    political_figures_df = pd.read_csv('data/moralization_terms/political_figures.csv')
    political_cues = []
    for i, row in political_figures_df.iterrows():
        row_terms = row['Terms'].split(',')
        new_rows = [s.lower().strip() for s in row_terms]
        political_cues += new_rows
    
    world_event_df = pd.read_csv('data/moralization_terms/world_event.csv')
    event_cues = []
    for i, row in world_event_df.iterrows():
        if pd.isna(row['Terms']):
            continue
        row_terms = row['Terms'].split(',')
        new_rows = [s.lower().strip() for s in row_terms]
        event_cues += new_rows
    
    wars_df = pd.read_csv('data/moralization_terms/wars_conflicts.csv')
    wars_cues = []
    for i, row in wars_df.iterrows():
        row_terms = row['Terms'].split(',')
        new_rows = [s.lower().strip() for s in row_terms]
        wars_cues += new_rows
    

    serial_killer_df = pd.read_csv('data/moralization_terms/serial_killer.csv')
    names = list(serial_killer_df['Name'])
    serial_killer_cues = []
    for n in names:
        last_name_first_name = n.split(',')
        if len(last_name_first_name) == 2:
            last_name = last_name_first_name[0].lower().strip()
            first_name = last_name_first_name[1].lower().strip()
            first_name = first_name.split(' ')[0] #We only take the first name, not the middle name
            serial_killer_cues += [first_name + ' ' + last_name]
    all_cues = tech_cues + \
        president_cues + \
            civil_terms + \
                disease_cues + \
                    epidemic_cues + \
                        political_cues + \
                            event_cues + \
                                serial_killer_cues + \
                                    wars_cues

    all_cues = list(set(all_cues))
    new_one_grams = [x for x in all_cues if len(x.split()) == 1]
    new_two_grams = [x for x in all_cues if len(x.split()) == 2]

    pickle.dump(
        {'tech_cues': tech_cues,
         'president_cues': president_cues,
         'civil_terms': civil_terms,
         'disease_cues': disease_cues,
         'epidemic_cues': epidemic_cues,
         'political_cues': political_cues,
         'event_cues': event_cues,
         'serial_killer_cues': serial_killer_cues,
            'wars_cues': wars_cues,
         }, open('data/moralization_terms/cues.pkl','wb')
    )

    return new_one_grams, new_two_grams


if __name__ == '__main__': 
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2000)
    parser.add_argument('--model', type=str,default='bert-based-uncased')
    parser.add_argument('--data', type=str,default='coha')
    parser.add_argument('--function', type=str,default='encoding')
    parser.add_argument('--length', type=int,default=200)
    parser.add_argument('--node_neighbors', type=int,default=100)
    parser.add_argument('--data_path', type=str,default='./data/COHA.zip')
    args = parser.parse_args()
    model_name = args.model
    data_name = args.data
    max_length = args.length
    data_function = args.function
    node_neighbors = args.node_neighbors
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise argparse.ArgumentTypeError(f"data path:{data_path} is not a valid path")
    
    os.makedirs('./data/SWOW_prediction', exist_ok = True)

    if torch.cuda.is_available():
        device_name = 'cuda'
    else:
        device_name = 'cpu'
    year_range = { #change here if using your own data
        'coha': list(np.arange(2000, 1840, -10)),
        'nyt':  list(np.arange(2007, 1986, -1))
        
    }
    years = year_range[data_name]
    
    
    data_features = {data_name:{'year':[args.year]}}

    #keeping hyperparameters fixed throughout

    token_strategy = 'frequency'
    graph_strategy = 'ppmi'
    graph_version = 2
    swow_version = 1
    fill = 'add'
    
   

    if data_function == 'encoding':

        sentence_positions,all_encodings = store_encoding_data(model_name,
                                    year = args.year,
                            max_length = max_length,
                            data_name = data_name,
                            data_path = data_path,
                            token_strategy = token_strategy,
                            graph_strategy = graph_strategy,
                            device_name = device_name,
                            graph_version = graph_version,
                            fill = fill,
                            add_self_loops = True,
                            node_neighbors = node_neighbors,
                            swow_version = swow_version,
                            two_gram = True
                            )

        pickle.dump((sentence_positions, all_encodings),
                 open(f'data/SWOW_prediction/{data_name}_{args.year}_encodings_{model_name}.pkl','wb'))
     
    elif data_function == 'embedding':
        
        final_cue_embeddings, final_word_counts, embedding_data = \
                           store_embedding_data(model_name,
                                                        year = args.year,
                                                        max_length = max_length,
                                                        data_name = data_name,
                                                        data_path = data_path,
                                                        token_strategy = token_strategy,
                                                        graph_strategy = graph_strategy,
                                                        device_name = device_name,
                                                        graph_version = graph_version,
                                                        fill = fill,
                                                        add_self_loops = True,
                                                        node_neighbors = node_neighbors,
                                                        swow_version = swow_version,
                                                        two_gram = True,
                            )
        pickle.dump((final_cue_embeddings, final_word_counts, embedding_data), 
                open(f'data/SWOW_prediction/{args.year}_{data_name}_emb_{model_name}.pkl','wb'))
    
    elif data_function == 'graph':    
        textual_data = store_textual_data(model_name,
                                    year = args.year,
                            max_length = max_length,
                            data_name = data_name,
                            data_path = data_path,
                            token_strategy = token_strategy,
                            graph_strategy = graph_strategy,
                            device_name = device_name,
                            graph_version = graph_version,
                            fill = fill,
                            add_self_loops = True,
                            node_neighbors = node_neighbors,
                            swow_version = swow_version,
                            two_gram = True,
                            )
        i = textual_data['index']
        pickle.dump(textual_data, open(f'data/SWOW_prediction/data_{data_name}_{i}_{model_name}.pkl','wb'))
