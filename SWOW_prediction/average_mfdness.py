
import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict


def get_neighbors_for_words(word_list, year, freqs):
    """
    Get neighboring words and their weights for each word in word_list.
    
    Args:
        word_list: List of words to find neighbors for
        year: The year data to use
        freqs: Dictionary containing vocabulary mappings and edge data
        
    Returns:
        Dictionary mapping each word to a DataFrame of its neighbors
    """
    vocab_mapping = freqs[year]['vocab_mapping']
    edge_index = freqs[year]['edge_index'].T
    edge_weight = freqs[year]['edge_weight']
    vocab_mapping_reverse = {v: k for k, v in vocab_mapping.items()}
    
    # Get indices for all words in word_list
    word_indices = {vocab_mapping[word] for word in word_list if word in vocab_mapping}
    
    
    src_mask = np.isin(edge_index[:, 0], list(word_indices))
    tgt_mask = np.isin(edge_index[:, 1], list(word_indices))
    mask = src_mask | tgt_mask
    
    relevant_edges = edge_index[mask]
    relevant_weights = edge_weight[mask]
    
    # Build neighbor dictionary
    neighbors_dict = defaultdict(list)
    
    for (src, tgt), weight in zip(relevant_edges, relevant_weights):
        src_item = src.item()
        tgt_item = tgt.item()
        
        if src_item == tgt_item:
            continue
            
        if src_item in word_indices:
            neighbors_dict[vocab_mapping_reverse[src_item]].append({
                'word': vocab_mapping_reverse[tgt_item],
                'weight': weight.item()
            })
               
        if tgt_item in word_indices:
            neighbors_dict[vocab_mapping_reverse[tgt_item]].append({
                'word': vocab_mapping_reverse[src_item],
                'weight': weight.item() if isinstance(weight, np.ndarray) else weight
            })
    
    # Convert to DataFrames
    result = {}
    for word in word_list:
        if word in neighbors_dict:
            result[word] = pd.DataFrame(neighbors_dict[word])
        else:
            result[word] = pd.DataFrame(columns=['word', 'weight'])
    
    return result


def add_mfdness(data, year_neighbors, word_year_df):
    """
    Calculate and add moral foundation scores (mfdness) to the dataframe.
    
    Args:
        data: Dataset name
        year_neighbors: Dictionary of word neighbors by year
        word_year_df: DataFrame with words and years
    """
    # Load moral foundation dictionary
    mfd_df = pd.read_csv('./data/mfd2.csv')
    mfd_words = set(mfd_df['word'].unique())
    
    def assign_mfdness(word, year):
        """Calculate mfdness score for a word in a given year"""
        neighbor_df = year_neighbors[year][word]
        if len(neighbor_df) == 0:
            return -1
            
        neighbor_df['mfdness'] = neighbor_df['word'].apply(lambda x: 1 if x in mfd_words else 0)
        sum_weights = np.sum(neighbor_df['weight'])
        
        if sum_weights == 0:
            return -1
            
        average_mfdness = np.sum(neighbor_df['mfdness'] * neighbor_df['weight']) / sum_weights
        return average_mfdness
    
    # Calculate mfdness for each word-year pair
    words_mfdness = []
    total_rows = len(word_year_df)
    
    for i, row in word_year_df.iterrows():
        word = row['words']
        year = row['year']
        mfdness = assign_mfdness(word, year)
        words_mfdness.append(mfdness)
        
        
    
    word_year_df['mfdness'] = words_mfdness
    output_path = f'./data/SWOW_prediction/eval/word_year_mfdness_{data}.csv'
    word_year_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def main(data):
    """
    Main function to process the specified dataset.
    
    Args:
        data: Dataset name ('coha','nyt' or custom dataset)
    """
    print(f"Processing dataset: {data}", flush=True)
    
    # Load time series data
    if data == 'coha':
        ts_df = pd.read_csv('./data/SWOW_prediction/eval/time_series/ts_df.csv')
        years = list(range(2000, 1840, -10))  # 2000, 1990, ..., 1850
    else:
        ts_path = f'./data/SWOW_prediction/eval/time_series/{data}_ts_df.csv'
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"File for the specified data does not exist: {ts_path}")
            
        ts_df = pd.read_csv(ts_path)
        years = list(range(2007, 1986, -1))  # 2007, 2006, ..., 1987
    
    # Load word frequencies for each year
    year_freqs = {}
    for year_index, year in enumerate(years):
        filename = f'./data/SWOW_prediction/data_{data}_{year}_bert-base-uncased.pkl'
        with open(filename, 'rb') as f:
            year_freqs[year] = pickle.load(f)
    
    # Get unique word-year pairs
    word_year_df = ts_df[['words', 'year']].drop_duplicates().reset_index(drop=True)
    
    # Get neighbors for each word in each year
    year_neighbors = {}
    for year in years:
        word_list = word_year_df[word_year_df['year'] == year]['words'].tolist()
        year_neighbors[year] = get_neighbors_for_words(word_list, year, year_freqs)
        print(f"Year {year} processed with {len(year_neighbors[year])} words")
    
    # Calculate and add mfdness scores
    add_mfdness(data, year_neighbors, word_year_df)
    print(f"Processing for {data} completed successfully")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate moral foundation scores for words across years')
    parser.add_argument('--data', type=str, default='coha', help='Dataset name (default: coha)')
    args = parser.parse_args()
    main(args.data)