import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import warnings
warnings.filterwarnings('ignore')
import re
import argparse
from nltk import WordNetLemmatizer, ToktokTokenizer


if __name__ == '__main__':
    #please download data from https://data.stanford.edu/congress_text and place in the folder ./data/hein-daily
    #Reference: https://data.stanford.edu/congress_text 
    #Reference https://stacks.stanford.edu/file/druid:md374tz9962/codebook_v4.pdf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--congress_id', type = int, default = 97)
    args = parser.parse_args()
    
    congress_id = str( args.congress_id)
    congress = '0' + congress_id if len(congress_id) == 2 else congress_id

    lemmatizer = WordNetLemmatizer()
    tokenizer = ToktokTokenizer()
    speech_mapping = {}
    
    speech_mapping = {}
    file_dir = f'./data/hein-daily/descr_{congress}.txt'
    with open(file_dir) as f:
        lines = f.readlines()
        headers = lines[0].split('|')
        id_index = headers.index('speech_id')
        date_index = headers.index('date')
        for line in lines[1:]:
            split_line = line.split('|')
            speech_id = split_line[id_index]
            date = split_line[date_index]
            year = int(date[:4])
            speech_mapping[speech_id] = year
            
    keyword_file = './data/hein-daily/keywords.txt'
    topic_keywords = {}
    with open(keyword_file) as f:
        lines = f.readlines()[1:]
        for line in lines:
            topic, word = line.split('|')
            topic_keywords[word.strip()] = topic.strip()
        
    #bigrams
    bigram_file = './data/hein-daily/topic_phrases.txt'
    with open(bigram_file) as f:
        lines = f.readlines()[1:]
        for line in lines:
            topic, word = line.split('|')
            
            b1, b2 = word.split()
            
            topic_keywords[word.strip()] = topic.strip()
            topic_keywords[b1] = topic.strip()
            topic_keywords[b2] = topic.strip()
    all_lemmas  ={}
    word_counts = {}
    speech_file = f'./data/hein-daily/speeches_{congress}.txt'
    with open(speech_file,encoding = 'iso8859_2') as f:
        lines = f.readlines()[1:]
        for i, line in enumerate(lines):
            # print(line)
            if i % 10000 == 0:
                print(i, len(lines))
            line_split = line.split('|')
            if len(line_split) != 2:
                continue
            speech_id, speech = line_split
            if speech_id not in speech_mapping:
                continue
            date = speech_mapping[speech_id]
            words = set(tokenizer.tokenize(speech.lower()))
            for word in words:
                if any([keyword in word for keyword in topic_keywords]):
                    lemma = all_lemmas.get(word, lemmatizer.lemmatize(word))
                    word_counts.setdefault(lemma, {})
                    word_counts[lemma].setdefault(date, 0)
                    word_counts[lemma][date] += 1
                    
    pickle.dump({
        'word_counts': word_counts,
        'all_lemmas': all_lemmas
    }, open(f'./data/hein-daily/word_counts_{congress}.pkl', 'wb'))
            



    
            
        

                    
            
        

