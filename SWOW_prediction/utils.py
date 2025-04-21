import time
import torch
import numpy as np
import pandas as pd
from scipy import stats
import yaml
from transformers import BertTokenizerFast, BertModel, AutoModel, RobertaModel, RobertaTokenizerFast
from transformers import AutoTokenizer, CLIPTextModelWithProjection

def time_function(func):
    '''Decorator that reports the execution time.'''
  
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(func.__name__, end-start)
        return result
    return wrapper
  

def r2_score(outputs, labels):

    labels_mean = np.mean(labels)
    ss_tot = np.sum((labels - labels_mean) ** 2)
    ss_res = np.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot

    return r2

def r2_score_for_words(outputs, labels, cues):
    res_df = pd.DataFrame({'output':outputs,'label':labels,'cue':cues})
    res_df = res_df.groupby('cue').mean().reset_index()
    r2 = r2_score(np.array(res_df.output), np.array(res_df.label))
    res = stats.spearmanr(np.array(res_df.output), np.array(res_df.label))
    return r2, res[0]

def get_model(model_name):
    if 'xlm' in model_name:
        model = AutoModel.from_pretrained(f'{model_name}')
    if 'roberta' in model_name:
        model = RobertaModel.from_pretrained(model_name)
    elif 'bert' in model_name:
        model = BertModel.from_pretrained(model_name)
    elif 'clip' in model_name:
        model =  CLIPTextModelWithProjection.from_pretrained(f'openai/{model_name}')
    
    return model

def get_tokenizer(model_name):
    if 'xlm' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'{model_name}')
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    elif 'bert' in model_name:
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    elif 'clip' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(f'openai/{model_name}')
        tokenizer.sep_token = tokenizer.eos_token
    
    return tokenizer


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

