import pickle
import torch
import numpy as np
import gc
from torch.utils.data import Dataset, DataLoader
from SWOW_prediction.utils import *
def get_sentence_encodings(tokenizer, context, max_length = 200):
    encodings = tokenizer(
                context,
                None,
                padding='max_length',
                truncation = 'longest_first',
                add_special_tokens=True,
                return_attention_mask=True,
                max_length=max_length,
                return_offsets_mapping=True
            )
    return encodings

def get_word_position(word, 
                       encodings,
                       context, 
                       lemmas, 
                       tokens, 
                       special_token_id, 
                       tokenizer,
                       model_name = 'bert-base-uncased'):
    
    if len(word.split()) == 1:
        if word not in lemmas:
            return {'has_data': False}
        word_index = lemmas.index(word)
        word_token = tokens[word_index]
    else: #two-gram
        word_indices = [lemmas.index(w) for w in word.split() if w in lemmas]
        if len(word_indices)  == 0:
            
            return {'has_data': False}
        word_token = ' '.join([tokens[i] for i in word_indices])
    if model_name == 'roberta-base':
        token_id = []
        begin_index = context.index(word_token)
        end_index = begin_index + len(word)
        offsets_mapping = encodings['offset_mapping']
        for index, positions in enumerate(offsets_mapping):
            if positions[0] >= begin_index and positions[1] <= end_index:
                token_id.append(encodings['input_ids'][index])
    else:
        token_inputs = tokenizer(word_token)['input_ids']
        token_id = token_inputs[1: token_inputs.index(special_token_id)]

    result = {
        'has_data':True,
        'data': [token_id, word]
    }
    return result


class CueDataset(Dataset):
    
    def __init__(self, encoding_data, position_data):
        super(CueDataset, self).__init__()
        self.encoding_data = encoding_data
        self.position_data = position_data

    def __len__(self):
        return len(self.encoding_data)

    def __getitem__(self, index):

        ids,mask = self.encoding_data[index]['input_ids'], self.encoding_data[index]['attention_mask']
        positions = self.position_data[index]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype= torch.long), positions

def get_sentence_embedding_for_test(sentences, model_name, max_length):
    tokenizer = get_tokenizer(model_name)
    encodings = [get_sentence_encodings(tokenizer, sentence, max_length = max_length) for sentence in sentences]
    special_token = tokenizer.special_tokens_map['sep_token']
    special_token_id = tokenizer.encode(special_token, add_special_tokens = False)[0]
    masks = torch.tensor([encoding['attention_mask'] for encoding in encodings], dtype=torch.long)
    ids = torch.tensor([encoding['input_ids'] for encoding in encodings], dtype=torch.long)
    positions = [list(id).index(special_token_id) for id in ids]
    
    model = get_model(model_name)
    outputs = model(ids, masks)[0]
    outputs = outputs.detach().cpu().numpy()
    outputs = [np.mean(outputs[i, 1: positions[i]], axis = 0) for i in range(len(positions))]
    return outputs


def get_word_embedding(all_encodings, 
                       position_results, 
                       model_name, 
                       og_cues, 
                       device,
                       word_embeddings = {},
                       word_counts = {}):
    '''
    all_encodings: list of encodings (encodings[i]['input_ids'], encodings[i]['attention_maks]), for each sentence
    position_results: list of position information for each encoding, position_results[i][j]: token j position in encodings i
    '''
    def collate_fn(data):
        
        return tuple(zip(*data))
    model = get_model(model_name)
    if device != 'cpu' and torch.cuda.is_available():
        model = model.to(device)
    all_cue_positions = []
    all_encodings_final = []
    
    for i, encoding in enumerate(all_encodings):

        cue_positions = [x['data'] for x in position_results[i] if 
                         x['has_data'] == True 
                         and x['data'][1] in og_cues
                         and len(x['data'][0]) > 0
                         and all([t in encoding['input_ids'] for t in x['data'][0]] )]
        if len(cue_positions) == 0:
            continue
        all_cue_positions.append(cue_positions)
        all_encodings_final.append(encoding)
   
    cue_dataset = CueDataset(all_encodings_final, all_cue_positions)
    cue_dataloader = DataLoader(cue_dataset, batch_size= 16, collate_fn=collate_fn)
    for j, batch in enumerate(cue_dataloader):
        
        ids, mask, sentence_positions = batch
        
        ids = torch.stack(list(ids)).to(device)
        mask = torch.stack(list(mask)).to(device)
       
        outputs = model(ids, mask) 
        outputs = outputs[0] 
        ids = ids.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()
        for b, class_label_output in enumerate(outputs):      
            if 'clip' in model_name: #We no longer have token embeddings, so we use the sentence embeddings
                positions_at_b = sentence_positions[b]
                for positions in positions_at_b:
                    token_ids = positions[0]
                    word = positions[1]
                    if word not in word_counts:
                        total = 1
                        cue_embeddings = np.copy(class_label_output)
                    else:
                        total = word_counts[word] + 1
                        cue_embeddings = word_embeddings[word] + np.copy(class_label_output)
                    word_counts[word] = total
                    word_embeddings[word] = cue_embeddings
            else:
                positions_at_b = sentence_positions[b]
                for positions in positions_at_b:
                    token_ids = positions[0]
                    word = positions[1]
                    positions_2 = list(ids[b]).index(token_ids[0])
                    positions_2 = np.arange(positions_2, positions_2 + len(token_ids))
                    positions_2 = np.array(positions_2,)
                    if np.max(positions_2) >= class_label_output.shape[0]:
                        continue
                    class_label_output_p = class_label_output[positions_2, :].mean(axis = 0)
                    
                    if word not in word_counts:
                        total = 1
                        cue_embeddings = np.copy(class_label_output_p)
                    else:
                        total = word_counts[word] + 1
                        cue_embeddings = word_embeddings[word] + np.copy(class_label_output_p)
                    word_counts[word] = total
                    word_embeddings[word] = cue_embeddings
    model.cpu()
    del model
    gc.collect()
    
    return word_embeddings, word_counts

