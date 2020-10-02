
#basics
import pandas as pd
import torch
import torch.nn as nn
# Feel free to add any new code to this script
device = torch.device('cuda:1')

def get_tensors(df, max_sample_length, id2word, more_features, w_embed):
    
    embeddings = nn.Embedding((len(id2word) + 1), w_embed)
    
    tokens = list(df['token_id'])
    sentences = [tokens[x:x+(max_sample_length)] for x in range(0, len(tokens),(max_sample_length))] # splitting the entire
    sentences = [tokens[x:x+(max_sample_length)] for x in range(0, len(tokens),(max_sample_length))]

    feat_sentences = []

    for sentence in sentences:
        feat_sentence = []
        for w_n in sentence:
            embed = embeddings(torch.tensor([w_n]))
            features = embed.squeeze()
            
            if more_features:
                word = id2word[w_n]
                if word.isalpha():
                    alpha = torch.tensor([1.])
                else:
                    alpha = torch.tensor([0.])
                w_len = len(word)
                w_len = torch.tensor([float(w_len)])
                features = torch.cat((features, w_len, alpha))
                
            feat_sentence.append(features)
        feat_sentences.append(torch.stack(feat_sentence))
        
    return torch.stack(feat_sentences)
    
def extract_features(data:pd.DataFrame, max_sample_length:int, id2word:dict, more_features=False, w_embed=128):
    
    train_df = data.loc[data.split == 'train']
    test_df = data.loc[data.split == 'test']
    val_df = data.loc[data.split == 'val']
    
    train_X = get_tensors(train_df, max_sample_length, id2word, more_features=more_features, w_embed=w_embed)
    test_X = get_tensors(test_df, max_sample_length, id2word, more_features=more_features, w_embed=w_embed)
    val_X = get_tensors(val_df, max_sample_length, id2word, more_features=more_features, w_embed=w_embed)
    
    return train_X.to(device), val_X.to(device), test_X.to(device)
    
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    pass