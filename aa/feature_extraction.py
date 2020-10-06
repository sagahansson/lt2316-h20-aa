import pandas as pd
import torch
import torch.nn as nn
device = torch.device('cuda:1')

def get_tensors(df, max_sample_length, id2word):
    # constructs the feature tensors
    # helper to extract_features
    
    tokens = list(df['token_id'])
    sentences = [tokens[x:x+(max_sample_length)] for x in range(0, len(tokens),(max_sample_length))] # splitting the entire token list into a nested list where each inner list represents a sentence
    feat_sentences = [] # will be nested list where each inner list represents a sentence; each inner list contains feature tensors for each word
    new_id = max(id2word.keys()) + 1
    
    for sentence in sentences:
        feat_sentence = [] # will contain features for one sentence
        last_w_idx = len(sentence) - 1
        for i, w_n in enumerate(sentence): 
            # getting 1 or 0 depending on whether the word contains any non-alphabetical characters
            word = id2word[w_n]
            if word.isalpha():
                alpha = torch.tensor([1.])
            else:
                alpha = torch.tensor([0.])
            # word length feature
            w_len = len(word)
            w_len = torch.tensor([float(w_len)])
            if i != 0: 
                prec_word = torch.tensor([float(sentence[i-1])])
            else: # if its the 1st word, it wont have a preceding word but gets a dummy id
                prec_word = torch.tensor([float(new_id)]) # new id to indicate no preceding word
            if i != last_w_idx:
                succ_word = torch.tensor([float(sentence[i+1])])
            else: # if its the last word, it wont have a succeeding word but gets a dummy id
                succ_word = torch.tensor([float(new_id)])
            
            features = torch.cat((alpha, w_len, prec_word, succ_word))

            feat_sentence.append(features)
        feat_sentences.append(torch.stack(feat_sentence))   
    return torch.stack(feat_sentences)
    
def extract_features(data:pd.DataFrame, max_sample_length:int, id2word:dict):
    
    # gets feature tensor for each word in each split
    # feature tensor consists of word length and whether the word contains only alphabetical characters (1) or not (0), and preceding and succeeding word tokens
    
    # splitting the data into train, test, val
    train_df = data.loc[data.split == 'train']
    test_df = data.loc[data.split == 'test']
    val_df = data.loc[data.split == 'val']
    
    # retrieving feature tensors with the help of get_tensors
    train_X = get_tensors(train_df, max_sample_length, id2word)
    test_X = get_tensors(test_df, max_sample_length, id2word)
    val_X = get_tensors(val_df, max_sample_length, id2word)
    
    return train_X.to(device), val_X.to(device), test_X.to(device)