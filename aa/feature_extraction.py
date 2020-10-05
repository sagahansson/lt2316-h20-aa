import pandas as pd
import torch
import torch.nn as nn
device = torch.device('cuda:1')

def get_tensors(df, max_sample_length, id2word, more_features, embeddings):
    # constructs the feature tensors
    # helper to extract_features
    
    tokens = list(df['token_id'])
    sentences = [tokens[x:x+(max_sample_length)] for x in range(0, len(tokens),(max_sample_length))] # splitting the entire token list into a nested list where each inner list represents a sentence
    sentences = [tokens[x:x+(max_sample_length)] for x in range(0, len(tokens),(max_sample_length))]
    feat_sentences = [] # will be nested list where each inner list represents a sentence; each inner list contains feature tensors for each word

    for sentence in sentences:
        feat_sentence = [] # will contain features for one sentence
        
        for w_n in sentence: # 
            embed = embeddings(torch.tensor([w_n]))
            features = embed.squeeze()
            
            if more_features: # getting word length and 1 or 0 depending on whether the word contains any non-alphabetical characters
                word = id2word[w_n]
                if word.isalpha():
                    alpha = torch.tensor([1.])
                else:
                    alpha = torch.tensor([0.])
                w_len = len(word)
                w_len = torch.tensor([float(w_len)])
                features = torch.cat((features, alpha, w_len))
                
            feat_sentence.append(features)
        feat_sentences.append(torch.stack(feat_sentence))
        
    return torch.stack(feat_sentences)
    
def extract_features(data:pd.DataFrame, max_sample_length:int, id2word:dict, more_features=False, w_embed=128):
    
    # gets feature tensor for each word in each split
    # feature tensor consists of word embeddings (where w_embed decided dimensionality of embeddings) and,
    # if more_features=True, word length and whether the word contains only alphabetical characters (1) or not (0)
    
    embeddings = nn.Embedding(len(id2word), w_embed)
    
    # splitting the data into train, test, val
    train_df = data.loc[data.split == 'train']
    test_df = data.loc[data.split == 'test']
    val_df = data.loc[data.split == 'val']
    
    # retrieving feature tensors with the help of get_tensors
    train_X = get_tensors(train_df, max_sample_length, id2word, more_features=more_features, embeddings=embeddings)
    test_X = get_tensors(test_df, max_sample_length, id2word, more_features=more_features, embeddings=embeddings)
    val_X = get_tensors(val_df, max_sample_length, id2word, more_features=more_features, embeddings=embeddings)
    
    return train_X.to(device), val_X.to(device), test_X.to(device)