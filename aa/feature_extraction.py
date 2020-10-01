
#basics
import pandas as pd

# Feel free to add any new code to this script


def extract_features(data:pd.DataFrame, max_sample_length:int):
    
    tokens = list(data_df['token_id'])
    sentences = [tokens[x:x+(max_sample_length)] for x in range(0, len(tokens),(max_sample_length)]
    
    
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    pass