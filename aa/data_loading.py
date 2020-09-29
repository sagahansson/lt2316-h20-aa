
#basics
import random
import pandas as pd
import torch


from nltk.tokenize import RegexpTokenizer
import os
import xml.etree.ElementTree as ET
import re
from random import shuffle
from collections import Counter

pd.options.mode.chained_assignment = None

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        #self.data_df=self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)

    
    def get_paths(self, rootdir):
        # fetches a list of absolute paths, given a dir with xml files
        # BEHÖVS FÖR OPEN_XMLS
        file_paths = []

        for folder, _, files in os.walk(rootdir):
            for filename in files:
                if filename.endswith('xml'):
                    file_paths.append(os.path.abspath(os.path.join(folder, filename)))
        return file_paths
    
    def string_to_span(self, s):
        # creates a tokenized version and a span version of a string
        # BEHÖVS FÖR OPEN_XMLS
        #s = (re.sub(r"[^A-Za-z\s]",'',s)).lower() # removes all non-alphanumerical characters LOL gör inte det
        tokenizer = RegexpTokenizer("[\w'-]+|[^\w\s]+") # tokenizes words and punctuation except hyphens in compound words and apostrophes
        tokenized = tokenizer.tokenize(s.lower())
        span = list(tokenizer.span_tokenize(s)) # gets the pythonic span i e (start, stop_but_not_including)
        new_span = []
        for tpl in span:
            new_span.append((tpl[0], (tpl[1]-1))) # to get non-pythonic span i e (start,last_char)
        return new_span, tokenized

    def open_xmls(self, fileList):

        vocab = []
        data_df_list = [] 
        ner_df_list = []
        ent2id = {
            'drug'   : 0,
            'drug_n' : 1,
            'group'  : 2, 
            'brand'  : 3
        }

        for file in fileList:
            tree = ET.parse(file)
            root = tree.getroot()
            for sentence in root:
                sent_id = sentence.attrib['id']
                sent_txt= sentence.attrib['text']
                char_ids, tokenized = self.string_to_span(sent_txt)
                unique_w = list(set(tokenized))
                vocab.extend(unique_w)
                for i, word in enumerate(tokenized): # creating data_df_list
                    if 'test' in file.lower():
                        split = 'test'
                    else:
                        split = 'train/dev'
                    word_tpl = (sent_id, word, int(char_ids[i][0]), int(char_ids[i][1]), split) # one row in data_df 
                    data_df_list.append(word_tpl)

                for entity in sentence: # creating the ner_df_list
                    if entity.tag == 'entity':
                        ent_txt = (entity.attrib['text']).lower()
                        ent_type = (entity.attrib['type']).lower()
                        ent_type = ent2id[ent_type]
                        char_offset = entity.attrib['charOffset']
                        char_span = (re.sub(r"[^0-9]+",' ', char_offset)).split(' ')

                        if len(char_span) > 2:
                            char_pairs = (list(zip(char_span[::2], char_span[1::2])))
                            for pair in char_pairs:
                                entity_tpl = (sent_id, ent_type, int(pair[0]), int(pair[1])) # one row in ner_df
                                ner_df_list.append(entity_tpl)
                        else:
                            ent_start_id, ent_end_id = char_span
                            ent_txt_one = ent_txt    

                            entity_tpl = (sent_id, ent_type, int(ent_start_id), int(ent_end_id)) # one row in ner_df

                            ner_df_list.append(entity_tpl)

        vocab = list(sorted(set(vocab)))
        return vocab, data_df_list, ner_df_list
    
    
    def word2int(self, vocabList):
        # tar hela vocabet från open_xmls och ger tillbaka en dict med word : index
        return {w:i for i,w in enumerate(sorted(vocabList))}
    
    
    def get_token_ids(self, tokensList, vocab2idDict):
        # efter data_df_list är en dataframe och efter vocab har blivit w2i
        # tar data_df-kolumnen för tokens och dicten från w2i
        # fetches token id from vocab2id dict
        
        return [vocab2idDict[w] for w in tokensList] 
    
    def _parse_data(self, data_dir):
        
        allFiles = self.get_paths(data_dir)
        vocab_list, data_df_list, ner_df_list  = self.open_xmls(allFiles)
        
        data_df = pd.DataFrame(data_df_list, columns=["sentence_id", "token", "char_start_id", "char_end_id", "split"])
        self.ner_df = pd.DataFrame(ner_df_list, columns=["sentence_id", "ner_id", "char_start_id", "char_end_id"])
        w2i = self.word2int(vocab_list)
        token_ids = self.get_token_ids(data_df['token'], w2i) # fetching token ids for all tokens in the dataframe
        data_df.insert(1, 'token_id', token_ids) # inserting token ids 
        data_df = data_df.drop(columns=['token']) # removing 'token' column
        test_df = data_df.loc[data_df.split == 'test'] # splitting df horizontally into test and train/dev
        traindev_df = data_df.loc[data_df.split != 'test']
        dev_len = len(test_df) # size of dev/val set is decided from how big the test set is
        train_len = len(traindev_df) - dev_len # train is basically whatever is left
        traindev_df.drop(columns=['split'])
        train_dev = ['train'] * train_len # creating list containing train_len instances of 'train' 
        dev = ['dev'] * dev_len # same as train_dev, but here instances of 'dev'
        train_dev.extend(dev)
        shuffle(train_dev)
        traindev_df.loc[:, 'split'] = train_dev
        self.data_df = (traindev_df.append(test_df)).reset_index(drop=True)
        self.id2ner = {
            0 : 'drug',
            1 : 'drug_n',
            2 : 'group',
            3 : 'brand',
            4 : 'other'
        }
        sent_dict = Counter(list(data_df.sentence_id)) # counting occurences of sentence_id in data df = len of sentences
        self.max_sample_length = max(sent_dict.values())
        self.id2word = {v:k for k, v in w2i.items()}
        self.vocab = list(w2i.keys())


    def get_y(self):
        self.number_samples = round(len(self.vocab)/self.max_sample_length)
        
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        pass

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        pass


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        pass


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        pass


    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass



