
#basics
import random
import pandas as pd
import torch


from nltk.tokenize import RegexpTokenizer
import os
import xml.etree.ElementTree as ET
import re
from random import choice
#from random import shuffle
from collections import Counter

pd.options.mode.chained_assignment = None

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cuda:2')

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
        # fetches a list of absolute paths to all xml files in subdirectories in rootdir
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
        punctuation = "-,.?!:;"
        tokenizer = RegexpTokenizer("\s|:|;", gaps=True)
        tokenized = tokenizer.tokenize(s.lower())
        tokenized = [word.strip(punctuation) if word[-1] in punctuation else word for word in tokenized] # removing punctuation if it's the last char in a word
        span = list(tokenizer.span_tokenize(s)) # gets the pythonic span i e (start, stop_but_not_including)
        new_span = []
        for tpl in span:
            new_span.append((tpl[0], (tpl[1]-1))) # to get non-pythonic span i e (start,last_char)
        return new_span, tokenized

    def pad_sentences(self, sentences, max_length):
        data_df_list = []
        for sent in sentences:
            split = sent[0][4]
            pad_len = max_length - len(sent) # how many padding token are needed to make len(sent) == max_length
            pad_rows = pad_len * [(0, 0, 0, 0, split)] # list of padding rows made to fit dataframe ie four 0's are for 'sent_id', 'token_id', 'char_start', 'char_end'
            sent.extend(pad_rows)                      # if sent_id is specified it gets stuck in get_random_sample
            data_df_list.extend(sent)
        return data_df_list
    
    def open_xmls(self, fileList):

        #vocab = []
        data_df_list = [] 
        ner_df_list = []
        all_sentences = []
        self.ner2id = {
            'other/pad' : 0,
            'drug'      : 1,
            'drug_n'    : 2,
            'group'     : 3, 
            'brand'     : 4
        }
        self.word2id = {}

        for file in fileList:
            tree = ET.parse(file)
            root = tree.getroot()
            for sentence in root:
                one_sentence = []
                sent_id = sentence.attrib['id']
                sent_txt = sentence.attrib['text']
                if sent_txt == "": # to exclude completely empty sentences i e DDI-DrugBank.d228.s4 in Train/DrugBank/Fomepizole_ddi.xml
                    continue
                #unique_w = list(set(tokenized))
                #vocab.extend(unique_w)
                if 'test' in file.lower():
                    split = 'test'
                else:
                    split = choice(["train", "train", "train", "train", "val"]) # making it a 20% chance that it's val and 80% chance that it's train
                char_ids, tokenized = self.string_to_span(sent_txt)
                for i, word in enumerate(tokenized): # creating data_df_list
                    if word in self.word2id.keys(): # make into function instead? # creating word2id, vocab
                        word = self.word2id[word]
                    else:
                        w_id = 1 + len(self.word2id) # zero is pad
                        self.word2id[word] = w_id
                        word = w_id
                    word_tpl = (sent_id, word, int(char_ids[i][0]), int(char_ids[i][1]), split) # one row in data_df 
                    one_sentence.append(word_tpl)
                for entity in sentence: # creating the ner_df_list
                    if entity.tag == 'entity':
                        ent_txt = (entity.attrib['text']).lower()
                        ent_type = (entity.attrib['type']).lower()
                        ent_type = self.ner2id[ent_type]
                        char_offset = entity.attrib['charOffset']
                        char_span = (re.sub(r"[^0-9]+",' ', char_offset)).split(' ') # substituting everything in char_offset that is not a number with a space
                                                                                     # to be able to split on spaces 
                        if len(char_span) > 2:
                            char_pairs = (list(zip(char_span[::2], char_span[1::2])))
                            for pair in char_pairs:
                                entity_tpl = (sent_id, ent_type, int(pair[0]), int(pair[1]))
                                ner_df_list.append(entity_tpl)
                        else:
                            ent_start_id, ent_end_id = char_span
                            ent_txt_one = ent_txt    
                            entity_tpl = (sent_id, ent_type, int(ent_start_id), int(ent_end_id))
                            ner_df_list.append(entity_tpl)
                all_sentences.append(one_sentence)

        
        self.max_sample_length = max([len(x) for x in all_sentences])
        data_df_list = self.pad_sentences(all_sentences, self.max_sample_length)
        #vocab = list(sorted(set(vocab))) # behöver du verkligen vocab??? 
        #return vocab, data_df_list, ner_df_list
        return data_df_list, ner_df_list
    
    def _parse_data(self, data_dir):
        
        allFiles = self.get_paths(data_dir)
        #vocab_list, data_df_list, ner_df_list  = self.open_xmls(allFiles)
        data_df_list, ner_df_list = self.open_xmls(allFiles)
        
        self.data_df = pd.DataFrame(data_df_list, columns=['sentence_id', 'token_id', 'char_start_id', 'char_end_id', 'split'])
        self.ner_df = pd.DataFrame(ner_df_list, columns=['sentence_id', 'ner_id', 'char_start_id', 'char_end_id']) # ner_id = entity type
        self.word2id['padding'] = 0 # do this another way
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2ner = {v:k for k, v in self.ner2id.items()}
        
        self.vocab = list(self.word2id.keys())


    def get_ners(df):

        data_sentence_ids = list(df.sentence_id)
        data_start = list(df.char_start_id)
        data_end = list(df.char_end_id)
        data_token = list(df.token_id)
        data_tpls = [(data_sentence_ids[i], data_token[i], data_start[i], data_end[i]) for i, elem in enumerate(data_sentence_ids)]

        labellist = []
        for tpl in data_tpls: # for every word in data_df, give it a label
            data_sent_id, data_token, data_char_start, data_char_end = tpl
            tpl = (data_sent_id, data_char_start, data_char_end)
            if data_token == 0: # if it's padding
                label = 0 # add 0 label for padding/other
                labellist.append(label)
                continue
            for i, ner in enumerate(self.ner_tpls): # enumerate ensures that we get correct label for row
                ner_sent_id, ner_char_start, ner_char_end = ner
                if tpl == ner: # flytta upp utanför loopen #if ner in data_tpls: till line 16
                    label = self.ner_id[i]
                    continue
                if data_sent_id == ner_sent_id:
                    if (data_char_start >= ner_char_start) and (data_char_end <= ner_char_end):
                        label = self.ner_id[i]
                    else:
                        label = 0
                else:
                    pass
            labellist.append(label)
        self.labellists = [labellist[x:x+102] for x in range(0, len(labellist),102)] # CHANGE TO SELF.MAX_SAMPLE_LENGTH
        
        
    def get_y(self):
        #self.number_samples = round(len(self.vocab)/self.max_sample_length)
        
        ner_sentence_ids = list(ner_df.sentence_id)
        ner_start = list(ner_df.char_start_id)
        ner_end = list(ner_df.char_end_id)
        self.ner_id = list(ner_df.ner_id)
        self.ner_tpls = [(ner_sentence_ids[i], ner_start[i], ner_end[i]) for i, elem in enumerate(ner_sentence_ids)]
        
        test_df = self.data_df.loc[self.data_df.split == 'test']
        train_df = self.data_df.loc[self.data_df.split == 'train']
        val_df = self.data_df.loc[self.data_df.split == 'val']
        
        self.test_y = torch.Tensor(self.get_y(test_df)).to(device)
        self.train_y = torch.Tensor(self.get_y(train_df)).to(device)
        self.val_y = torch.Tensor(self.get_y(val_df)).to(device)
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



