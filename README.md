# LT2316 H20 Assignment A1

Name: Saga Hansson

## Notes on Part 1.

In Part 1, I made a few helper functions as to improve readability and make debugging easier. The main function, *_parse_data*, calls two functions: *get_paths* and *parse_xmls*.

*Get_paths*: simply gets a list all of files that have the file extension .xml in the given directory. This list of files is then passed as an argument to *parse_xmls*.

*Parse_xmls*: gets all the information needed from the list of .xml files produced in *get_paths*, and organises the data before converting to dataframes. Since I wanted to iterate through all the data as few times as possible, the function is on the longer side. After looking at each .xml file, the function looks at each sentence within a file. If the attribute 'text' is empty, meaning it isn't actually a sentence, the function will not consider it. Subsequently, the split of train/test/validation is made. If the path name contains "test", all sentences in that file will belong to the test set. If not, the function will give each document (and the sentences it contains) a 20/80 chance of belonging to validation and train sets, respectively. After that, the helper function *string_to_span* is called.
  - *String_to_span*: tokenizes and gets the word span, i.e. what the index of the starting character and the end character is. I decided to use RegexpTokenizer from nltk to be able to split the words on certain characters, but keep other letter-charcter combinations intact. For example, in some of the data, different drugs would be enumerated, separated only by semicolons and no space. On the other hand, I wanted to keep contractions such as "can't" and drug names such as "1-methyl-4-phenyl-1,2,3,6-tetrahydropyridine" intact. Therefore, the splitting characters are \s, ; and :. Punctuation at the end of words are removed, although the word spans do include punctuation to allow the spans to be as accurate as possible.


Following tokenization, the *word2id* dictionary is created with the tokens from *string_to_span*. The rows for *data_df* are then created as tuples with sentence id's, id's from word2id, character start and ends from *string_to_span*, and split category. The tuples are added to the list *one_sentence* (which represents one sentence), which are then added to the list *all_sentences*. 
The function then retrieves the entities from the .xml sentences, after which the entity category and character start and ends are retrieved. The entity information (sentence id, entity type, character start and end) is added to a tuple, which is added to the list *ner_df_list*. 
The function continus by getting the maximum sentence length through comparing *all_sentences*. Finally, all sentences are padded by calling the function *pad_sentences*:
  - *Pad_sentences*: the padding is fairly straight-forward: all five values that are to be columns in *data_df*, except split, are set to 0. The split category is taken from the split category that the rest of the sentence has. 

*Parse_xmls* returns the padded *data_df_list* and the *ner_df_list*.

*_Parse_data*: In *_parse_data*, the tuple lists ner_df and data_df are converted to dataframes, and some dictionaries are reversed. 

*Get_y*: constructs a tensor for each split category (train/test/validation) containing labels for each sentence. Firstly, the information contained in *ner_df* is converted to a list of tuples (sentence_id, start_char, end_char) and a list of ner_id's. The dataframe *data_df* is then split into train, test and validation, after which the helper function *get_ners* is called for each dataframe split.
    - *Get_ners*: to begin with, the data contained in the dataframe passed to *get_ners* is converted in a similar way to data conversion in *get_y*: a list of tuples is made, each tuple consisting of (sentence_id, token_id, char_start, char_end). The lists of tuples (*data_tpls* and *ner_tpls*) are then compared: if the token_id is 0, which is the padding token, the label given is also 0; if the (sentence_id, start_char, end_char) from *data_tpls* and *ner_tpls* are the exact same, the label belonging to that ner_tpl is given; if the sentence id's in both tuples match and the character span of the tuple from *data_tpls* can be within the span of the tuple from *ner_tpls*, the label belonging to that ner_tpl is given; in all other cases, the label 0 is given. An example for the most complicated case, the second to last one, is when the ner text is "retinyl acetate", which, due to the tokenization, would be split into to different tokens for *data_df*, i.e. "retinyl" and "acetate". If "retinyl acetate" has the character span 0, 14, then "retinyl" would have the character span 0, 6, which doesn't match, but is within the span of "retinyl acetate". *Get_ners* returns a list of all labels (*labellist*), and a nested list (*nested_lists*), divided into sentences, of labels.

Back in *get_y*, for each split category, *nested_lists* is converted to a tensor. 

## Notes on Part 2.

*Extract_features*: For the feature extraction, the function *extract_features* creates word embeddings for the entire vocabulary, after which the *data_df* is split into train, test and validation. *Extract _features* takes five arguments: *data*, *max_sample_length*, *id2word*, *more_features* and *w_embed*. *Data* is a dataframe, needed to create the split categories; *max_sample_length* is the maxiumum sentences lenght, *id2word* is a dictionary mapping integers to words; *more_features* is a boolean deciding whether to "just" use word embeddings (False) or to add word length and whether the word only contains alphabetical characters to the features (False); *w_embed* is the dimensionality of the word embeddings. *Get_tensors* is then called with the arguments *df* (either train, test or validation), *max_sample_length*, *id2word*, *more_features*, *embeddings* (the word embeddings created in *extract_features*). 
    - *Get_tensors*: creates a tensor of the dimensions (no_of_sentences, no_words_per_sentence, no_features_per_word). To start with, a nested lists, where the inner lists represents sentences, is created by taking the *'token_id'* column from the dataframe in question and putting every *max_sample_length* words in one list. That is, if the *max_sample_length* = 10, it will take the first 10 token id's and put them in a list, followed by taking the next 10 token id's and putting them in a lists, etc. The nested list is then iterated over to retrieve embeddings and features for each token id. 
In *extract_features*, each tensor returned by *get_tensors* is put on the gpu and returned.

*Plot_split_ner_distribution*: In *plot_ner_distribution*, a dictionary containing label and label counts per split category is created, excluding all mentions of the label 0 since it clouds the distribution of the other categories significantly. The three dictionaries are then turned into a dataframe, which is plotted.


## Notes on Part Bonus.

*Plot_sample_length_distribution*: The function *plot_sample_distribution* mainly uses Pandas methods to retrieve the needed data. Firstly, the *token_id* column from *data_df* is accessed, and tokens with the value 0 are not included. This is done as 0 represents padding, which would be the dominating category if included. The remaining data is then made into a Pandas Series, since it simplifies converting the final counts into a list. Lastly, the Series is grouped by *'sentence_id'*, counted (making it count the number of times one sentence id occurs = sentence length), and converted into a list. Some plotting settings are adjusted, after which the list of counts is plotted.

*Plot_ner_sample_distribution*: *Plot_ner_sample_distribution* is almost the same as *plot_sample_distribution*, the main differences being that 0's aren't removed (since they don't exist in ner_df) and that *.cumsum()* is used instead of *.count()*. The series, turned dictionary, is then turned into a list (*ners_per_sentence*), since *.to_list()* for some reason doesn't seem to work with *.cumsum()*. Following that, the number of sentences that don't contain any ners are counted, after which a list (*no_ners*) containing as many 0's as there are sentences without ners is created. That list is then added to *ners_per_sentence*, followed by some plotting settings being adjusted, and finally *ners_per_sentence* being plotted.

