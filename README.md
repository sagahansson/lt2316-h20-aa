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

*_Parse_data*: In *_parse_data*, the tuple lists ner_df and data_df are converted to dataframes, and some dictionaries are reversed. 

## Notes on Part 2.

*to be continued*
*fill in notes and documentation for part 2 as mentioned in the assignment description*

## Notes on Part Bonus.

*fill in notes and documentation for the bonus as mentioned in the assignment description, if you choose to do the bonus*
