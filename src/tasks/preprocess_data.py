import pandas as pd
from helpers import get_settings

# Align the number of tokens with the number of tags for each sentence in the dataset.
def align_labels_with_tokens(labels,word_ids) :
    prev_word = None
    new_labels = []

    for id in word_ids :
        # Current token came from word index different than the previous token.
        if id != prev_word :
            prev_word = id
            label = -100 if prev_word == None else labels[id]
            new_labels.append(label)
        # Current token has no word index -special word added by the tokenizer-.
        elif prev_word == None :
            new_labels.append(-100)
        # Current token came from the same word index as the prev one.
        else :
            label = labels[id]

            new_labels.append(label)

    return new_labels

# Tokenize each sentence and fix the misalignment between the tokens and tags for each row
def tokenize_and_align_labels(tokenizer,row) :
    # Tokenize sentence
    tokenized_sentence = tokenizer(row["sentence"], is_split_into_words = True)
    # Extract the source words idx for each token
    word_ids = tokenized_sentence.word_ids()
    # Get the tags ids
    labels_ids = row["Tags"]
    # Apply predefined function to align tokens with labels
    new_labels = align_labels_with_tokens(labels_ids,word_ids)
    return  new_labels

def process_data(RAW_DATA_PATH,bert_tokenizer,PREPROCESSED_DATA_PATH) :
    # Importing data.
    data =  pd.read_csv(RAW_DATA_PATH,encoding="unicode_escape")

    # Fill null values with the previous non null value for each column.
    data.ffill(axis=0,inplace=True)
    
    # Due to the imbalanced data, I'll select only 5 tags as labels for the NER task. 
    # These 5 tags will be chosen because they represent the most frequent entities in the dataset.
    # All other tags will be converted into 'O' (Outside) labels.

    # Map each tag to an index.
    labels_to_ids = {'O': 0, 'B-geo': 1 , 'I-geo': 1 ,'B-tim': 2, 'I-tim' : 2,'B-org': 3,
                     'I-org': 3 , 'B-per': 4 , 'I-per': 4, 'B-gpe': 5 ,'I-gpe': 5 ,
                     'B-nat': 0 , 'I-nat' : 0, 'B-eve' : 0, 'I-eve': 0, 'B-art': 0, 'I-art' : 0 }


    # The 5 selected labels will be treated equally, with both B- (Beginning) and I- (Inside) entity tags considered for each label.
    # Get map for index to tag.
    ids_to_labels = {0 : 'O' , 1 : 'geo' , 2 : 'tim' , 3 : 'org' , 4 : 'per' , 5 : 'gpe'}

    # Map each tag label into it's corresponding index
    data["Tag"] = data["Tag"].map(labels_to_ids)

    # Modify dataframe by getting row for each sentence, sentence column for the sentence words & Tags for the labels indices.
    grouped_data = data.groupby(["Sentence #"]).apply(lambda x : pd.Series({"sentence" : x["Word"].tolist() , "Tags" :x["Tag"].tolist()})).reset_index()

    grouped_data.drop("Sentence #",inplace=True,axis=1)

    # Apply predefined function to tokenize each sentence and align the tags
    grouped_data["new_tags"] = grouped_data.apply(lambda x : tokenize_and_align_labels(bert_tokenizer,x) , axis = 1)

    grouped_data["tokens"] = grouped_data["sentence"].apply(lambda x : bert_tokenizer(x, is_split_into_words = True)["input_ids"])

    # Drop column "Tags"
    grouped_data.drop("Tags",inplace=True,axis=1)


    # Save preprocessed dataframe
    grouped_data.to_csv(PREPROCESSED_DATA_PATH)
    return grouped_data
