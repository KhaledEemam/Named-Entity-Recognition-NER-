import pandas as pd
import os
from dotenv import load_dotenv
from transformers import BertTokenizerFast
load_dotenv(r"E:\Khaled\Data\Projects\Named-Entity-Recognition-NER-\.env")

# Importing data.
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
data =  pd.read_csv(RAW_DATA_PATH,encoding="unicode_escape")

# Fill null values with the previous non null value for each column.
data.ffill(axis=0,inplace=True)

# Get Bert tokenizer.
tokenizer_model_name = os.getenv("PRETRAINED_MODEL_NAME")
bert_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_model_name)

# Map each tag to an index.
labels_to_ids = {'O': 0,'B-nat': 1 , 'I-nat' : 2, 'B-eve' : 3, 'I-eve': 4, 'B-art': 5, 'I-art' : 6, 'B-org': 7,
 'I-org': 8 , 'B-tim': 9, 'I-tim' : 10, 'B-per': 11 , 'I-per': 12, 'B-geo': 13 , 'I-geo': 14 , 'B-gpe': 15 ,
 'I-gpe': 16 }

# Get map for index to tag.
ids_to_labels = {}
for label , id in labels_to_ids.items() :
    ids_to_labels[id] = label

# Map each tag label into it's corresponding index
data["Tag"] = data["Tag"].map(labels_to_ids)

# Modify dataframe by getting row for each sentence, sentence column for the sentence words & Tags for the labels indices.
grouped_data = data.groupby(["Sentence #"]).apply(lambda x : pd.Series({"sentence" : x["Word"].tolist() , "Tags" :x["Tag"].tolist()})).reset_index()

grouped_data.drop("Sentence #",inplace=True,axis=1)

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
            # Check if the word tag starts with (B-) and convert it into (I-)
            if label % 2 ==1 :
                label += 1

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

# Apply predefined function to tokenize each sentence and align the tags
grouped_data["new_tags"] = grouped_data.apply(lambda x : tokenize_and_align_labels(bert_tokenizer,x) , axis = 1)

# Drop column "Tags"
grouped_data.drop("Tags",inplace=True,axis=1)

# Save preprocessed dataframe
PREPROCESSED_DATA_PATH = os.getenv("PREPROCESSED_DATA_PATH")
grouped_data.to_csv(PREPROCESSED_DATA_PATH)
print("Data has been preprocessed and saved to the new path.")
