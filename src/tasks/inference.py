import torch
from models.bert_model import BertModel
from helpers import get_settings
from transformers import BertTokenizerFast
from collections import defaultdict

settings = get_settings()
tokenizer = torch.load(settings.SAVED_TOKENIZER_PATH)
ids_to_labels = {0 : 'O' , 1 : 'geo' , 2 : 'tim' , 3 : 'org' , 4 : 'per' , 5 : 'gpe'}

model = BertModel(settings.PRETRAINED_MODEL_NAME,settings.TAGS_COUNT).get_model()
state_dict = torch.load(settings.SAVED_MODEL_PATH)
model.load_state_dict(state_dict)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)



def get_tags(sentence) :
    tokenized = tokenizer(sentence)
    words = tokenizer.tokenize(sentence)

    tokenized_sentence = torch.tensor(tokenized['input_ids']).to(device).unsqueeze(0)
    attention_mask =  torch.tensor(tokenized['attention_mask']).to(device).unsqueeze(0)
    output = model(tokenized_sentence,attention_mask = attention_mask)
    tags = torch.argmax(output["logits"],axis=2).cpu()

    wordid_to_tagid = {}
    outputs = defaultdict(list)
    for id , label in zip(tokenized.word_ids(), tags[0]):
        wordid_to_tagid[id] = label.item()

    output = []
    for i in tokenized.word_ids() :
        if i != None :
            tag_id = wordid_to_tagid[i]
            tag_label = ids_to_labels[tag_id]
            output.append(tag_label)

    for label , word in zip(output,words) :
        outputs[label].append(word)

    return outputs