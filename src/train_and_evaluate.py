from models.bert_model import BertModel
from helpers.config import get_settings
from transformers import BertTokenizer ,BertTokenizerFast
from tasks import process_data
import torch
import pandas as pd
import os
from tasks import Loader , train_model , evaluate_model
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
os.environ["TOKENIZERS_PARALLELISM"] = "false"

settings = get_settings()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = BertModel(settings.PRETRAINED_MODEL_NAME,settings.TAGS_COUNT).get_model()
model = model.to(device)
print("Model imported successfully.")

bert_tokenizer = BertTokenizerFast.from_pretrained(settings.PRETRAINED_MODEL_NAME)
preprocessed_data = process_data(settings.RAW_DATA_PATH,bert_tokenizer,settings.PREPROCESSED_DATA_PATH)
#preprocessed_data = pd.read_csv(settings.PREPROCESSED_DATA_PATH)
print("Data has been preprocessed and saved successfully.")

# splitting dataset
training_data , validation_data = train_test_split(preprocessed_data,test_size=.15)
validation_data , testing_data = train_test_split(validation_data,test_size=.5)
print("Data has been splitted successfully.")

batch_size=4

training_data_loader = Loader(sentences_tokens=training_data["tokens"],tags=training_data["new_tags"],batch_size = batch_size ,
                              tokenizer=bert_tokenizer ,max_length=settings.MAX_LENGTH,shuffle=True).get_data_loader()
validation_data_loader = Loader(sentences_tokens=validation_data["tokens"],tags=validation_data["new_tags"],batch_size = batch_size ,
                              tokenizer=bert_tokenizer ,max_length=settings.MAX_LENGTH,shuffle=True).get_data_loader()
testing_data_loader = Loader(sentences_tokens=testing_data["tokens"],tags=testing_data["new_tags"],batch_size = batch_size ,
                              tokenizer=bert_tokenizer ,max_length=settings.MAX_LENGTH,shuffle=True).get_data_loader()
print("Data loaders has been created successfully.")

optimizer = AdamW(model.parameters(),lr=0.00002)
training_steps = settings.EPOCHS * len(training_data_loader)
warmup_steps = int(.2 * training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=warmup_steps,num_training_steps=training_steps)
print("scheduler & optimizer has been created successfully.")

best_score = 0
for epoch in range(settings.EPOCHS) :
    print(f"Epoch {epoch}")
    training_loss , training_accuracy , training_f1_score = train_model(model,
                                                                        scheduler=scheduler,optimizer=optimizer,device=device,
                                                                        data_loader=training_data_loader)

    validation_loss , validation_accuracy , validation_f1_score = evaluate_model(model, device=device,data_loader=validation_data_loader)

    print(f"Training loss : {training_loss} , Training accuracy : {training_accuracy} , Training f1_score : {training_f1_score} ")
    print(f"Validation loss : {validation_loss} , Validation aaccuracy : {validation_accuracy} , Validation f1_score : {validation_f1_score} ")


    if validation_f1_score > best_score :
        best_score = validation_f1_score
        torch.save(model.state_dict(),settings.SAVED_MODEL_PATH)

        #Training loss : 0.06268392809153332 , Training accuracy : 0.977556984102655 , Training f1_score : 0.9776607531271987
        #Validation loss : 0.10360086464652947 , Validation aaccuracy : 0.9654256847622441 , Validation f1_score : 0.9655944372190728 
        


model.eval()
testing_loss , testing_accuracy , testing_f1_score = evaluate_model(model, device=device,data_loader=testing_data_loader)
print(f"Testing loss : {testing_loss} , Testing accuracy : {testing_accuracy} , Testing f1_score : {testing_f1_score} ")

# Testing loss : 0.05091910494123618 , Testing accuracy : 0.9827252657213738 , Testing f1_score : 0.9827213417574071