from transformers import BertForTokenClassification
import os
from dotenv import load_dotenv
load_dotenv(r"E:\Khaled\Data\Projects\Named-Entity-Recognition-NER-\.env")
PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME")

class BertModel :
    def __init__(self,num_labels) :
        self.model = BertForTokenClassification.from_pretrained(PRETRAINED_MODEL_NAME,num_labels=num_labels)


    def get_model(self) :
        return self.model
