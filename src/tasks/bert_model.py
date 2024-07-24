from transformers import BertForTokenClassification
import os

class BertModel :
    def __init__(self,model_name,num_labels) :
        self.model = BertForTokenClassification.from_pretrained(model_name,num_labels=num_labels)


    def get_model(self) :
        return self.model
