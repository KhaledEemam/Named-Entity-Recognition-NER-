from .BaseController import BaseController
from models import ResponseSignal

class DataController(BaseController) :
    def __init__(self) :
        super().__init__()

    def validate_sentence_length(self,sentence) :
        sentence_length = len(sentence.split(' '))

        if sentence_length > self.settings.MAX_LENGTH :
            return False , ResponseSignal.SENTENCE_LENGTH_EXCEEDED.value
        
        return True , ResponseSignal.SENTENCE_VALIDATED_SUCCESS.value