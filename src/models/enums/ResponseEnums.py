from enum import Enum

class ResponseSignal(Enum) :

    SENTENCE_VALIDATED_SUCCESS = "Sentence imported successfully."
    SENTENCE_LENGTH_EXCEEDED = "Sentence size exceeded."