from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import ast


class ReviewDataset(Dataset) :
    def __init__(self,sentences_tokens , tags , tokenizer ,max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        sentences_tokens = sentences_tokens.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        tags = tags.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Convert tokens into tensors type
        tensor_tokens = [torch.tensor(tokens) for tokens in sentences_tokens]

        # Pad tokens to the input max length
        padded_tensor_tokens =  pad_sequences(tensor_tokens , maxlen=self.max_length , padding="post" , value=self.tokenizer.get_vocab()['[PAD]'])

        # Convert tags to tensors type
        tensor_tags = [torch.tensor(tag) for tag in tags]

        # Build mask: a list of ones representing the original sentence length and zeros for any padded tokens
        self.mask = [torch.ones(len(seq)) for seq in tensor_tags]
        self.mask = torch.tensor(pad_sequences(self.mask , maxlen=self.max_length , padding="post",value=0),dtype=torch.uint8)

        # Pad tags to the input max length
        padded_tensor_tags = pad_sequences(tensor_tags , maxlen=self.max_length , padding="post",value=-100)
        self.sentences_tokens = padded_tensor_tokens
        self.tags =  padded_tensor_tags

    def __len__(self) :
        return len(self.sentences_tokens)

    def __getitem__(self, index) :
        sentence_token = self.sentences_tokens[index]
        target = self.tags[index]
        masks = self.mask[index]

        return {"tokens" : sentence_token , "tags" : target , "masks" : masks }
