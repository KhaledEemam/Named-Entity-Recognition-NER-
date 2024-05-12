from torch.utils.data import DataLoader
from .build_dataset import ReviewDataset

class Loader :
    def __init__(self,sentences_tokens , tags , tokenizer ,max_length,batch_size=16,pin_memory=True,num_workers=2,shuffle = True) :
        self.sentences_tokens = sentences_tokens
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataset = ReviewDataset(self.sentences_tokens , self.tags , self.tokenizer ,self.max_length)

    def get_data_loader(self) :
        data_loader = DataLoader(self.dataset ,batch_size=self.batch_size,pin_memory=self.pin_memory,
                                 num_workers=self.num_workers,shuffle = self.shuffle)
        return data_loader
