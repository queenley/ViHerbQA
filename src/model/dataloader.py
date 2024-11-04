import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from .config import *


class Dataloader:
    def __init__(self,
                data_path: str):
        self.data_path = data_path        

    def __call__(self):
        self.read_data()
        self.sampling_data()
        self.load_data()

    def read_data(self):
        self.data = pd.read_csv(f"{self.data_path}/viherbqa_official_v2.csv")
        self.train_data = pd.read_csv(f"{self.data_path}/train_v2.csv")
        self.val_data = pd.read_csv(f"{self.data_path}/val_v2.csv")
        self.test_data = pd.read_csv(f"{self.data_path}/test_v2.csv")

    def sampling_data(self):
        self.train_sampler = RandomSampler(self.train_data.index)
        self.val_sampler = RandomSampler(self.val_data.index)
        self.test_sampler = RandomSampler(self.test_data.index)

    def load_data(self):        
        qa_dataset = QADataset(TOKENIZER, self.data, Q_LEN, T_LEN)
        self.train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=self.train_sampler)
        self.val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=self.val_sampler)
        self.test_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=self.test_sampler)


class QADataset(Dataset):
    def __init__(self, 
                tokenizer, 
                dataframe, 
                q_len, 
                t_len):

        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.context = self.data["clean_newline"]
        self.answer = self.data['answer']        
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):        
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]

        input_text = f'CÂU HỎI: {question} </s> NGỮ CẢNH: {context} </s>'
        
        question_tokenized = self.tokenizer(input_text,                                             
                                            max_length=self.q_len, 
                                            padding="max_length",
                                            truncation=True,
                                            pad_to_max_length=True, 
                                            add_special_tokens=True,                                             
                                        )
        
        answer_tokenized = self.tokenizer(answer, 
                                          max_length=self.t_len, 
                                          padding="max_length", 
                                          truncation=True, 
                                          pad_to_max_length=True, 
                                          add_special_tokens=True,
                                        )
        
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100
        
        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }


