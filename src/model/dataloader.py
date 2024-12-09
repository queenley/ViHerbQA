import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler

# from .config import *


class Dataloader:
    def __init__(self,
                data_path: str,
                tokenizer,
                batch_size: int,
                q_len: int, 
                t_len: int,
                is_open: bool,
                ):
        self.data_path = data_path  
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.q_len = q_len
        self.t_len = t_len
        self.is_open = is_open


    def __call__(self):
        self.read_data()
        # self.sampling_data()
        self.load_data()

    def read_data(self):
        # self.data = pd.read_csv(f"{self.data_path}/viherbqa_official_v2.csv")
        self.train_data = pd.read_csv(f"{self.data_path}/train.csv")
        self.val_data = pd.read_csv(f"{self.data_path}/val.csv")
        self.test_data = pd.read_csv(f"{self.data_path}/test.csv")    

    def load_data(self):        
        # qa_dataset = QADataset(TOKENIZER, self.data, Q_LEN, T_LEN)
        train_dataset = QADataset(self.train_data, self.tokenizer, self.q_len, self.t_len, self.is_open)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        val_dataset = QADataset(self.val_data, self.tokenizer, self.q_len, self.t_len, self.is_open)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        test_dataset = QADataset(self.test_data, self.tokenizer, self.q_len, self.t_len, self.is_open)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)


class QADataset(Dataset):
    def __init__(self,                 
                dataset, 
                tokenizer,
                q_len,
                t_len,
                is_open,
                ):                
        self.data = dataset
        self.questions = self.data["question"]        
        self.answer = self.data['final_answer']     
        self.context = self.data["context"]   
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.is_open = is_open
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):        
        question = self.questions[idx]
        context = self.context[idx]
        answer = self.answer[idx]

        input_text = f'CÂU HỎI: {question} </s> '
        if self.is_open:
            print("---> Training with Open Book.....")
            input_text += f'NGỮ CẢNH: {context} </s>'
        
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


