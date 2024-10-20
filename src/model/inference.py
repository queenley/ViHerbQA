import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
import nltk
import spacy
import string
import evaluate  # Bleu
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
import transformers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments

import warnings
warnings.filterwarnings("ignore")

TOKENIZER = AutoTokenizer.from_pretrained("/root/masterthesis/model/base/best_tokenizer")  
MODEL = AutoModelForSeq2SeqLM.from_pretrained("/root/masterthesis/model/base/best_model")
OPTIMIZER = Adam(MODEL.parameters(), lr=0.00001)
Q_LEN = 512   # Question Length
T_LEN = 1024    # Target Length
BATCH_SIZE = 4
DEVICE = "cuda:0"
MODEL.to(DEVICE)


def predict_answer(context, question, ref_answer=None):
    inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)
    
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)
  
    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
    
    if ref_answer:
        # Load the Bleu metric
        bleu = evaluate.load("google_bleu")
        score = bleu.compute(predictions=[predicted_answer], 
                            references=[ref_answer])
    
        print("Context: \n", context)
        print("\n")
        print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer, 
            "Predicted Answer: ": predicted_answer, 
            "BLEU Score: ": score
        }
    else:
        return predicted_answer


def predict_close_answer(question, ref_answer=None):
    inputs = TOKENIZER(question, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)
    
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

    outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)
  
    predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
    
    if ref_answer:
        # Load the Bleu metric
        bleu = evaluate.load("google_bleu")
        score = bleu.compute(predictions=[predicted_answer], 
                            references=[ref_answer])
            
        print("\n")
        print("Question: \n", question)
        return {
            "Reference Answer: ": ref_answer, 
            "Predicted Answer: ": predicted_answer, 
            "BLEU Score: ": score
        }
    else:
        return predicted_answer        


if __name__ == "__main__":
    context = """Tên khoa học: Clinacanthus nutans Lindau. Ngoài ra Clinacanthus burmanni Nees, Clinacanthus burmanni var. robinsonii Benoist cũng là từ đồng nghĩa của C. nutans (Burm. f.) Lindau. Họ Acanthaceae (Ô rô) là một trong những họ hàng đầu của thực vật có hoa hai lá mầm. Họ Ô rô bao gồm 250 chi và khoảng 2500 loài. Clinacanthus nutans (C.nutans ) Lindau là một trong những loài quan trọng của họ này. Nó đã được sử dụng làm thuốc từ lâu đời ở các nước Đông Nam Á. Hiện loài cây này đang thu hút sự quan tâm của nhiều nhà nghiên cứu vì tác dụng chữa bệnh của nó. Ở Thái Lan, Xương khỉ để điều trị viêm da và tổn thương do virus gây ra."""
    predicted_answer = predict_answer(context, "Cây Xương khỉ thuộc họ thực vật nào?", "Cây Xương khỉ thuộc họ Acanthaceae (Ô rô).")
    print(predicted_answer)
    