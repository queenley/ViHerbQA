import os
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

bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

TOKENIZER_PATH = "model/open/best_tokenizer"
MODEL_PATH = "model/open/best_model"
Q_LEN = 1024
DEVICE = "cuda:0"

TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
MODEL.to(DEVICE)

def predict_answer(question, context=None, ref_answer=None):
    try:      
        if context is None:
            input_text = f'CÂU HỎI: {question} </s>'            
        else:    
            input_text = f'CÂU HỎI: {question} </s> NGỮ CẢNH: {context} </s>'
      
        inputs = TOKENIZER(input_text, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

        outputs = MODEL.generate(input_ids=input_ids,
                               attention_mask=attention_mask,
                               max_new_tokens=Q_LEN,
                               )

        predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)

        # Load the metrics

        bleu_score = bleu.compute(predictions=[predicted_answer],
                          references=[ref_answer])
        rouge_score = rouge.compute(predictions=[predicted_answer],
                          references=[ref_answer])
        bert_score = bertscore.compute(predictions=[predicted_answer],
                          references=[ref_answer], lang='vi')

      
        return (predicted_answer,
              bleu_score["google_bleu"],
              rouge_score["rouge1"],
              rouge_score["rouge2"],
              rouge_score["rougeL"],
              rouge_score["rougeLsum"],
              bert_score["precision"][0],
              bert_score["recall"][0],
              bert_score["f1"][0],
              )

    except:
        return [None] * 9


if __name__ == "__main__":
    tqdm.pandas()
    path = "dataset/train.csv"
    df_test = pd.read_csv(path)
    df_temp = df_test
    response_columns = ["predicted_answer", "bleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "bert_precision", "bert_recall", "bert_f1"]
    df_temp[response_columns] = df_temp.progress_apply(lambda col: predict_answer(col["question"], col["context"], col["answer"]), axis=1, result_type="expand")
    df_temp.to_csv(f"result/open/{os.path.basename(path)}")