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

from config import *

bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load('meteor')

def predict_answer(context, question, ref_answer=None):
    try:
        if type(context) != str:
            context = ""
        inputs = TOKENIZER(question, context, max_length=Q_LEN, padding="max_length", truncation=True, add_special_tokens=True)

        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(DEVICE).unsqueeze(0)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(DEVICE).unsqueeze(0)

        outputs = MODEL.generate(input_ids=input_ids, attention_mask=attention_mask)

        predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
        
        # Load the Bleu metric

        bleu_score = bleu.compute(predictions=[predicted_answer],
                            references=[ref_answer])
        rouge_score = rouge.compute(predictions=[predicted_answer],
                            references=[ref_answer])
        bert_score = bertscore.compute(predictions=[predicted_answer],
                            references=[ref_answer], lang='vi')
        meteor_score = meteor.compute(predictions=[predicted_answer],
                            references=[ref_answer])



        # print("Context: \n", context)
        # print("\n")
        # print("Question: \n", question)
        return predicted_answer, bleu_score["google_bleu"], rouge_score, bert_score, meteor_score['meteor']
    
    except:
        return None, None, None, None, None
  


if __name__ == "__main__":
    tqdm.pandas()
    df_test = pd.read_csv("/root/masterthesis/dataset/viherbqa/test.csv")
    df_temp = df_test
    df_temp[["predicted_answer", "bleu", "rouge", "bert_score", "meteor"]] = df_temp.progress_apply(lambda col: predict_answer(col["context"], col["question"], col["answer"]), axis=1, result_type="expand")
    df_temp.to_csv("eval_large_close.csv", index=False)