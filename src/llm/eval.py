import json
from tqdm import tqdm
import nltk
import spacy
import string
import evaluate  # Bleu
import pandas as pd
import numpy as np
from tqdm import tqdm

bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load('meteor')

def predict_answer(predicted_answer, ref_answer):
    # try:                   
    bleu_score = bleu.compute(predictions=[predicted_answer],
                        references=[ref_answer])
    rouge_score = rouge.compute(predictions=[predicted_answer],
                        references=[ref_answer])
    bert_score = bertscore.compute(predictions=[predicted_answer],
                        references=[ref_answer], lang='vi')
    # meteor_score = meteor.compute(predictions=[predicted_answer],
    #                     references=[ref_answer])

    return rouge_score, bert_score, bleu_score["google_bleu"]
    
    # except:
    #     return None, None, None
  


if __name__ == "__main__":
    tqdm.pandas()
    # test data    
    examples = []  
    ref_answers = []    
    pred_answers = []
    df_predict = pd.DataFrame(columns=["question", "reference_answer", "predict_answer"])
        
    jsonObj = pd.read_json(path_or_buf="/root/masterthesis/MedicalGPT/data/viherbqa_json/test/test.jsonl", lines=True)      
    for obj in jsonObj["conversations"]:
        examples.append(obj[0]['value'])
        ref_answers.append(obj[1]['value'])
    
    df_predict["question"] = examples
    df_predict["reference_answer"] = ref_answers
    
    df_pred_ans = pd.read_json(path_or_buf="/root/masterthesis/MedicalGPT/predict_llama3.1.jsonl", lines=True)          
    df_pred_ans["reference_answer"] = df_pred_ans["Input"].apply(lambda x: df_predict[df_predict["question"] == x]["reference_answer"].tolist()[0])
    df_pred_ans["Output"] = df_pred_ans["Output"].apply(lambda x: x.split("USER:")[0])
    print(df_pred_ans["Output"][0])
    print(df_pred_ans["Output"][3000])
    print(df_pred_ans["Output"][6000])
    print(df_pred_ans.head())
    
    
    df_temp = df_pred_ans

    tqdm.pandas()
    df_temp[["rouge", "bert_score", "bleu"]] = df_temp.progress_apply(lambda col: predict_answer(col["Output"], col["reference_answer"]), axis=1, result_type="expand")

    df_temp["rouge1"] = df_temp["rouge"].apply(lambda x: x["rouge1"] if x else None)
    df_temp["rouge2"] = df_temp["rouge"].apply(lambda x: x["rouge2"] if x else None)
    df_temp["rougeL"] = df_temp["rouge"].apply(lambda x: x["rougeL"] if x else None)
    df_temp["rougeLsum"] = df_temp["rouge"].apply(lambda x: x["rougeLsum"] if x else None)

    df_temp["precision_bert_score"] = df_temp["bert_score"].apply(lambda x: x["precision"][0] if x else None)
    df_temp["recall_bert_score"] = df_temp["bert_score"].apply(lambda x: x["recall"][0] if x else None)
    df_temp["f1_bert_score"] = df_temp["bert_score"].apply(lambda x: x["f1"][0] if x else None)

    df_temp.to_csv("eval_llama3.1.csv", index=False)
