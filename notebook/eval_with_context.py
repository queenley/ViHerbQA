import os
import evaluate
import pandas as pd
from glob import glob
from tqdm import tqdm
from itertools import combinations, islice

import warnings
warnings.filterwarnings("ignore")

bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


def compute_metric(predict_answer, context):
    try:              
        # Load the metrics

        bleu_score = bleu.compute(predictions=[predict_answer],
                          references=[context])
        rouge_score = rouge.compute(predictions=[predict_answer],
                          references=[context])
        bert_score = bertscore.compute(predictions=[predict_answer],
                          references=[context], lang='vi')

      
        return (
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
        return [None] * 8


if __name__ == "__main__":
    tqdm.pandas()
    path = "result/open/val.csv"
    df_test = pd.read_csv(path)
    df_temp = df_test
    response_columns = ["bleu_with_context", "rouge1_with_context", "rouge2_with_context", "rougeL_with_context", "rougeLsum_with_context", "bert_precision_with_context", "bert_recall_with_context", "bert_f1_with_context"]
    df_temp[response_columns] = df_temp.progress_apply(lambda col: compute_metric(col["predicted_answer"], col["context"]), axis=1, result_type="expand")
    df_temp.to_csv(f"result/open/context_{os.path.basename(path)}")        