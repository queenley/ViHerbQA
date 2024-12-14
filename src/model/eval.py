from .utils import format_result

import torch
import evaluate
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

bleu = evaluate.load("google_bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


class Validate:
    def __init__(self,
                 model,
                 tokenizer,
                 q_len,
                 t_len,
                 dataset,        
                 device,
                 output_folder,
                 is_open,
                ):

        self.model = model
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len        
        self.device = device
        self.output_folder = output_folder
        self.is_open = is_open

        self.df_test = pd.read_csv(f"{dataset}/test.csv")   
        # self.df_test = self.df_test[:2]
        

    def __call__(self):
        response_columns = ["predicted_answer", "bleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "bert_precision", "bert_recall", "bert_f1"]
        tqdm.pandas()

        if self.is_open:
            self.df_test[response_columns] = self.df_test.progress_apply(lambda col: self.eval_with_context(col["question"], col["context"], col["final_answer"]), axis=1, result_type="expand")
            self.df_test.to_csv(f"{self.output_folder}/eval_with_context.csv", index=False)
        
        else:
            self.df_test[response_columns] = self.df_test.progress_apply(lambda col: self.eval_without_context(col["question"], col["final_answer"]), axis=1, result_type="expand")
            self.df_test.to_csv(f"{self.output_folder}/eval_without_context.csv", index=False)
                

    def generate_answer(self, input_text):
        inputs = self.tokenizer(input_text,                                             
                                        max_length=self.q_len, 
                                        padding="max_length",
                                        truncation=True,
                                        pad_to_max_length=True, 
                                        add_special_tokens=True,                                             
                                    )
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(self.device).unsqueeze(0)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(self.device).unsqueeze(0)

        outputs = self.model.generate(input_ids=input_ids,
                               attention_mask=attention_mask,
                               max_new_tokens=self.t_len,
                               )

        predicted_answer = self.tokenizer.decode(outputs.flatten(), skip_special_tokens=True)
        
        return predicted_answer

    
    def compute_metrics(self, predicted_answer, ref_answer):
        bleu_score = bleu.compute(predictions=[predicted_answer],
                          references=[ref_answer])
        rouge_score = rouge.compute(predictions=[predicted_answer],
                          references=[ref_answer])
        bert_score = bertscore.compute(predictions=[predicted_answer],
                          references=[ref_answer], lang='vi')
      
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


    def eval_with_context(self, question, context, gold_answer):
        input_text = f'CÂU HỎI: {question} </s> NGỮ CẢNH: {context} </s>'
        predicted_answer = self.generate_answer(input_text)
        metrics = self.compute_metrics(predicted_answer, gold_answer)
        
        return format_result(predicted_answer, metrics)                


    def eval_without_context(self, question, gold_answer):
        input_text = f'CÂU HỎI: {question} </s>'            
        predicted_answer = self.generate_answer(input_text)
        metrics = self.compute_metrics(predicted_answer, gold_answer)
        
        return format_result(predicted_answer, metrics)                


# if __name__ == "__main__":
#     tqdm.pandas()
#     path = "dataset/train.csv"
#     df_test = pd.read_csv(path)
#     df_temp = df_test
#     response_columns = ["predicted_answer", "bleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "bert_precision", "bert_recall", "bert_f1"]
#     df_temp[response_columns] = df_temp.progress_apply(lambda col: predict_answer(col["question"], col["context"], col["answer"]), axis=1, result_type="expand")
#     df_temp.to_csv(f"result/open/{os.path.basename(path)}")