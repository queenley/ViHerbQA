import ast
import json
import pandas as pd
from tqdm import tqdm
import time
from glob import glob
import argparse
import os
import json
import json_repair
import logging
from dotenv import load_dotenv

from call_llm import GenLib

load_dotenv('.env')
env = os.environ


def make_parser():    
    parser = argparse.ArgumentParser("Generate questions by LLM")
    parser.add_argument("-m", "--model", 
                        type=str, 
                        default='gpt-4o-mini',                         
                        help="LLM is called to generate")
    
    parser.add_argument("-p", "--path", 
                        type=str, 
                        default='/Users/queen/Desktop/masterthesis/dataset/knowledge/csv/y_hoc_co_truyen_new.csv', 
                        help="path of documents")
    
    parser.add_argument("-q", "--question",
                        type=str,
                        default="/Users/queen/Desktop/masterthesis/dataset/question/generated_data/",
                        help="question directory")
    
    parser.add_argument("-d", "--folder",
                        type=str,
                        default="chatgpt_cot",
                        help="data folder")
    
    parser.add_argument("-o", "--output",
                        type=str,
                        default="/Users/queen/Desktop/masterthesis/dataset/answer/",
                        help="output directory")
    
    parser.add_argument("-key", "--openai_key",
                        type=str,                        
                        help="openai key")
    
    parser.add_argument("-proj", "--vertex_project",
                        type=str,                        
                        help="vertex project")
    
    parser.add_argument("-locate", "--vertex_location",
                        type=str,                        
                        help="vertex location")
    
    parser.add_argument("-t", "--temperature",
                        default=1.0,
                        type=float,                        
                        help="temperature")
    
    

    return parser 


def design_prompt(herb, num_ques=None, list_of_questions=None, menu=None, context=None, is_sys=False, model=""):
    sys_prompt_file = "/Users/queen/Desktop/masterthesis/prompt/system_answer.txt"
    user_prompt_file = "/Users/queen/Desktop/masterthesis/prompt/user_answer.txt"
    
    if "gemini" in model:
        sys_prompt_file = "/Users/queen/Desktop/masterthesis/prompt/gemini/system_answer.txt"
        user_prompt_file = "/Users/queen/Desktop/masterthesis/prompt/gemini/user_answer.txt"
    
    
    if is_sys:
        with open(sys_prompt_file) as f:
            prompt_txt = f.read()    
        prompt_txt = prompt_txt.replace("{{**duoc_lieu**}}", herb)
    
    else:
        with open(user_prompt_file) as f:
            prompt_txt = f.read()    

        prompt_txt = prompt_txt.replace("{{**duoc_lieu**}}", herb)
        prompt_txt = prompt_txt.replace("{{**num_questions**}}", str(num_ques))        
        prompt_txt = prompt_txt.replace("{{**danh_muc**}}", menu)
        prompt_txt = prompt_txt.replace("{{**context**}}", context)
        prompt_txt = prompt_txt.replace("{{**list_of_questions**}}", list_of_questions)
    
    return prompt_txt 


def main(args, df_result, llm_model):     
    out_dir = f"{args.output}/{args.folder}"    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with tqdm(total=len(df_result)) as pbar:   
        for _, row in df_result.iterrows():   
            herb = row['Tên dược liệu']   
            out_name = f"{out_dir}/{row['Tên dược liệu']}.json"

            if os.path.exists(out_name):
                pbar.update(1)
                continue
            
            # load questions
            question_path = f"{args.question}/{args.folder}/{herb}.txt"
            if not os.path.exists(question_path):
                pbar.update(1)
                continue
            with open(question_path) as f:
                list_questions = [q.strip() for q in f.readlines()]

            
            context = row["full_context"]
            
            content = row["content"].split("\n")
            for i, c in enumerate(content):
                if "bài thuốc" in c.lower() or "đơn thuốc" in c.lower():
                    context = "\n".join(content[i:])
                    break 
            
            if context is None:
                pbar.update(1)
                continue 

            # Get prompt
            try:
                system_prompt = design_prompt(row["Cây thuốc"], is_sys=True)            
                user_prompt = design_prompt(row["Cây thuốc"],
                                        len(list_questions),
                                    "\n".join(list_questions),                                   
                                    "\n".join(ast.literal_eval(row["menu"])), 
                                    context)
            except:
                pbar.update(1)
                continue
            
            try:
                # Call LLM
                if "gpt" in args.model:
                    llm_model.get_response(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        top_p=0.95,
                        max_output_tokens=16089,
                        temperature=args.temperature,
                        model=args.model)
                
                elif "gemini" in args.model:
                    llm_model.get_response(                
                    user_prompt=user_prompt,
                    top_p=0.95,
                    # max_output_tokens=16089,
                    temperature=args.temperature,
                    model=args.model)
            
                reply = llm_model.result["text"]
            except Exception as e:
                # logging.info("\n")
                # logging.info(herb)
                # logging.error(e)
                print(herb)
                # print(e)
                pbar.update(1)
                continue

            json_reply = json_repair.repair_json(reply, return_objects=True)                                     
                        
            with open(out_name, "w") as f:
                json.dump(json_reply, f, ensure_ascii=False, indent=6)                        
            
            pbar.update(1)            
            


if __name__=="__main__":
    args = make_parser().parse_args()    
    df_result = pd.read_csv(args.path)

    if "gpt" in args.model:        
        genlib = GenLib("openai")
        genlib.get_model(api_key=args.openai_key)    
        

    elif "gemini" in args.model:
        genlib = GenLib("vertex")
        genlib.get_model(project=args.vertex_project,
                         location=args.vertex_location)    
    
    
    llm_model = genlib.llm_model
    main(args, df_result, llm_model)     
