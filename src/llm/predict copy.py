# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import json
import os
from threading import Thread

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
    BitsAndBytesConfig,
)

from template import get_conv_template
import pandas as pd    

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


@torch.inference_mode()
def batch_generate_answer(
        sentences,
        model,
        tokenizer,
        prompt_template,
        system_prompt,
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
):
    """Generate answer from prompt with GPT, batch mode"""
    generated_texts = []
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
    )
    # sentences = ["Thời gian đắp thuốc từ lá bàng tươi lên vùng da bị bệnh là bao lâu?"]
    print(sentences)
    messages = [[s, ''] for s in sentences]
    prompts = [prompt_template.get_prompt(messages=messages, system_prompt=system_prompt)]
    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=False)
    input_ids = inputs_tokens['input_ids'].to(device)
    outputs = model.generate(input_ids=input_ids, **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=True)
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

    return generated_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='auto', type=str)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument('--system_prompt', default="", type=str)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (default multi-turn)")
    parser.add_argument('--single_tune', action='store_true', help='Whether to use single-tune model')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit')
    args = parser.parse_args()
    print(args)
    load_type = 'auto'
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_type,
        "low_cpu_mem_usage": True,
        "device_map": 'auto',
    }
    if args.load_in_8bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_type,
        )
    base_model = model_class.from_pretrained(args.base_model, **config_kwargs)
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
    model.eval()
    print(tokenizer)
    
    # test data    
    df_predict = pd.DataFrame(columns=["question", "reference_answer", "predict_answer"])
    examples = []  
    ref_answers = []
    jsonObj = pd.read_json(path_or_buf=args.data_file, lines=True)      
    for obj in jsonObj["conversations"]:
        examples.append(obj[0]['value'])
        ref_answers.append(obj[1]['value'])

    df_predict['question'] = examples
    df_predict['reference_answer'] = ref_answers        

    # Chat
    prompt_template = get_conv_template(args.template_name)
    system_prompt = args.system_prompt
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str
    
    print("Start inference.")
    counts = 0
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    eval_batch_size = args.eval_batch_size
    pred_answers = []
    for batch in tqdm(
            [
                examples[i: i + eval_batch_size]
                for i in range(0, len(examples), eval_batch_size)
            ],
            desc="Generating outputs",
    ):
        responses = batch_generate_answer(
            batch,
            model,
            tokenizer,
            prompt_template,
            system_prompt,
            model.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            stop_str=stop_str,
        )
        results = []        
        for example, response in zip(batch, responses):
            # print(f"===")
            # print(f"Input: {example}")
            # print(f"Output: {response}\n")
            clean_response = response.split("USER:")[0]
            pred_answers.append(response)
            results.append({"Input": example, "Output": clean_response})
            counts += 1
        
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for entry in results:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

    df_predict["predict_answer"] = pred_answers
    df_predict.to_csv("predict.csv", index=False)
    print(f'save to {args.output_file}, size: {counts}')


if __name__ == '__main__':
    main()
