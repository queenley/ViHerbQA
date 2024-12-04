from model import Trainer, Dataloader

import os
import wandb
import torch
import argparse
import warnings
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()


def make_parser():
    parser = argparse.ArgumentParser("ViHerbQA project")
    parser.add_argument('--dataset', type=str, default="dataset", help="dataset path") 
    parser.add_argument('--model', type=str, default="VietAI/vit5-large", help="model name or model path") 
    parser.add_argument('--tokernizer', type=str, default="VietAI/vit5-large", help="tokenizer name or tokenizer path")
    parser.add_argument('--max_length', type=int, default=1024, help="max length of output")
    parser.add_argument('--question_length', type=int, default=1024, help="max length of input")
    parser.add_argument('--batch', type=int, default=8, help="batch size")
    parser.add_argument('--epoch', type=int, default=5, help="number of epoches")
    parser.add_argument('--batch_log', type=int, default=100, help="number of batch to write log")    
    parser.add_argument('--project', type=str, default="viherbqa", help="project of wandb")
    parser.add_argument('--name', type=str, default="[open]vit5_large", help="name of experiment")
    parser.add_argument('--save_path', type=str, default="model/large", help="save path of the model")
    parser.add_argument('--is_open', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0', help="Device")
    parser.add_argument('--lr', default=1e-5, help="Learning rate")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":    
    args = make_parser()    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tokenizer = AutoTokenizer.from_pretrained(args.tokernizer)         
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(args.device)
    model.gradient_checkpointing_enable()
    optimizer = Adam(model.parameters(), lr=args.lr)

    dataloader = Dataloader(data_path=args.dataset, 
                            tokenizer=tokenizer, 
                            batch_size=args.batch, 
                            q_len=args.question_length, 
                            t_len=args.max_length, 
                            is_open=args.is_open,
                            )    
    dataloader()                            

    trainer = Trainer(dataloader,
                      tokenizer,
                      model,
                      optimizer,
                      device=args.device,
                      epoch=args.epoch,
                      batch_log=args.batch_log,
                      project_name=args.project,
                      folder_name=args.name,
                      save_path=args.save_path,
                    )
    trainer()
    trainer.save_model("last")
    
    print(f"\n Best model at epoch: {trainer.best_epoch} with loss {trainer.best_loss}")
    