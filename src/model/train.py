import os
import torch
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()
WANDB_KEY = os.getenv('WANDB_KEY')


class Trainer: 
    def __init__(self,                
                dataloader,
                tokenizer,
                model,
                optimizer,
                device: str,
                epoch: int,
                batch_log: int,
                project_name: str,
                folder_name: str,
                save_path: str,                
                ):                    

        wandb.login(key=WANDB_KEY)
        self.run = wandb.init(project=project_name, name=folder_name)        

        self.save_path = save_path
        self.epoch = epoch  
        self.batch_log = batch_log
        self.dataloader = dataloader    
        self.device = device
        self.tokenizer = tokenizer
        self.model = model     
        self.optimizer = optimizer  
        
        self.best_loss = -1
        self.best_model = None
        self.best_epoch = 0

                
    def __call__(self):
        # run.watch(MODEL)
        for epoch in range(self.epoch):
            # Train            
            self.model.train()        
            train_loss = self.trainer(self.dataloader.train_loader, "train")

            # self.save_model(name=f"epoch{epoch}")                
                                            
            # Evaluation
            self.model.eval()            
            val_loss = self.trainer(self.dataloader.val_loader, type="val")
            if (self.best_loss == -1) or (val_loss < self.best_loss):
                self.best_loss = val_loss
                self.best_epoch = epoch                 
                self.save_model()                
                
            # Logging
            self.run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })         
            print(f"\n{epoch+1}/{self.epoch} -> Train loss: {train_loss}\tValidation loss: {val_loss}")                               

        wandb.finish()        

    # def print_answer(self, outputs):
    #     predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
    #     print(predicted_answer)                                         


    def trainer(self, data_loader, type="train", use_cache=False):    
        sum_loss = 0    
        batch_count = 0
        with torch.no_grad() if type == "val" else torch.enable_grad():
            for batch in tqdm(data_loader, desc=f"{type} batches"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

                outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                decoder_attention_mask=decoder_attention_mask,
                                use_cache=use_cache,
                                )

                if type == "train":
                    self.optimizer.zero_grad()
                    outputs.loss.backward()
                    self.optimizer.step()
                
                sum_loss += outputs.loss.item()
                batch_count += 1                 
                            
                if batch_count % self.batch_log == 0:                            
                    self.run.log({f"{type} loss": sum_loss / batch_count})  
                      
        
        loss = sum_loss / batch_count

        if type == "val":
            if loss < self.best_loss:                                
                self.save_model(name=f"step_{batch_count}")                

        return loss


    def make_model_contiguous(self):
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()


    def save_model(self, name="best"):        
        self.make_model_contiguous()
        self.model.save_pretrained(f"{self.save_path}/{name}_model")
        self.tokenizer.save_pretrained(f"{self.save_path}/{name}_tokenizer")   
