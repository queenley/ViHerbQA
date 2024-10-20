import wandb
run = wandb.init(project="viherbqa", name="[close]vit5_base")
from tqdm import tqdm
from .config import *

class Trainer: 
    def __init__(self,
                    dataloader):
        self.dataloader = dataloader        
        
        self.best_loss = -1
        self.best_model = None
        self.best_epoch = 0

    def __call__(self):
        # run.watch(MODEL)
        for epoch in range(EPOCHS):
            # Train
            MODEL.train()        
            train_loss = self.trainer(self.dataloader.train_loader, "train")
                                            
            # Evaluation
            MODEL.eval()
            val_loss = self.trainer(self.dataloader.val_loader, "val")                        
            if (self.best_loss == -1) or (val_loss < self.best_loss):
                self.best_loss = val_loss
                self.best_epoch = epoch                 
                self.save_model()                
                
            run.log({f"Train loss by epoch": train_loss})            
            run.log({f"Val loss by epoch": val_loss})            
            print(f"\n{epoch+1}/{EPOCHS} -> Train loss: {train_loss}\tValidation loss: {val_loss}")   


    # def print_answer(self, outputs):
    #     predicted_answer = TOKENIZER.decode(outputs.flatten(), skip_special_tokens=True)
    #     print(predicted_answer)                                         


    def trainer(self, data_loader, type="train"):    
        sum_loss = 0    
        batch_count = 0
        for batch in tqdm(data_loader, desc=f"{type} batches"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

            outputs = MODEL(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            decoder_attention_mask=decoder_attention_mask
                            )

            OPTIMIZER.zero_grad()
            outputs.loss.backward()
            OPTIMIZER.step()
            
            sum_loss += outputs.loss.item()
            batch_count += 1          
                        
            if batch_count % BATCH_LOG == 0:                            
                run.log({f"{type} loss": sum_loss / batch_count})  
                      
        
        loss = sum_loss / batch_count

        if type == "val":
            if loss < self.best_loss:                                
                self.save_model(name=f"step_{batch_count}")                

        return loss


    def make_model_contiguous(self):
        for param in MODEL.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()


    def save_model(self, name="best"):        
        self.make_model_contiguous()
        MODEL.save_pretrained(f"viherbqa/{name}_model")
        TOKENIZER.save_pretrained(f"viherbqa/{name}_tokenizer")   
