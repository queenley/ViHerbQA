from model import Trainer, Dataloader

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":    
    data_path = "/root/masterthesis/dataset/viherbqa"
    dataloader = Dataloader(data_path)
    dataloader()

    trainer = Trainer(dataloader)
    trainer()
    trainer.save_model("last")
    print(f"\n Best model at epoch: {trainer.best_epoch} with loss {trainer.best_loss}")
    