from torch.optim import Adam
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM


# TOKENIZER = T5TokenizerFast.from_pretrained("/root/masterthesis/model/base/best_model")
# MODEL = T5ForConditionalGeneration.from_pretrained("/root/masterthesis/model/base/best_tokenizer", return_dict=True)
# TOKENIZER = AutoTokenizer.from_pretrained("/root/masterthesis/open_book_model/base/best_tokenizer")  
# MODEL = AutoModelForSeq2SeqLM.from_pretrained("/root/masterthesis/open_book_model/base/best_model")

TOKENIZER = AutoTokenizer.from_pretrained("VietAI/vit5-large")  
MODEL = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-large")

OPTIMIZER = Adam(MODEL.parameters(), lr=1e-5)
Q_LEN = 1024   # Question Length
T_LEN = 1024    # Target Length
BATCH_SIZE = 8
DEVICE = "cuda:0"
MODEL.to(DEVICE)
MODEL.gradient_checkpointing_enable()
EPOCHS = 5
BATCH_LOG = 100
