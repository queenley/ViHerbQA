import time
import torch
from transformers import AutoModel, AutoTokenizer, pipeline, T5Tokenizer, T5ForConditionalGeneration
from torch.quantization import quantize_dynamic

tokenizer = AutoTokenizer.from_pretrained("model/close/best_tokenizer")
model = T5ForConditionalGeneration.from_pretrained("model/close/best_model")
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
device = "cuda:0"
quantized_model.to(device)

def predict(input_text):
    inputs = tokenizer(input_text,                                             
                        max_length=1024, 
                        padding="max_length",
                        truncation=True,
                        pad_to_max_length=True, 
                        add_special_tokens=True,
                    )
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(device).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(device).unsqueeze(0)

    outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=1024,
                            
                            )

    predicted_answer = tokenizer.decode(outputs.flatten(), skip_special_tokens=True)
    return predicted_answer


i = 0
while True:
    if i == 50:
        break

    st = time.time()
    results = predict("Actiso có tên khoa học là gì?")
    print(time.time() - st)
    i += 1

print(results)

