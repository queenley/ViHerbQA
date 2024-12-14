import torch

def generate_answer(model,
                    tokenizer, 
                    input_text, 
                    q_len, 
                    t_len,
                    device):

    inputs = tokenizer(input_text,                                             
                        max_length=q_len, 
                        padding="max_length",
                        truncation=True,
                        pad_to_max_length=True, 
                        add_special_tokens=True,
                    )
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).to(device).unsqueeze(0)
    attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).to(device).unsqueeze(0)

    outputs = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=t_len,
                            )

    predicted_answer = tokenizer.decode(outputs.flatten(), skip_special_tokens=True)
    
    return predicted_answer