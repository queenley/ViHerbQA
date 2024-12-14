    import time
    from transformers import AutoTokenizer, pipeline
    from optimum.onnxruntime import ORTModelForSeq2SeqLM


def inference_onnx():
    tokenizer = AutoTokenizer.from_pretrained("model/close/best_tokenizer")
    model = ORTModelForSeq2SeqLM.from_pretrained("model/close/viherbqa_onnx_v2", provider="CPUExecutionProvider")
    translator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, device=0)

    i = 0
    while True:
        if i == 50:
            break

        st = time.time()
        results = translator("Actiso có tên khoa học là gì?")
        print(time.time() - st)
        i += 1

    print(results)


