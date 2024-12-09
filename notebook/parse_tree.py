from pyvi import ViTokenizer, ViPosTagger

text = u"Actiso trồng ở đồng bằng."
tokenizer = ViTokenizer.tokenize(text)
postagger = ViPosTagger.postagging(tokenizer)
print(postagger)