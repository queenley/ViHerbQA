import pandas as pd

viherbqa_official = pd.read_csv("../dataset/viherbqa_official_cleaned_v2.csv")
viherbqa_official.fillna("", inplace=True)
print(viherbqa_official.columns)

def check_large_context():
    _df_temp = viherbqa_official[viherbqa_official["length_question_conxtext"] > 1024]
    _df_temp.reset_index(inplace=True, drop=True)    
    print("Total records: ", len(viherbqa_official))
    print("Total large context records: ", len(_df_temp))
    print("Total normal context records: ", len(viherbqa_official) - len(_df_temp))

    _df_temp.to_csv("../dataset/viherbqa_official_cleaned_large.csv", index=False)
    
    # print("\n***Sample large context***")
    # print("- Question: ", _df_temp["question"][0])
    # print("- Context: ", _df_temp["answer"][0])
    # print("- Answer: ", _df_temp["context"][0])

    # _df_temp.sort_values(by="length_conxtext", inplace=True)

    # print("\n***Sample min large context***")
    # print("- Question: ", _df_temp["question"][0])
    # print("- Context: ", _df_temp["answer"][0])
    # print("- Answer: ", _df_temp["context"][0])

    # print("\n***Sample max large context***")
    # print("- Question: ", _df_temp["question"][len(_df_temp) - 1])
    # print("- Context: ", _df_temp["answer"][len(_df_temp) - 1])
    # print("- Answer: ", _df_temp["context"][len(_df_temp) - 1])


def check_token_length():
    from transformers import AutoTokenizer
    # Khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-large')

    answer_texts = viherbqa_official["answer"].tolist()
    question_texts = viherbqa_official["question"].tolist()
    context_texts = viherbqa_official["clean_newline"].tolist()
    
    answer_encoded = tokenizer(answer_texts, padding=False, truncation=False)
    answer_lengths = [len(ids) for ids in answer_encoded['input_ids']]

    question_encoded = tokenizer(question_texts, padding=False, truncation=False)
    question_lengths = [len(ids) for ids in question_encoded['input_ids']]

    context_encoded = tokenizer(context_texts, padding=False, truncation=False)
    context_lengths = [len(ids) for ids in context_encoded['input_ids']]

    encoded = tokenizer(question_texts, context_texts, padding=False, truncation=False, return_token_type_ids=True)
    lengths = [len(ids) for ids in encoded['input_ids']]

    print(f"Min token answer: ", min(answer_lengths))
    print(f"Max token answer: ", max(answer_lengths))

    print(f"Min token question: ", min(question_lengths))
    print(f"Max token question: ", max(question_lengths))

    print(f"Min token context: ", min(context_lengths))
    print(f"Max token context: ", max(context_lengths))

    print(f"Min token: ", min(lengths))
    print(f"Max token: ", max(lengths))

    viherbqa_official["length_conxtext"] = context_lengths
    viherbqa_official["length_question_conxtext"] = lengths
    viherbqa_official.to_csv("../dataset/viherbqa_official_cleaned_v2.csv", index=False)


def correct_vietnamese():
    from transformers import pipeline
    corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction", device=0)
    MAX_LENGTH = 512
    text = 'Giảm cholesterol\nNhững tác động tích cực của tinh dầu và chiết xuất từ quế đối với mức cholesterol và đường huyết trong cơ thể con người được đánh giá tích cực:\nNghiên cứu trên động vật, khi kết hợp tập luyện thường xuyên cùng bổ sung tinh dầu này làm giảm đáng kể nồng độ cholesterol toàn phần, LDL trong huyết thanh. Đồng thời làm tăng mức lipoprotein tốt (HDL) so với nhóm chứng.\nNgoài ra, trong một nghiên cứu trên chuột, hỗn hợp nhóm tinh dầu (bao gồm dầu quế) giúp điều chỉnh mức đường huyết lưu thông. Chúng cũng cải thiện hoạt động của tuyến tụy, bao gồm tiết ra insulin. Do đó, chiết xuất từ quế được nhận định giúp giảm các yếu tố nguy cơ liên quan đến bệnh tim mạch và đái tháo đường.\nCinnamaldehyd còn kích hoạt các phản ứng sinh nhiệt, trao đổi chất trong tế bào mỡ dưới da của chuột và người. Điều này góp phần đưa ra lời giải thích cơ học cho tác dụng chống béo phì của chúng và hỗ trợ thêm cho lợi ích trao đổi chất tiềm năng đối với con người.\nChống viêm\nNgày càng nhiều các nghiên cứu đa dạng khác nhau mô tả hoạt động chống viêm của cinnamaldehyde. Đây là một hoạt chất dồi dào có trong tinh dầu quế. Chúng được mô tả thông qua con đường tín hiệu khác nhau để điều chỉnh các phản ứng chống viêm. Hơn thế, đây còn là thành phần được biết đến có tác dụng kháng nấm, chống ung thư… nhưng cần phải cần nghiên cứu chi tiết hơn.\nỨc chế hoạt động vi sinh vật\nTinh dầu quế chứa cinnamaldehyd, hoạt chất có tác dụng ức chế hoạt động vi sinh vật in vitro. Có thể kể đến như:\nKháng khuẩn: Salmonella typhi, tụ cầu vàng, Bacillus mycoides, Bacillus subtilis…\nKháng nấm: Candida albicans, Trychphyton mentagrophytes, Microsporum audoim,… cùng nhiều loại nấm mốc và nấm men khác.\nVới tác dụng kể trên, tinh dầu này là sự lựa chọn hiệu quả để chống lại các bệnh lý rối loạn đường hô hấp như đau họng, sổ mũi, nghẹt mũi, đờm tắc nghẽn… thông qua các phương pháp xông hơi, khuếch tán…\nHơn thế, có thể tận dụng lợi ích này để khử mùi và thanh lọc cho không gian sống.\nGiảm căng thẳng\nTừ xa xưa, liệu pháp hương thơm đã được phát hiện để giúp kiểm soát căng thẳng cũng như các triệu chứng trầm cảm. Đặc biệt, khi kết hợp với tinh dầu từ quế được cho là nguồn cung cấp các hoạt chất chống lại stress oxy hóa. Chúng mang đến trải nghiệm thư giãn, cân bằng cảm xúc, cải thiện sự tỉnh táo và nhận thức. Chúng có thể ức chế sản xuất một s\nố dấu ấn sinh học protein liên quan đến viêm da và tái tạo mô. Thế nhưng, lợi ích này cần được nghiên cứu thêm để xác định hiệu quả l\nâm sàng và an toàn.'
    predictions = corrector([text], max_length=MAX_LENGTH)
    print(predictions[0]['generated_text'])

if __name__ == "__main__":
    check_large_context()
