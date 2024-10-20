import json
import pandas as pd
from tqdm import tqdm
from glob import glob
import argparse
import os


def make_parser():    
    parser = argparse.ArgumentParser("Generate questions by LLM")
    parser.add_argument("-m", "--model", 
                        type=str, 
                        default='gpt3.5', 
                        choices=['bard', 'gpt3.5', 'gpt4.0'],
                        help="LLM is called to generate")
    
    parser.add_argument("-p", "--path", 
                        type=str, 
                        default='y_hoc_co_truyen_youmed_v1.csv', 
                        help="path of documents")
    
    parser.add_argument("-r", "--role", 
                        type=str, 
                        default='patient', 
                        choices = ['student', 'patient'], 
                        help="path of documents")
    
    parser.add_argument("-o", "--output",
                        type=str,
                        default="Questions",
                        help="output directory")
    
    return parser 


def make_prompt(role, docs):
    
    prompt = f"""As a {role}, please generate 5 Vietnamese questions naturally and generally about oriental medicine remedies, 
                specifically medicinal herbs used in Vietnamese Traditional Medicine. 
                The generated questions should not contain the proper names of any specific herbs, using this knowledge: '{docs}'"""
    
    return prompt 


def get_content(path):
    df_result = pd.read_csv(path)
    df_result["full_content"] = df_result.apply(lambda col: str(col["Tên dược liệu"]) + "\n\n" + str(col["abstract"]) + "\n\n" + str(col["content"]), axis=1)
    return df_result


def main(args, df_result, bard=None):     
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with tqdm(total=len(df_result)) as pbar:   
        for _, row in df_result.iterrows():      
            if f"{row['Tên dược liệu']}.txt" in [os.path.basename(p) for p in glob(f"{args.output}/*.txt")]:
                pbar.update(1)
                continue
            
            doc = None
            content = row["content"].split("\n")
            for i, c in enumerate(content):
                if "bài thuốc" in c.lower() or "đơn thuốc" in c.lower():
                    doc = "\n".join(content[i:])
                    break 
            
            if doc is None:
                pbar.update(1)
                continue 

            prompt = make_prompt(args.role, doc)

            if args.model == "bard":
                try:
                    reply = bard.get_answer(prompt)["content"]           
                except:
                    print(row['Tên dược liệu']) 
                    pbar.update(1)                   
                    continue

            elif args.model == "gpt3.5":
                messages=[{"role": "system", "content": f"You are a {args.role}.You are meeting some problems about your health, and you need to ask a Vietnamese traditional medicine doctor to get some treatment medicine."},
                          {"role": "user", "content": """As a {args.role}, please generate 5 Vietnamese questions naturally and generally about oriental medicine remedies to ask a Vietnamese traditional medicine doctor, specifically medicinal herbs used in Vietnamese Traditional Medicine. The generated questions should not contain the proper names of any specific herbs, using this knowledge:
                                                            "
                                                            6. Một số bài thuốc ứng dụng
                                                            Bài thuốc trị ho lâu ngày
                                                            Anh túc xác đem nướng mật, tán nhuyễn. Mỗi ngày uống 2gr pha với nước pha mật. (theo “Thế y đắc hiệu phương”)
                                                            Bài thuốc trị hen suyễn, lao, ho lâu năm, mồ hôi tự ra
                                                            Anh túc xác sao giấm 100gr, Ô mai 20gr. Cả 2 đem tán bột, mỗi lần uống 8gr trước khi đi ngủ. (theo “Tiểu Bách Lao Tán Tuyên Minh Phương”)
                                                            Bài thuốc trị lỵ lâu ngày
                                                            Anh túc xác, nướng với giấm, tán bột, trộn với mật làm hoàn. Ngày uống 6 ~ 8g với nước sắc gừng ấm (theo “Bản thảo cương mục”)
                                                            Bài thuốc chữa trẻ nhỏ bị thổ tả bạch lỵ, không muốn ăn uống
                                                            Anh túc xác (sao), Trần bì (sao), Kha tử (nướng, bỏ hạt), đều 40gr, Sa nhân, Chích thảo đều 8gr. Tất cả tán bột. Ngày uống 8 ~ 12g với nước cơm (theo “Anh Túc Tán – Phổ Tế Phương”).
                                                            7. Một số lưu ý khi dùng Anh túc xác
                                                            Trẻ em dưới 3 tuổi, những người cơ thể yếu, bị các bệnh gan thận hay các bé gái đang tuổi dậy thì không được sử dụng.
                                                            Những bệnh nhân mới bị ho hay bị lỵ cũng không dùng loại dược liệu này.
                                                            Anh túc xác là một vị thuốc chữa bệnh, nhưng mọi người không nên tự ý sử dụng, vì có thể dẫn đến việc bị nghiện nếu dùng quá liều. Và không chỉ riêng Anh túc xác, bất cứ một vị thuốc nào, bạn đọc cũng nên có sự cẩn trọng và cần tham khảo ý kiến thầy thuốc nếu muốn dùng nó để tránh việc sử dụng sai lầm đưa đến những tác dụng không mong muốn. Rất mong nhận được những phản hồi cũng như sự đồng hành của các bạn ở những bài viết kế tiếp. YouMed luôn sẵn sàng hỗ trợ bạn!
                                                            "
                                                    """},
                           {"role": "system", "content": """1. Chồng tôi bị ho kéo dài, bác sĩ hãy tư vấn giúp tôi một bài thuốc dân gian để giúp giảm triệu chứng đó cho anh ấy.
                                                            2. Bác sĩ có gợi ý nào về bài thuốc dân gian dành cho người bị lao lâu năm không?
                                                            3. Có phương pháp truyền thống nào hiệu quả để điều trị lỵ kéo dài không vậy bác sĩ?
                                                            4. Tôi đang lo lắng về trường hợp thổ tả bạch lỵ ở con tôi, có bài thuốc nào trong y học cổ truyền phù hợp không?
                                                            5. Ba chồng tôi bị chứng hen suyễn lâu năm, tôi không biết làm cách nào để giúp ba, bác sĩ có thể chia sẻ cho tôi một số biện pháp cổ truyền hiệu quả giúp giảm triệu chứng trên không?
                                                        """},                  
                                                        ]
                messages.append({"role": "user", "content": prompt})
                chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.5)
                reply = chat.choices[0].message.content

            out_name = f"{args.output}/{row['Tên dược liệu']}.txt"
            with open(out_name, "w") as f:
                f.write(reply)
                        
            # if os.stat(out_name).st_size < 1000:
            #     print("*****small*****", out_name)
            #     os.remove(out_name)
            
            pbar.update(1)
            


if __name__=="__main__":
    args = make_parser().parse_args()
    df_result = get_content(args.path)   
    
    bard = None 
    if args.model == "bard":          
        import requests      
        from bardapi import BardCookies, Bard                
        from bardapi.constants import SESSION_HEADERS        

        token = "g.a000gwgrPrkH6WbEv5rcMs0Bf20xSB0FxdiSG2xsw0NaK0_-lmkV__NvgDrtZHYnQgwBljOmpQACgYKAQQSAQASFQHGX2MivNEx_8Jv7GMqILdH5Aw9BBoVAUF8yKrYHeRip6aLqduALNX3CWGs0076."
        
        session = requests.Session()
        session.headers = SESSION_HEADERS        
        session.cookies.set("__Secure-1PSID", token)
        session.cookies.set("__Secure-1PSIDTS", "sidts-CjIBYfD7ZxBiEKJ01pQo0AL6Uj55sc_ggERaTftk47-FE-RP53433d8nksY-EJ67qq6SfBAA")
        session.cookies.set("__Secure-1PSIDCC", "ABTWhQGPBJ9FLY-QyloyhnHQH8TV5lijBizpmwpOLiUz8ohmiaoLBcreB53Rd3WtUL0sMePb2Q")

        bard = Bard(token=token, session=session)

        # cookie_dict = {
        #     "__Secure-1PSID": "g.a000gwgrPrkH6WbEv5rcMs0Bf20xSB0FxdiSG2xsw0NaK0_-lmkV__NvgDrtZHYnQgwBljOmpQACgYKAQQSAQASFQHGX2MivNEx_8Jv7GMqILdH5Aw9BBoVAUF8yKrYHeRip6aLqduALNX3CWGs0076",
        #     "__Secure-1PSIDTS": "sidts-CjIBYfD7ZxBiEKJ01pQo0AL6Uj55sc_ggERaTftk47-FE-RP53433d8nksY-EJ67qq6SfBAA",
        #     "__Secure-1PSIDCC": "ABTWhQGPBJ9FLY-QyloyhnHQH8TV5lijBizpmwpOLiUz8ohmiaoLBcreB53Rd3WtUL0sMePb2Q",            
        # }

        # bard = BardCookies(cookie_dict=cookie_dict)             

    elif args.model == "gpt3.5":
        import openai 
        openai.api_key = ""
    
    main(args, df_result, bard)     
