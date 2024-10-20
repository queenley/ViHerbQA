import json
import pandas as pd
from tqdm import tqdm
import time
from glob import glob
import argparse
import os


def make_parser():    
    parser = argparse.ArgumentParser("Generate questions by LLM")
    parser.add_argument("-m", "--model", 
                        type=str, 
                        default='bard', 
                        choices=['bard', 'gpt3.5', 'gpt4.0', 'gemini-pro', 'gemini-1.5-pro-latest'],
                        help="LLM is called to generate")
    
    parser.add_argument("-p", "--path", 
                        type=str, 
                        default='y_hoc_co_truyen_youmed_v1.csv', 
                        help="path of documents")
    
    parser.add_argument("-r", "--role", 
                        type=str, 
                        default='student', 
                        choices = ['student', 'patient'], 
                        help="path of documents")
    
    parser.add_argument("-o", "--output",
                        type=str,
                        default="Questions",
                        help="output directory")
    
    parser.add_argument("-t", "--type",
                        type=str,
                        default="basic",
                        help="type of promting")
    
    return parser 


def make_prompt(role, docs, herb, type):
    
    prompt = f"""As a {role}, generate 20 Vietnamese questions about <{herb.upper()}> which is a medicinal herb Vietnamese Traditional Medicine using this knowledge: 
                "
                {docs}
                "
            """
    
    if type == "cot":
        prompt = f"""            
            Q: As a {role}, Generate 20 Vietnamese questions about <{herb}> which is a medicinal herb Vietnamese Traditional Medicine using this knowledge:
            "{docs}"
            """
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
                
            prompt = make_prompt(args.role, row["full_content"], row["Cây thuốc"], args.type)

            if args.model == "bard":
                try:
                    reply = bard.get_answer(prompt)["content"]           
                except:
                    print(row['Tên dược liệu'])                    
                    continue

            elif args.model == "gpt3.5":
                messages=[{"role": "system", "content": f"You are a {args.role} relating medicinal herb Vietnamese Traditional Medicine."},
                          {"role": "user", "content": """As a {args.role}, Generate 20 Vietnamese questions about <Bướm bạc (Hồ điệp)> which is a medicinal herb Vietnamese Traditional Medicine using this knowledge:
                                                            "
                                                            Bướm bạc (Hồ điệp): Bí ẩn vị thuốc cùng tên loài bướm
                                                            Từ lâu, Bướm bạc là dược liệu thường được sử dụng để điều trị bệnh trong Đông y. Đặc biệt, vị thuốc có khả năng làm thanh nhiệt, chữa cảm nắng. Bài viết sau sẽ giúp bạn hiểu rõ hơn về đặc điểm, công dụng và cách dùng của thảo dược này.
                                                            1. Giới thiệu về Bướm bạc
                                                            Tên gọi khác: Bướm bạc, Bươm bướm, Bứa chùa, Hoa bướm, Hồ điệp…
                                                            Tên khoa học: Herba Mussaendae pubenscentis.
                                                            Thuộc họ Cà phê (Rubiaceae).
                                                            1.1. Đặc điểm sinh trưởng và thu hái
                                                            Cây Bướm bạc sinh sống nhiều ở Trung Quốc và Bắc Việt Nam. Ở nước ta, theo tài liệu của Viện Dược liệu, loài này có gặp ở các tỉnh vùng Tây Bắc. Đây là cây ưa ẩm và ưa sáng. Cây mọc hoang khắp nơi, thường gặp ở đồi núi, ven rừng.
                                                            Thu hái bộ phận rễ và thân suốt quanh năm, lá thường dùng tươi. Còn hoa thì thu hoạch từ tháng 6 đến tháng 7 hằng năm. Các bộ phận của cây Bướm bạc dùng tươi hay khô đều được. Nếu dùng khô thì đem thảo dược rửa sạch và phơi hoặc sấy khô.
                                                            1.2. Mô tả toàn cây
                                                            Cây Bướm bạc là loại cây nhỏ, mọc trườn cao từ 1m đến 2m. Các cành non có chứa lông mịn.
                                                            Lá nguyên mọc đối nhau, dài 4 – 9cm, rộng 1,5 – 4,5cm. Mặt trên có màu xanh lục sẫm, mặt dưới có lông tơ mịn. Lá kèm hình sợi.
                                                            Cụm hoa xim mọc ở đầu cành. Hoa màu vàng, có lá đài phát triển thành từng bản màu trắng. Trước khi ra hoa, cành xuất hiện một chùm lá bắc màu trắng bạc hình trứng rũ trông như những cánh bướm bao bọc bông hoa trông rất đẹp nên còn có tên gọi là hoa bươm bướm. Tràng hoa 5 cánh, ống tràng dài và hẹp, nhị 5 dính vào chỗ loe của ống tráng, bầu 2 ô, nhiều noãn.
                                                            Hoa của cây Bướm bạc
                                                            Quả hình cầu, dài 6 – 9mm, rộng 6 – 7mm, màu đen, có gân dọc trên quả, nhẵn. Quả có chứa nhiều hạt nhỏ màu đen, nếu vò sẽ thấy chất dính. Ra hoa kết quả vào mùa hè.
                                                            1.3. Bộ phận làm thuốc bào chế
                                                            Bộ phận được dùng đó chính là hoa, thân, rễ, lá của cây. Riêng rễ và thân được dùng nhiều hơn.
                                                            Khi hái về, rửa sạch, cắt ra từng khúc. Sau đó phơi khô, cho vào túi nylon để bảo quản và sử dụng dần. Lá thường dùng tươi.
                                                            >> Xem thêm: Mộc thông: Vị thuốc có công dụng lợi tiểu.
                                                            1.4. Bảo quản
                                                            Bảo quản ở những nơi thoáng mát, nhiệt độ phòng, đóng gói kỹ càng trong bao bì sau mỗi lần sử dụng đối với thuốc sấy khô.
                                                            2. Thành phần hóa học
                                                            Bướm bạc có chứa các thành phần sau:
                                                            Toàn cây chứa acid cafeic, acid ferulic, acid cumaric, beta-sitosterol-D glucosid (Trung dược từ hải I,1993). Ngoài ra còn có saponin, triterpenic, mussaendosid O, P, Q, R, S.
                                                            Lá chứa hợp chất acid amin, phenol, acid hữu cơ, đường, beta-sitosterol.
                                                            Thân có beta-sitosterol và acid arjunblic.
                                                            3. Công dụng
                                                            3.1. Y học hiện đại
                                                            Hoa Bướm bạc được dùng làm thuốc lợi tiểu, chữa ho hen, sốt cách nhật. Dùng ngoài giã nát đắp lên những nơi sưng tấy, gãy xương. 
                                                            Rễ, cành và thân Bướm bạc dùng làm thuốc giảm đau, chữa tê thấp, khí hư bạch đới (mệt mỏi, chán ăn, dịch âm đạo màu trắng xuất hiện bất thường…).
                                                            Viện Y học Cổ truyền đã xây dựng một phác đồ điều trị ho, viêm họng đỏ hoặc viêm amidan cấp với lá và thân Bướm bạc 150g/ngày, sắc uống trong 3 ngày.
                                                            3.2. Y học cổ truyền
                                                            Tính vị hơi ngọt, tính mát.
                                                            Quy kinh Phế, Tâm, Can.
                                                            Công dụng: thanh nhiệt, giải biểu (làm ra mồ hôi đưa tà khí ra ngoài), giải uất, lương huyết (làm mát), tiêu viêm.
                                                            3.3. Cách dùng và liều dùng
                                                            Tùy thuộc vào mục đích sử dụng và từng bài thuốc mà có thể dùng dược liệu với nhiều cách khác nhau. Ví dụ như sắc lấy nước uống, tán thành bột mịn, làm viên hoàn hay sử dụng ngoài da.
                                                            Cụ thể:
                                                            Toàn cây: 15 – 30g dưới dạng thuốc sắc.
                                                            Hoa: 6 – 12g dưới dạng thuốc sắc. Dùng ngoài không kể liều lượng.
                                                            Rễ: 10 – 20g dưới dạng thuốc sắc. Cành, thân lá 6 – 12g.
                                                            Một số đối tượng có thể sử dụng dược liệu để điều trị bệnh:
                                                            Sổ mũi, say nắng.
                                                            Ho hen, hen suyễn.
                                                            Gãy xương, phong tê thấp, chấn thương.
                                                            Các bệnh ngoài da: mụn nhọt, lở loét, chốc ghẻ…
                                                            Dược liệu Bướm bạc thường được sử dụng dưới dạng thuốc sắc để trị bệnh
                                                            4. Một số bài thuốc kinh nghiệm
                                                            4.1. Điều trị say nắng
                                                            Sử dụng khoảng 60 – 90g thân và rễ Bướm bạc khô, đun sôi với 1 lít nước và nấu nước uống thay chè.
                                                            4.2. Chữa bệnh sổ mũi, say nắng
                                                            12g thân cây Bướm bạc, 3g Bạc hà, 10g lá Ngũ trảo. Đem tất cả nguyên liệu trên rửa sạch và để ráo nước. Đun sôi cùng với nước để dùng thay thế cho nước trà hằng ngày.
                                                            4.3. Chữa bệnh ho, sốt, sưng amidan
                                                            30g rễ cây Bướm bạc, 10g rễ Bọ mẩy, 20g Huyền sâm. Đem tất cả các vị thuốc của thang thuốc trên rửa sạch, sau đó sắc với một lượng nước phù hợp và sử dụng.
                                                            4.4. Trị viêm thận, phù, giúp lợi tiểu
                                                            Thân Bướm bạc 30g, Kim ngân hoa 60g, Mã đề 30g, sắc nước uống.
                                                            4.5. Chữa sốt, khô khát, táo bón, tân dịch khô kiệt
                                                            Rễ Bướm bạc 60g, Hành tăm 12g (đều sao vàng), sắc uống.
                                                            4.6. Chữa bệnh khí hư bạch đới
                                                            Đem 10 đến 20g rễ thảo dược, rửa sạch, sắc kỹ với nước lọc một lượng phù hợp và sử dụng mỗi ngày. 
                                                            4.7. Chữa lở loét da
                                                            Dùng lá cây Mướp tươi và lá cây Bướm bạc tươi liều lượng hai thứ bằng nhau. Đem đi rửa sạch, để ráo rồi giã nát ra đắp bã vào các vùng lở loét da cố định lại, sau đó rửa lại với nước sạch.
                                                            5. Lưu ý
                                                            Không dùng cho trẻ nhỏ và phụ nữ có thai hay dị ứng với bất kỳ thành phần nào có trong bài thuốc.
                                                            Bướm bạc là một vị thuốc cổ truyền được sử dụng từ rất lâu trong dân gian. Nhờ có nhiều tác dụng quý mà dược liệu này được dùng nhiều trong các bài thuốc chữa bệnh cũng như cuộc sống hằng ngày. Tuy nhiên, để có thể phát huy hết công dụng của vị thuốc đối với sức khỏe, bạn nên tham khảo ý kiến bác sĩ để kiểm soát rủi ro và những tác dụng không mong muốn. Hãy chia sẻ bài viết nếu thấy hữu ích.
                                                            "
                           """},
                           {"role": "system", "content": """1. Làm thế nào Bướm bạc, với tên gọi phổ biến là Hồ điệp, tương tác với các cơ chế sinh học trong cơ thể để thực hiện các công dụng y học của nó?
                                                            2. Trong y học cổ truyền, liệu có những quy tắc cụ thể nào để lựa chọn liều lượng và cách sử dụng Bướm bạc dựa trên tình trạng sức khỏe và cơ địa của mỗi người?
                                                            3. Bướm bạc làm thế nào để làm giảm viêm và đau trong cơ thể? Cơ chế hoạt động này có ảnh hưởng đến hệ thống miễn dịch như thế nào?
                                                            4. Trong y học hiện đại, đã có những nghiên cứu cụ thể nào xác nhận hiệu quả và an toàn của Bướm bạc trong điều trị các bệnh lý cụ thể không?
                                                            5. Liệu có sự khác biệt nào giữa việc sử dụng Bướm bạc trong y học cổ truyền và y học hiện đại, đặc biệt là trong cách tiếp cận và đánh giá hiệu quả của nó?
                                                            6. Bướm bạc có thể được sử dụng trong các phương pháp điều trị kết hợp hay không? Nếu có, liệu có những cách nào để tối ưu hóa hiệu quả của việc kết hợp này?
                                                            7. Làm thế nào để đảm bảo an toàn khi sử dụng Bướm bạc, đặc biệt là khi tự điều trị hoặc kết hợp với các loại thuốc khác?
                                                            8. Trong trường hợp cần thiết, liệu có những biện pháp thay thế nào có thể sử dụng để thay thế hoặc bổ sung cho việc sử dụng Bướm bạc?
                                                            9. Bướm bạc có những ảnh hưởng đến các cơ quan nội tạng khác ngoài cơ chế giảm đau và giảm viêm không?
                                                            10. Có những yếu tố nào nên được xem xét khi đưa ra quyết định về việc sử dụng hoặc không sử dụng Bướm bạc trong các phương pháp điều trị?
                                                            11. Có những nghiên cứu nào về tương tác giữa Bướm bạc và các loại thuốc hoặc thực phẩm khác mà chúng ta nên biết?
                                                            12. Làm thế nào để xác định chất lượng của Bướm bạc khi mua từ các nguồn cung cấp khác nhau?
                                                            13. Có những hạn chế hay rủi ro gì khi sử dụng Bướm bạc, đặc biệt là trong nhóm người dễ bị ảnh hưởng như trẻ em, phụ nữ mang thai hoặc cho con bú, người già, và người mắc các bệnh mãn tính?
                                                            14. Có những tác động phụ nào có thể xuất hiện khi sử dụng Bướm bạc trong điều trị dài hạn?
                                                            15. Trong quá trình điều trị bằng Bướm bạc, liệu có cần phải thay đổi cách sống hoặc chế độ ăn uống không?
                                                            16. Có nên sử dụng Bướm bạc trong trường hợp tự điều trị các triệu chứng nhẹ mà không tham khảo ý kiến của chuyên gia y tế không?
                                                            17. Làm thế nào để đảm bảo tuân thủ đúng liều lượng và cách sử dụng khi sử dụng Bướm bạc?
                                                            18. Liệu việc sử dụng Bướm bạc có thể gây ra tình trạng phụ thuộc hay nghiện nếu sử dụng kéo dài?
                                                            19. Bướm bạc có thể ảnh hưởng đến quá trình tiêu hóa hay hấp thụ chất dinh dưỡng không? Nếu có, thì làm thế nào?
                                                            20. Trong y học cổ truyền, liệu có những lời khuyên nào về cách lựa chọn, thu hái và bảo quản Bướm bạc để đảm bảo hiệu quả và an toàn?
                            """}]
                messages.append({"role": "user", "content": prompt})
                chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=1)
                reply = chat.choices[0].message.content

            
            elif "gemini" in args.model:
                model = genai.GenerativeModel(args.model)
                try:
                    response = model.generate_content(prompt)
                    reply = response.text
                except:
                    print(row['Tên dược liệu'])
            
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
    if args.model == "gpt3.5":
        import openai 
        openai.api_key = ""

    elif "gemini" in args.model:
        import google.generativeai as genai        
        GOOGLE_API_KEY=''
        genai.configure(api_key=GOOGLE_API_KEY)        
    
    
    main(args, df_result, bard)     
