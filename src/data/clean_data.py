import re
import pandas as pd
from tqdm import tqdm


def remove_reference(text):
    pattern = r'\d+Nguồn tham khảo.*?(http\S+)?Ngày tham khảo: \d{2}/\d{2}/\d{4}'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text


def clean_newline(cleaned_text):
    output = re.sub(r'\n(?=[a-zà-ỹ])', '', cleaned_text)
    return output


if __name__ == "__main__":
    viherbqa_official = pd.read_csv("dataset/viherbqa_official_cleaned.csv")
    viherbqa_official.fillna("", inplace=True)
    tqdm.pandas()
    viherbqa_official["remove_reference"] = viherbqa_official["context"].progress_apply(lambda x: remove_reference(x))
    viherbqa_official["clean_newline"] = viherbqa_official["remove_reference"].progress_apply(lambda x: clean_newline(x))

    viherbqa_official.to_csv("dataset/viherbqa_official_cleaned_v2.csv", index=False)