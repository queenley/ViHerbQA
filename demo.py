import re
import nltk
import torch
import gradio as gr
from nltk.util import ngrams
from collections import Counter
from transformers import pipeline
from difflib import SequenceMatcher
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")

class RepetitionRemover:
    """
    A class to handle repetitive text post-processing for NLP model outputs.
    Implements multiple strategies to detect and remove repetitions.
    """

    def __init__(self, language='vi'):
        """
        Initialize the repetition remover.

        Args:
            language (str): Language code ('vi' for Vietnamese, 'en' for English, etc.)
        """
        self.language = language
        # You may need to download NLTK packages once
        # nltk.download('punkt')

    def _get_phrases(self, text, min_phrase_len=3, max_phrase_len=10):
        """Extract all possible phrases from the text within length bounds."""
        words = text.split()
        phrases = []

        for n in range(min_phrase_len, min(max_phrase_len + 1, len(words))):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                phrases.append(phrase)

        return phrases

    def _find_repeated_phrases(self, text, min_occurrences=3, min_phrase_len=3, max_phrase_len=10):
        """Find phrases that repeat more than min_occurrences times."""
        phrases = self._get_phrases(text, min_phrase_len, max_phrase_len)
        phrase_counts = Counter(phrases)

        repeated = {phrase: count for phrase, count in phrase_counts.items()
                   if count >= min_occurrences}

        # Sort by count (most frequent first) and then by length (longer phrases first)
        return sorted(repeated.items(), key=lambda x: (-x[1], -len(x[0])))

    def _remove_overlapping_matches(self, matches):
        """Remove overlapping matches, keeping the longer ones."""
        if not matches:
            return []

        # Sort by start position and then by length (longest first)
        sorted_matches = sorted(matches, key=lambda m: (m[0], -len(m[2])))

        result = [sorted_matches[0]]
        for current in sorted_matches[1:]:
            prev = result[-1]
            # Check if current match overlaps with previous
            if current[0] >= prev[0] + len(prev[2]):
                result.append(current)

        return result

    def _get_repeated_ngrams(self, text, n_range=(2, 5), threshold=3):
        """Find repeated n-grams in the text."""
        words = text.split()
        repeated = []

        for n in range(n_range[0], n_range[1] + 1):
            ngram_counts = Counter(ngrams(words, n))

            for gram, count in ngram_counts.items():
                if count >= threshold:
                    repeated.append((' '.join(gram), count))

        return sorted(repeated, key=lambda x: (-x[1], -len(x[0])))

    def _get_longest_common_substring(self, s1, s2, min_length=5):
        """Find the longest common substring between two strings."""
        match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))

        if match.size >= min_length:
            return s1[match.a:match.a + match.size]
        return None

    def remove_repetitions(self, text, strategy='comprehensive'):
        """
        Remove repetitions from text using the specified strategy.

        Args:
            text (str): The text to process
            strategy (str): Strategy to use ('phrases', 'ngrams', 'substrings', or 'comprehensive')

        Returns:
            str: Processed text with repetitions removed
        """
        if not text:
            return text

        processed_text = text

        if strategy in ('phrases', 'comprehensive'):
            # Find and remove repeated phrases
            repeated_phrases = self._find_repeated_phrases(processed_text,
                                                          min_occurrences=2,
                                                          min_phrase_len=2)

            # Process longer phrases first to avoid removing parts of meaningful repetitions
            for phrase, count in repeated_phrases:
                if count > 2:  # Keep at least one occurrence
                    # Replace all but first occurrence
                    pattern = f"({re.escape(phrase)})((?:.*?\\1){{{count-1}}})"
                    processed_text = re.sub(pattern, r"\1", processed_text, flags=re.DOTALL)

        if strategy in ('ngrams', 'comprehensive'):
            # Further clean up using n-gram analysis
            words = processed_text.split()
            new_words = []

            i = 0
            while i < len(words):
                # Check for repetition in the next few words
                repeat_found = False

                for window in range(2, 6):  # Check for repeats of 2-5 word sequences
                    if i + window * 2 <= len(words):
                        seq1 = ' '.join(words[i:i+window])
                        seq2 = ' '.join(words[i+window:i+window*2])

                        if seq1 == seq2:
                            new_words.extend(words[i:i+window])
                            i += window * 2
                            repeat_found = True
                            break

                if not repeat_found:
                    new_words.append(words[i])
                    i += 1

            processed_text = ' '.join(new_words)

        # Final clean-up for any remaining simple repetitions
        processed_text = re.sub(r'(\b\w+\b)(\s+\1)+', r'\1', processed_text)

        return processed_text

    def cleanup_vietnamese_text(self, text):
        """Special handling for Vietnamese text."""
        # Step 1: Remove basic direct repetitions
        processed = re.sub(r'(\b[\w\s]+\b)(,\s*\1)+', r'\1', text)

        # Step 2: Apply general repetition removal
        processed = self.remove_repetitions(processed)

        # Step 3: Additional Vietnamese-specific processing
        # Handle comma-separated lists with repetitions
        phrases = processed.split(', ')
        unique_phrases = []
        for phrase in phrases:
            if phrase not in unique_phrases:
                unique_phrases.append(phrase)
        processed = ', '.join(unique_phrases)

        return processed

def postprocess(text, language='vi'):
    """
    Process NLP model output to remove repetitions.

    Args:
        text (str): Text to process
        language (str): Language code

    Returns:
        str: Processed text with repetitions removed
    """
    remover = RepetitionRemover(language=language)

    if language == 'vi':
        return remover.cleanup_vietnamese_text(text)
    else:
        return remover.remove_repetitions(text)


# Initialize with empty models (will load on selection)
loaded_models = {}
loaded_tokenizers = {}
# Function to load model on demand
def load_model(model_name):
    if model_name not in loaded_models:
        model_path = models[model_name] + "best_model"
        tokenizer_path = models[model_name] + "best_tokenizer"
        print(f"Loading {model_name} from {model_path}...")
        loaded_models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda:0')
        loaded_tokenizers[model_name] = AutoTokenizer.from_pretrained(tokenizer_path)
    return loaded_models[model_name], loaded_tokenizers[model_name]


# Function to generate answer with the selected model
def generate_answer(model, tokenizer, input_text, q_len, t_len, device='cuda:0'):
    inputs = tokenizer(input_text,
                      max_length=q_len,
                      padding="max_length",
                      truncation=True,
                      pad_to_max_length=True,
                      add_special_tokens=True,
                      return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=t_len,
            use_cache=True,
        )

    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    cleaned_answer = postprocess(predicted_answer)
    final_answer = cleaned_answer
    # \n*** Lưu ý, thông tin trên chỉ mang tính chất tham khảo. Để hiểu rõ hơn bạn nên liên hệ với bác sĩ chuyên môn. ***"
    return final_answer

# Dictionary of available models
models = {
    "ViHerbQA-Base": "/content/drive/MyDrive/RESEARCHES/ViHerbQA/model/base/close-book/",
    "ViHerbQA-Large": "/content/drive/MyDrive/RESEARCHES/ViHerbQA/model/large/close-book/",      
}

load_model("ViHerbQA-Large")
load_model("ViHerbQA-Base")


with gr.Blocks() as demo:
  gr.Markdown("# ViHerbQA")
  gr.Markdown("Trò chuyện với ViHerbQA - mô hình hỏi đáp dược liệu Y học cổ truyền Việt Nam")

  with gr.Row():
      model_dropdown = gr.Dropdown(
          choices=list(models.keys()),
          value="ViHerbQA-Large",
          label="Select Model"
      )

  chatbot = gr.ChatInterface(
      fn=lambda message, history, model_name: generate_answer(*load_model(model_name), message, 1024, 1024),
      additional_inputs=[model_dropdown],
      chatbot=gr.Chatbot(height=600),
      examples=[
          ["Actiso có tên khoa học là gì?"],
          ["Những ai không nên dùng Hoa nhài?"],
          ["Lá Thường xuân có hiệu quả như thế nào trong việc điều trị bệnh nào về đường hô hấp?"]
      ],
  )

if __name__ == "__main__":
    demo.launch(debug=True, show_error=True)

  
