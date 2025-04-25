from deep_translator import GoogleTranslator
import time
import nltk


# Make sure to download the Punkt tokenizer models if not already done

nltk.download("punkt")
nltk.download("punkt_tab")


# Method to split text into chunks while avoiding abrupt sentence splitting
def split_text_into_chunks(text, max_chunk_size=100):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_length = len(sentence.split())  # Word count of the sentence
        if current_chunk_size + sentence_length <= max_chunk_size:
            current_chunk.append(sentence)
            current_chunk_size += sentence_length
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_chunk_size = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add the last chunk

    return chunks


# Secondary method to handle long text translation
def translate_long_text(text, src_lang, target_lang):
    chunks = split_text_into_chunks(text)
    translated_chunks = []

    for chunk in chunks:
        translated = translate(chunk, src_lang, target_lang)
        if translated:
            translated_chunks.append(translated)
        else:
            return None  # Return None if any chunk fails to translate

    return " ".join(translated_chunks)  # Join all translated chunks into one string


def translate(text, src_lang, target_lang):
    retries = 5
    for attempt in range(retries):
        try:
            translated = GoogleTranslator(
                source=src_lang, target=target_lang
            ).translate(text)
            return translated
        except Exception as e:
            print(f"Error occurred during translation: {e}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None


# ad hoc translation for the label only, to match check4facts format
def translate_label(text):
    text = text.strip()
    if "relatively inaccurate" in text.lower():
        return 2
    elif "relatively accurate" in text.lower():
        return 3
    elif "inaccurate" in text.lower():
        return 1
    elif "accurate" in text.lower():
        return 4
    elif "unverifiable" in text.lower():
        return 0
    else:
        return str(text).strip()
