from requests.exceptions import SSLError
from bs4 import BeautifulSoup, Comment
import requests
import sys

sys.path.append("scripts")
from check4facts.scripts.rag.text_embedding import *
import numpy as np
import pandas as pd
import datetime
from scipy.spatial.distance import cosine
from langdetect import detect
import nltk
import stanza

# stanza.download('el')
nlp = stanza.Pipeline("el")
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import re
import time
from sentence_transformers import SentenceTransformer, util
import logging
import warnings

logging.getLogger("stanza").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)


class Harvester:

    def __init__(self, url_list, claim, timeout, claim_id, max_sources):
        self.timeout = timeout
        self.url_list = url_list
        self.claim = claim
        self.claim_id = claim_id
        self.max_sources = max_sources
        self.timeout_seconds = 20 * 60
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    def signal_handler(signum, frame):
        raise TimeoutError("Function execution timed out")

    def remove_nonstandard_chars(self, text):
        # Only keep standard characters (e.g., letters, numbers, spaces, and punctuation)
        cleaned_text = re.sub(r"[^\x00-\x7F]+", "", text)  # ASCII range
        return cleaned_text

    def get_relevant_sentences(self, claim, body, threshold):

        doc = nlp(body)

        # Extract sentences
        body_sentences = [
            sentence.text for sentence in doc.sentences if not self.is_english(sentence)
        ]

        a = claim
        b = pd.DataFrame(body_sentences)

        # Compute embeddings
        claim_emb = self.model.encode(claim)
        body_sen_emb = self.model.encode(body_sentences)

        # Compute cosine similarities
        cosine_scores = util.cos_sim(claim_emb, body_sen_emb)

        # Extract most relevant segments based on a threshold
        relevant_segments = [
            segment
            for score, segment in zip(cosine_scores[0], body_sentences)
            if score >= threshold
        ]

        return " ".join(relevant_segments)

    def get_html_text(self, url):

        try:
            response = requests.get(url, timeout=self.timeout, verify=False)
            time.sleep(2)
            if response.status_code == 200:
                return BeautifulSoup(response.text, "html.parser")
            else:
                return None
        except SSLError as e:
            print("SSL Error:", e)
            print("On url: ", url)
            return None
        except Exception as e:

            print("An error occurred:", e)
            print("On url: ", url)
            return None

    def get_title(self, soup):

        title = soup.find(lambda tag: "title" in tag.get("class", []))
        if title:
            return title.text.strip()
        elif (
            soup.find("meta", property="og:title")
            and soup.find("meta", property="og:title").get("content") is not None
        ):
            title = soup.find("meta", property="og:title").get("content").strip()
        elif soup.find("h1"):
            title = soup.find("h1").text.strip()
        else:
            return None
        return title

    # def text_preprocess(self, texts):

    #     filtered_arr = [text for text in texts if len(text) > 1 and len(text.split()) > 3]
    #     return " ".join(filtered_arr)

    def get_body(self, soup, claim):

        [
            s.extract()
            for s in soup(["style", "script", "[document]", "head", "title", "footer"])
        ]
        # [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title',
        #                     'footer', 'nav', 'aside', 'header', 'form', 'img', 'blockquote', 'meta'])]

        body_text = soup.get_text(separator="\n", strip=True)

        texts = body_text.split("\n")

        result = "\n".join(
            text.strip() for text in texts if len(text.split()) > 3
        )  # .replace("\n", " ")

        if texts is not None:
            try:
                # get the most similar body
                claim_emb = single_text_embedding(claim)
                body_emb = single_text_embedding(result)
                dot_product = cos_sim(claim_emb, body_emb)
            except Exception as e:
                print(f"Input is too large. Skipping this web source....")
                body_emb = None

        else:
            dot_product = None

        return result, dot_product

    def similarity_text(self, claim, texts):
        claim_emb = single_text_embedding(claim)

        # spaw to texts se paragrafous kai vriskw most similar paragraph
        # for each paragraph spaw se protaseis? kai vriskw similar sentence

        paragraphs = texts.split("\n")
        # sentences = sent_tokenize(texts)

        # paragraph_tuples
        tuples = [
            (cos_sim(claim_emb, single_text_embedding(text)), text)
            for text in paragraphs
            if not self.is_english(text)
            and len(text.split()) > 3
            and single_text_embedding(text) is not None
        ]
        if not tuples:
            print("paragraph tuples is empty")
            print(texts)
            print("---------------")
            print(paragraphs)
        # #sentence_tuples
        # tuples2  = [( np.dot(claim_emb, single_text_embedding(text)), text) for text in sentences if not self.is_english(text)
        #             and len(text.split())>3 and single_text_embedding(text) is not None]
        # if not tuples2:
        #     print('sentence tuples is empty')

        # tuples2  = [( np.dot(claim_emb, single_text_embedding(p)), text) for s in paragraph if not self.is_english(p)]
        if tuples:
            similarity, result = max(tuples, key=lambda x: x[0])
        else:
            similarity, result = None, None
        # if tuples2:
        #     similarity2 , result2 = max(tuples2, key=lambda x: x[0])
        # else:
        #     similarity2, result2 = None,None

        return similarity, result  # , similarity2, result2

    def run(self):

        df = pd.DataFrame(
            columns=[
                "id",
                "claim_id",
                "title",
                "body",
                "body_similarity",
                "most_similar_paragraph",
                "harvest_date",
                "url",
                "most_similar_par_cos",
                "similar_sentences",
            ]
        )

        for url in self.url_list:

            print(f"Harvesting url: {url}")

            html = self.get_html_text(url)
            if html is None:
                print(
                    f"""Invalid url: {url}, 
                      skipping procedure...."""
                )
                continue
            title = self.get_title(html)
            if title is None or len(title) <= 3:
                print("No title found")
                print("skipping procedure....")
                title = ""
            body, body_dot = self.get_body(html, claim=self.claim)

            if body is None or len(body.split()) < 50:
                print("body is none")
                print("skipping procedure....")
                continue

            if len(body) > 500000:
                print("Web page is too long")
                print("skipping procedure....")
                continue

            similarity_p, result_p = self.similarity_text(self.claim, body)
            sim_sentences = self.get_relevant_sentences(self.claim, body, threshold=0.4)

            data = {
                "id": len(df),
                "claim_id": self.claim_id,
                "title": title,
                "body": body.replace("\n", " "),
                "body_similarity": body_dot,
                "most_similar_paragraph": (
                    result_p.replace("\xa0", "") if result_p is not None else ""
                ),
                "harvest_date": datetime.date.today(),
                "url": url,
                "most_similar_par_cos": similarity_p,
                "similar_sentences": sim_sentences,
            }

            df.loc[len(df)] = data

            if len(df) >= self.max_sources:
                break

            print()
        #             print(f'''Most similar paragraph: {result_p}
        # Cosine similarity: {similarity_p}
        # Body similarity: {body_dot}
        #                   ''')
        # print(f'''Information: {sim_sentences}''')

        return df

    def is_english(self, text):
        try:
            return detect(text) == "en"
        except:
            # Handle cases where language detection fails
            return False
