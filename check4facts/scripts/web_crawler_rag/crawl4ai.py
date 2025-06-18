from sentence_transformers import SentenceTransformer, util
import torch
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from markdown import markdown
from bs4 import BeautifulSoup
import nltk
import time
import numpy as np
import re
from check4facts.scripts.rag.groq_api import *
from check4facts.scripts.rag.gemini_llm import *
from check4facts.scripts.rag.ollama_llm import *
from check4facts.scripts.rag.mistral_llm import mistral_llm
from check4facts.scripts.rag.search_api import google_search, search_queries
from check4facts.scripts.web_crawler_rag.search_engine import SearchEngine
from check4facts.scripts.web_crawler_rag.url_generation import url_generation
import aiohttp
import requests
import os
import hashlib
from datetime import datetime
import pymupdf4llm
import langdetect


nltk.download("punkt")


class crawl4ai:
    def __init__(self, claim, web_sources, article_id, provided_urls):
        self.claim = claim
        self.web_sources = web_sources
        self.article_id = article_id
        # self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        self.model = SentenceTransformer("sentence-transformers/LaBSE")
        self.provided_urls = provided_urls
        self.search_engine = SearchEngine(2)

    def get_urls(self):
        # if no urls were provided
        if not self.provided_urls:
            # traditional blacklist searching api
            # print("No urls provided. Searching the web for urls....")
            # self.urls = google_search(self.claim, self.web_sources)
            # return self.urls
            # search api with llm assisted domain query generation
            generator = url_generation(self.claim)
            queries = generator.generate_queries()
            # calling the api
            # self.urls = search_queries(queries)
            # calling the self hosted search engine
            self.urls = self.search_engine.call_engine(queries)
            for url in self.urls:
                print(url)
                print("=========================================================")
            return self.urls, queries

        else:
            self.urls = self.provided_urls
            return self.urls, ""

    def chunk_text(self, text, chunk_size=500, overlap_size=50):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # print(current_chunk)
                # print("-----------------------------------")
                chunks.append(current_chunk)
                current_chunk = sentence[:overlap_size]
                current_chunk += sentence[overlap_size:]

        if current_chunk:
            # print(current_chunk)
            # print("-----------------------------------")
            chunks.append(current_chunk)

        return chunks

    def get_dynamic_threshold(chunk_text):
        lang = langdetect.detect(chunk_text)
        return 0.4 if lang == "el" else 0.3

    def get_sim_text(
        self,
        text,
        min_threshold=0.2,
        chunk_size=500,
    ):
        if not text:
            return []
        claim_embedding = self.model.encode(
            self.claim, convert_to_tensor=True, show_progress_bar=False
        )
        filtered_results = []
        chunks = self.chunk_text(text, chunk_size)
        if not chunks:
            return []
        # print("LEN OF CHUNKS OF THE FILE IS: ")
        # print(len(chunks))
        chunk_embeddings = self.model.encode(
            chunks, convert_to_tensor=True, show_progress_bar=False
        )
        chunk_similarities = util.cos_sim(claim_embedding, chunk_embeddings)
        for chunk, similarity in zip(chunks, chunk_similarities[0]):
            if similarity >= min_threshold:
                print(chunk)
                print()
                print(similarity)
                print("--------------------------------------------------")
                filtered_results.append(chunk)
        # print(f"NUMBER OF FILTERED CHUNKS IS: {len(filtered_results)}")
        if len(filtered_results) == 0:
            return []
        return filtered_results

    def single_text_embedding(self, text):
        embedding = self.model.encode(
            text, convert_to_tensor=True, show_progress_bar=False
        )
        torch.cuda.empty_cache()
        return embedding

    def cos_sim(embedding1, embedding2):
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def convert_markdown_to_text(self, markdown_str: str) -> str:
        html = markdown(markdown_str)
        plain_text = BeautifulSoup(html, "html.parser").get_text()
        return plain_text

    def run_groq(self, info, article_id):
        api = groq_api(info, self.claim)
        response = api.run_api(article_id)
        return response

    def run_gemini(self, info, article_id):
        try:
            gemini_instance = gemini_llm(query=self.claim, external_knowledge=info)
            answer = gemini_instance.google_llm(article_id)
        except Exception as e:
            print(f"Running gemini llm failed: {e}")
            answer = None
        return answer

    def run_mistral(self, info, article_id):
        mistral_instance = mistral_llm(query=self.claim, external_knowledge=info)
        answer = mistral_instance.run_mistral_llm(article_id)
        return answer

    def run_ollama(self, info, article_id):
        try:
            ollama_instance = ollama_llm(
                query=str(self.claim), external_knowledge=str(info)
            )
            answer = ollama_instance.run_ollama_llm(article_id)
            return answer
        except Exception as e:
            print(f"Running local llm failed: {e}")
            return None

    def harvest_pdf(self, pdf_url):
        # create a file name from its hashed url and save it temporarily to read it
        os.makedirs("data", exist_ok=True)
        url_hash = hashlib.md5(pdf_url.encode()).hexdigest()
        filename = f"{url_hash}.pdf"
        save_path = os.path.join("data", filename)

        response = requests.get(pdf_url)
        if response.status_code == 200:
            try:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"PDF downloaded successfully to: {'./data'}")

                # extract text from pdf
                md_text = pymupdf4llm.to_markdown(save_path)
                result = self.convert_markdown_to_text(md_text)
                return result

            # delete the file after the processing is done
            finally:
                if os.path.exists(save_path):
                    os.remove(save_path)
                    print(f"Deleted downloaded file: {save_path}")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")

    async def get_external_knowledge(self):
        urls, queries = self.get_urls()
        # store the urls that actually provided us with information
        content_urls = []
        final_info = ""
        # for debugging
        # urls = [
        #     "https://www.hellenicparliament.gr/UserFiles/8c3e9046-78fb-48f4-bd82-bbba28ca1ef5/vouli_en_low_28_1_09.pdf"
        # ]
        # urls = [
        #     "https://www.hellenicparliament.gr/UserFiles/8c3e9046-78fb-48f4-bd82-bbba28ca1ef5/vouli_site_version.pdf"
        # ]
        html_urls = []
        pdf_urls = []

        # async with aiohttp.ClientSession() as session:
        for url in urls:
            if url.lower().endswith(".pdf"):
                pdf_urls.append(url)
            else:
                # extra block of code for checking for pdf files (unnecessary for now)
                # try:
                #     async with session.head(
                #         url, allow_redirects=True, timeout=5
                #     ) as resp:
                #         content_type = resp.headers.get("Content-Type", "").lower()
                #         if "pdf" in content_type:
                #             pdf_urls.append(url)
                #         else:
                #             html_urls.append(url)
                # except Exception as e:
                #     print(f"[HEAD ERROR] {url} => {e}")
                html_urls.append(url)  # fallback

        # run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=True)
        # async with AsyncWebCrawler() as crawler:

        #     # parsing the non-pdf urls
        #     print(f"URLS to be harvested {html_urls}")
        #     async for result in await crawler.arun_many(html_urls, config=run_conf):

        #         if result.success:
        #             print(
        #                 f"[OK] {result.url}, length: {len(result.markdown.raw_markdown)}"
        #             )
        #         else:
        #             print(f"[ERROR] {result.url} => {result.error_message}")

        #     run_conf = run_conf.clone(stream=False)
        #     results = await crawler.arun_many(html_urls, config=run_conf)
        #     similar_texts = []
        #     for res in results:
        #         if res.success:

        #             print(f"[OK] {res.url}, length: {len(res.markdown.raw_markdown)}")
        #             similar_chunks = self.get_sim_text(
        #                 self.convert_markdown_to_text(res.markdown.raw_markdown),
        #             )
        #             if similar_chunks:
        #                 content_urls.append(res.url)
        #             similar_texts.append(similar_chunks)
        #         else:
        #             print(f"[ERROR] {res.url} => {res.error_message}")
        run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun_many(html_urls, config=run_conf)
            similar_texts = []

            for res in results:
                if res.success:
                    print(f"[OK] {res.url}, length: {len(res.markdown.raw_markdown)}")
                    similar_chunks = self.get_sim_text(
                        self.convert_markdown_to_text(res.markdown.raw_markdown),
                    )

                    if similar_chunks != []:
                        content_urls.append(res.url)

                    similar_texts.append(similar_chunks)
                else:
                    print(f"[ERROR] {res.url} => {res.error_message}")

            # parsing the pdf files
            # for pdf in pdf_urls:
            # pdf_text = self.harvest_pdf(pdf)
            # similar_chunks = self.get_sim_text(
            #     pdf_text,
            # )
            # if similar_chunks:
            #         content_urls.append(pdf)

            # similar_texts.append(similar_chunks)

            # aggregating everything
            for text in similar_texts:
                final_info += "\n".join(text) + "\n\n"

        return final_info, content_urls, queries

    def run_crawler(self):
        start_time = time.time()
        if not isinstance(self.claim, str) or not isinstance(self.web_sources, int):
            print(
                "Either claim is not a string or num_of_web_sources is not an integer."
            )
            return None

        external_sources, urls, queries = asyncio.run(self.get_external_knowledge())

        # for debugging purposes
        # print(
        #     "----------------------------EXTERNAL SOURCES HARVESTED----------------------------"
        # )
        # print(external_sources)
        # print(
        #     "----------------------------------------------------------------------------------"
        # )
        extraction_time = np.round(
            (time.time() - start_time),
        )

        # Invoke the gemini llm
        start_time = time.time()
        gemini_response = self.run_gemini(external_sources, self.article_id)
        if gemini_response:
            label_match = re.search(
                r"(?i)Result of the statement:\s*(.*?)\s*justification:",
                str(gemini_response),
                re.DOTALL,
            )
            label = label_match.group(1) if label_match else None
            justification_match = re.search(
                r"Justification:\s*(.*)", str(gemini_response["response"]), re.DOTALL
            )
            justification = (
                justification_match.group(1).strip() if justification_match else None
            )
            gemini_response["sources"] = urls
            gemini_response["external_sources"] = external_sources
            gemini_response["label"] = re.sub(r"\s+$", "", label)
            gemini_response["label"] = translate_label(str(gemini_response["label"]))
            gemini_response["justification"] = justification
            gemini_response["elapsed_time"] = (
                np.round((time.time() - start_time), 2) + extraction_time
            )
            gemini_response["timestamp"] = time.strftime(
                "%Y-%m-%d %H:%M:%S%z", time.localtime()
            )
            gemini_response["model"] = "Gemini"
            gemini_response["queries"] = queries
            return gemini_response

        # if the connections fails to be established or the response is empty, invoke the groq api.
        start_time = time.time()
        groq_response = self.run_groq(external_sources, self.article_id)
        if groq_response:
            label_match = re.search(
                r"(?i)Result of the statement:\s*(.*?)\s*justification:",
                str(groq_response),
                re.DOTALL,
            )
            label = label_match.group(1) if label_match else None
            justification_match = re.search(
                r"Justification:\s*(.*)", str(groq_response["response"]), re.DOTALL
            )
            justification = (
                justification_match.group(1).strip() if justification_match else None
            )
            groq_response["sources"] = urls
            groq_response["external_sources"] = external_sources
            groq_response["label"] = re.sub(r"\s+$", "", label)
            groq_response["label"] = translate_label(str(groq_response["label"]))
            groq_response["justification"] = justification
            groq_response["elapsed_time"] = (
                np.round((time.time() - start_time), 2) + extraction_time
            )
            groq_response["timestamp"] = time.strftime(
                "%Y-%m-%d %H:%M:%S%z", time.localtime()
            )
            groq_response["model"] = "GROQ"
            groq_response["queries"] = queries
            return groq_response

        # #if the connection fails again, invoke the mistral llm
        start_time = time.time()
        mistral_response = self.run_mistral(external_sources, self.article_id)
        if mistral_response:
            label_match = re.search(
                r"(?i)Result of the statement:\s*(.*?)\s*justification:",
                str(mistral_response),
                re.DOTALL,
            )
            label = label_match.group(1) if label_match else None
            justification_match = re.search(
                r"Justification:\s*(.*)", str(mistral_response["response"]), re.DOTALL
            )
            justification = (
                justification_match.group(1).strip() if justification_match else None
            )
            mistral_response["sources"] = urls
            mistral_response["external_sources"] = external_sources
            mistral_response["label"] = re.sub(r"\s+$", "", label)
            mistral_response["label"] = translate_label(
                str(mistral_response["label"]).replace("/n", "")
            )
            mistral_response["justification"] = justification
            mistral_response["elapsed_time"] = (
                np.round((time.time() - start_time), 2) + extraction_time
            )
            mistral_response["timestamp"] = time.strftime(
                "%Y-%m-%d %H:%M:%S%z", time.localtime()
            )
            mistral_response["model"] = "Mistral"
            mistral_response["queries"] = queries
            return mistral_response

        # if the connections fails to be established or the response is empty, invoke the local LLM.
        start_time = time.time()
        ollama_response = self.run_ollama(external_sources, self.article_id)
        if ollama_response:
            label_match = re.search(
                r"(?i)Result of the statement:\s*(.*?)\s*justification:",
                str(ollama_response),
                re.DOTALL,
            )
            label = label_match.group(1) if label_match else None
            justification_match = re.search(
                r"Justification:\s*(.*)", str(ollama_response["response"]), re.DOTALL
            )
            justification = (
                justification_match.group(1).strip() if justification_match else None
            )
            print(f"JUSTIFICATION: {justification}")
            ollama_response["sources"] = urls
            ollama_response["label"] = translate_label(str(label))
            ollama_response["justification"] = translate_long_text(
                justification, src_lang="en", target_lang="el"
            )
            ollama_response["elapsed_time"] = (
                np.round((time.time() - start_time), 2) + extraction_time
            )
            ollama_response["timestamp"] = time.strftime(
                "%Y-%m-%d %H:%M:%S%z", time.localtime()
            )
            ollama_response["model"] = "Ollama"
            ollama_response["queries"] = queries
            return ollama_response

        return {"error": "All llm invokation attempts failed"}
