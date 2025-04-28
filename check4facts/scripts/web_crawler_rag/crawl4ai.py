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
from check4facts.scripts.rag.search_api import google_search

nltk.download("punkt")


class crawl4ai:
    def __init__(self, claim, web_sources, article_id):
        self.claim = claim
        self.web_sources = web_sources
        self.article_id = article_id
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    def get_urls(self):
        self.urls = google_search(self.claim, self.web_sources)
        return self.urls

    def chunk_text(self, text, chunk_size=500, overlap_size=50):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence[:overlap_size]
                current_chunk += sentence[overlap_size:]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_sim_text(
        self,
        text,
        threshold=0.3,
        chunk_size=500,
    ):
        if not text:
            return []
        claim_embedding = self.model.encode(self.claim, convert_to_tensor=True)
        filtered_results = []
        chunks = self.chunk_text(text, chunk_size)
        if not chunks:
            return []
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        chunk_similarities = util.cos_sim(claim_embedding, chunk_embeddings)
        for chunk, similarity in zip(chunks, chunk_similarities[0]):
            if similarity >= threshold:
                filtered_results.append(chunk)
        return filtered_results

    def single_text_embedding(self, text):
        embedding = self.model.encode(text, convert_to_tensor=True)
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
        gemini_instance = gemini_llm(query=self.claim, external_knowledge=info)
        answer = gemini_instance.google_llm(article_id)
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

    async def get_external_knowledge(self):
        urls = self.get_urls()
        run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=True)

        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun_many(urls, config=run_conf):
                if result.success:
                    print(
                        f"[OK] {result.url}, length: {len(result.markdown.raw_markdown)}"
                    )
                else:
                    print(f"[ERROR] {result.url} => {result.error_message}")

            run_conf = run_conf.clone(stream=False)
            results = await crawler.arun_many(urls, config=run_conf)
            similar_texts = []
            for res in results:
                if res.success:

                    print(f"[OK] {res.url}, length: {len(res.markdown.raw_markdown)}")
                    similar_chunks = self.get_sim_text(
                        self.convert_markdown_to_text(res.markdown.raw_markdown),
                    )
                    similar_texts.append(similar_chunks)
                else:
                    print(f"[ERROR] {res.url} => {res.error_message}")

            print("Similar texts:")
            final_info = ""
            for text in similar_texts:
                final_info += "\n".join(text) + "\n\n"
        return final_info, urls

    def run_crawler(self):
        start_time = time.time()
        if not isinstance(self.claim, str) or not isinstance(self.web_sources, int):
            print(
                "Either claim is not a string or num_of_web_sources is not an integer."
            )
            return None

        external_sources, urls = asyncio.run(self.get_external_knowledge())

        # for debugging purposes
        print(
            "----------------------------EXTERNAL SOURCES HARVESTED----------------------------"
        )
        print(external_sources)
        print(
            "----------------------------------------------------------------------------------"
        )
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
            return ollama_response

        return {"error": "All llm invokation attempts failed"}
