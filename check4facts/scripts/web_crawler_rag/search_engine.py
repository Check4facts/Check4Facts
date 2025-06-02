import pprint
from langchain_community.utilities import SearxSearchWrapper
import os
from dotenv import load_dotenv

load_dotenv()


class SearchEngine:

    def __init__(self, num_of_results):
        self.wrapper = SearxSearchWrapper(searx_host=os.getenv("SEARXNG_HOST"))
        self.num_of_results = num_of_results

    def call_engine(self, queries):
        urls = []
        for query in queries:
            query = str(query).replace('"', "")
            print("SearxSearchWrapper STARTING")
            results = self.wrapper.results(query, num_results=self.num_of_results)
            urls_ = [res.get("link") for res in results if "link" in res]
            urls.extend(urls_)
        return urls
