from duckduckgo_search import DDGS
from urllib.parse import urlparse
from googleapiclient.discovery import build
import re
import os
import psycopg2
import pandas as pd
import json
import time
import random

# doc_extensions = ["doc", "docx", "php", "pdf", "txt", "theFile", "file", "xls"]

# with open("data/whitelist.json", "r") as f:
#     whitelist = json.load(f)


doc_extensions = []
pattern = r"[./=]([a-zA-Z0-9]+)$"


def blacklist_urls():
    from check4facts.api import dbh

    try:
        url_list = dbh.fetch_blacklist()
        return url_list
    except Exception as e:
        print(f"Error fetching blacklist: {e}")
        return []


def filter_urls(url_list):
    black_urls = []
    # black_urls = blacklist_urls()
    filtered_urls = []
    for url in url_list:
        url_domain = str(urlparse(url).netloc).replace("www.", "")
        match = re.search(pattern, url[-6:])
        if match:
            file_extension = match.group(1)
        else:
            file_extension = None
        if (
            file_extension not in doc_extensions
            and url_domain not in black_urls
            # and url_domain in whitelist
            and not "/document/" in url
        ):
            filtered_urls.append(url)

    return filtered_urls


def google_search_backup(query, web_sources):
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))

    res = (
        service.cse()
        .list(q=query, cx=os.getenv("GOOGLE_CX_KEY"), num=web_sources)
        .execute()
    )
    urls = [item["link"] for item in res.get("items", [])]
    return filter_urls(urls)


def google_search(query, web_sources):
    try:
        urls = []
        search_query = query
        results = DDGS().text(
            keywords=search_query,
            region="gr-el",
            safesearch="off",
            max_results=web_sources,
        )
        for dict in results:
            urls.append(dict["href"])
        urls = filter_urls(urls)
        if urls == []:
            print("Initializing backup search....")
            return filter_urls(google_search_backup(query, web_sources))
        else:
            return urls

    except Exception as e:
        print(e)
        print("Initializing backup search....")
        return filter_urls(google_search_backup(query, web_sources))


def search_queries(query_list):
    urls = []
    service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))
    for q in query_list:
        q = str(q).replace('"', "")
        print(f"Searching for query: {q}")
        try:
            time.sleep(random.uniform(6, 10))  # Randomized sleep
            results = DDGS().text(keywords=q, safesearch="off", max_results=2)
            if results:
                urls.extend(res["href"] for res in results)
            else:
                print("No DDG results. Skipping Google fallback to avoid quota.")
        except Exception as e:
            print(f"DDG error: {e}. Trying Google.")
            try:
                res = (
                    service.cse()
                    .list(q=q, cx=os.getenv("GOOGLE_CX_KEY"), num=2)
                    .execute()
                )
                urls.extend(item["link"] for item in res.get("items", []))
            except Exception as g_err:
                print(f"Google Search failed: {g_err}")
    return urls
