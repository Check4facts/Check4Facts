from duckduckgo_search import DDGS
from urllib.parse import urlparse
from googleapiclient.discovery import build
import re
import os
import psycopg2
import pandas as pd


# sites_source = [
#     "ellinikahoaxes.gr",
#     "factcheckgreek.afp.com",
#     "check4facts.gr",
#     "factcheckcyprus.org",
#     "www.youtube.com",
#     "www.linkedin.com",
#     "m.facebook.com",
# ]
# doc_extensions = ["doc", "docx", "php", "pdf", "txt", "theFile", "file", "xls"]
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
    black_urls = blacklist_urls()
    filtered_urls = []
    for url in url_list:
        url_domain = urlparse(url).netloc
        match = re.search(pattern, url[-6:])
        if match:
            file_extension = match.group(1)
        else:
            file_extension = None
        if (
            file_extension not in doc_extensions
            and url_domain not in black_urls
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
            return google_search_backup(query, web_sources)
        else:
            return urls

    except Exception as e:
        print(e)
        print("Initializing backup search....")
        return google_search_backup(query, web_sources)
