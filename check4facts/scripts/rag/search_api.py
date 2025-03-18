from duckduckgo_search import DDGS
from urllib.parse import urlparse
import re
import os
import requests
sites_source = ["ellinikahoaxes.gr","factcheckgreek.afp.com","check4facts.gr",
                "factcheckcyprus.org",'www.youtube.com','www.linkedin.com', "m.facebook.com"]
doc_extensions = ["doc", "docx", 'php', 'pdf', 'txt', 'theFile', 'file', 'xls']
pattern = r'[./=]([a-zA-Z0-9]+)$'

def google_search(query, web_sources):
    try:
        urls = []
        search_query = query
        results = DDGS().text(
            keywords=search_query,
            region='gr-el',
            safesearch='off',
            max_results=web_sources,

        )
        for dict in results:
            url_domain = urlparse(dict['href']).netloc
            match = re.search(pattern, dict['href'][-6:])
            if match:
                file_extension = match.group(1)
            else:
                file_extension = None
            if file_extension not in doc_extensions and url_domain not in sites_source and not '/document/' in dict['href']:
                urls.append(dict['href'])
        if urls:
            return urls
        else: 
            print('Initializing backup search....')
            return google_search_backup(query, web_sources)
    except Exception as e:
        print(e)
        print('Initializing backup search....')
        return google_search_backup(query, web_sources)


#extra search api in case the first one fails
def google_search_backup(query, web_sources):
    """Fetches Google search results and returns only the links."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": os.getenv("GOOGLE_SEARCH_KEY"),
        "cx": os.getenv("GOOGLE_CX_KEY"),
        "num": web_sources
    }

    urls = []

    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("items", [])
        url_list =  [item["link"] for item in results if "link" in item]

        for url in url_list:
            url_domain = urlparse(url).netloc
            match = re.search(pattern, url[-6:])
            if match:
                file_extension = match.group(1)
            else:
                file_extension = None
                print(file_extension, url_domain, url)
            if file_extension not in doc_extensions and url_domain not in sites_source and not '/document/' in url:
                urls.append(url)
        return urls

    else:
        print("Error:", response.json())
        return []
