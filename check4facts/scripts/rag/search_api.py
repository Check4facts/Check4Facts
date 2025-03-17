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
                print(file_extension)
            else:
                file_extension = None
            if file_extension not in doc_extensions and url_domain not in sites_source and not '/document/' in dict['href']:
                urls.append(dict['href'])
            #print(url_domain)

        return urls
    except Exception as e:
        print(e)
        print('Initializing backup search....')
        return google_search_backup(query, web_sources)


#extra search api in case the first one fails
def google_search_backup(query, web_sources):
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": os.getenv("GOOGLE_SEARCH_KEY"),
            "cx": os.getenv("GOOGLE_CX_KEY"),
            "num": web_sources
        }

        response = requests.get(url, params=params)
        results = response.json()

        if "items" not in results:
            return []

        urls = []
        for item in results["items"]:
            link = item["link"]
            url_domain = urlparse(link).netloc

            
            match = re.search(pattern, link[-6:])
            file_extension = match.group(1) if match else None

            
            if (
                file_extension not in doc_extensions and
                url_domain not in sites_source and
                "/document/" not in link
            ):
                urls.append(link)

        return urls

    except Exception as e:
        print(f"Error: {e}")
        print('Could not find any search results')
        return []
