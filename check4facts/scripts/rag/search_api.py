from duckduckgo_search import DDGS
from urllib.parse import urlparse
import re
sites_source = ["ellinikahoaxes.gr","factcheckgreek.afp.com","check4facts.gr",
                "factcheckcyprus.org",'www.youtube.com','www.linkedin.com', "m.facebook.com"]
doc_extensions = ["doc", "docx", 'php', 'pdf', 'txt', 'theFile', 'file', 'xls']
pattern = r'[./=]([a-zA-Z0-9]+)$'

def google_search(query, web_sources):
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

