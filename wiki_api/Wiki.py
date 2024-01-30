import time
import functools

import wikipediaapi as wikiapi
import requests
from bs4 import BeautifulSoup
from Levenshtein import ratio as level_ratio


class OnelineSearchEngine:
    def __init__(self, max_suggest=10, max_distance=0.85, max_retry=5):
        self._max_sugg = max_suggest
        self._min_ratio = max_distance
        self._max_retry = max_retry
        self._wiki = wikiapi.Wikipedia("MedQA_CaseStudy (xuansheng.wu@uga.edu)")

    def _get_suggest(self, query):
        sleep = 0.1
        for ntry in range(self._max_retry):
            response = requests.get("https://en.wikipedia.org/wiki/Special:Search/%s" % query)
            try:
                if response.status_code != 200:
                    raise
                soup = BeautifulSoup(response.content, "html.parser")
                paras = []
                for para in soup.find_all("div"):
                    if "mw-search-result-heading" in "".join(para.get("class", "")):
                        para = para.text
                        if level_ratio(para, query, processor=str.lower) >= self._min_ratio:
                            paras.append(para)
                    if len(paras) == self._max_sugg:
                        break
                return paras
            except:
                time.sleep(sleep)
                sleep *= 2
        return []

    @functools.lru_cache(1024)
    def _prepared_page(self, query):
        sleep = 0.1
        for ntry in range(self._max_retry):
            try:
                return self._wiki.page(query)
            except:
                time.sleep(sleep)
                sleep *= 2
        return None

    def _get_page(self, query):
        page = self._prepared_page(query)
        if page and page.exists():
            return query, page
        for query in self._get_suggest(query):
            page = self._wiki.page(query)
            if page and page.exists():
                return query, page
        return None, None

    def normalize(self, query):
        query, page = self._get_page(query)
        if query is None:
            return None
        return page._attributes["title"]

    def search(self, query):
        query, page = self._get_page(query)
        if query is None:
            return None, None
        contents = []
        def render(sections, level=0):
            for s in sections:
                if s.title in {"References", "Further reading",
                               "External links", "Explanatory notes"}:
                    continue
                contents.append("%s - %s" % (s.title, s.text))
                render(s.sections, level + 1)
        render(page.sections)
        return query, contents



if __name__ == "__main__":
    wikisearch = OnelineSearchEngine()
    for search in ["Khalistan"][:1]:
        print(wikisearch.normalize(search))
        query, contents = wikisearch.search(search)
        print("\n" * 3)
        print("Original: %s | Refined: %s" % (search, query))
        print("Sections: %s" % "\n".join(contents))
