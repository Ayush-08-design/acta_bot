import os

import httpx2
import trafilatura
from ddgs import DDGS
from langchain.tools import tool
from tavily import TavilyClient


@tool
async def extract_clean_content(url: str) -> str:
    '''Extract text from a webpage by a given url'''
    # Fetch the raw HTML using requests
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = httpx2.get(url, headers=headers, timeout=10)
    
    if response.status_code != 200:
        return f"Error: Status code {response.status_code}"

    # Extract only the main body content
    clean_text = trafilatura.extract(response.text, include_links=False)
    
    return clean_text

@tool
async def get_top_urls_ddg(query: str, n: int = 5) -> list:
    """Fetches top N URLs for a query via DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=n)]
            return [item['href'] for item in results if 'href' in item]
    except Exception as e:
        return [f"DuckDuckGo engine error: {e}"]

@tool
async def news_search(headline : str) -> list[dict]:
    '''Search news by a headline'''
    with DDGS() as ddgs:
        news_results = list(ddgs.news(headline , max_results=3))
        news = []
        for article in news_results:
            dic = {}
            dic["Title"] = article['title']
            dic["Source"] = article['source']
            dic["Date"] = article['date']
            dic["Url"] = article['url']
            news.append(dic)
        return news
    
@tool
async def local_query(query : str) -> list[dict]:
    '''For searching something related to a particular locality'''
    places_info = []
    with DDGS() as ddgs:
        places = list(ddgs.maps(query , max_results=3))
        
        for place in places:
            info = {}
            info["Name"] = place['title']
            info["Address"] = place['address']
            info['Phone'] = place.get('phone', 'N/A')
            places_info.append(info)
    return places_info

@tool
async def local_news(keywords : str , region : str = "wt-wt"):
    '''Get a local news of a particular region - us-en' for USA, 'uk-en' for UK, 'in-en' for India, 'wt-wt' for Worldwide'''
    with DDGS() as ddgs:
        localized_results = ddgs.text(
            keywords=keywords,
            region=region,     
            safesearch="moderate",
            timelimit="d"        
        )
    return localized_results

@tool
async def web_search(query : str) -> list:
    '''Tavily search tool'''
    tavily_key = os.getenv('TAVILY_API_KEY')
    tavily_client = TavilyClient(api_key=tavily_key)

    response = tavily_client.search(query , max_results=3 , search_depth='ultra-fast')
    web_result = []
    for result in response['results']:
        web_result.append({'url' : result['url'] , 'title' : result['title'] , 'content' : result['content']})
    return web_result

