import requests
import json
from typing import Dict, Any, Optional
import os

import aiohttp
import asyncio
import logging

logger = logging.getLogger(__name__)

sources_list = ["serper-search", "google-search", "baidu-search-v2", "baidu-search", "bocha-search", "quark-search"]

sources_now = os.environ.get("MT_SEARCH_SOURCE", "serper-search")
DEFAULT_BASE_URL = os.environ.get("MT_SEARCH_BASE_URL", "")
DEFAULT_AUTHORIZATION = os.environ.get("MT_SEARCH_AUTHORIZATION", "")
DEFAULT_USER_NAME = os.environ.get("MT_SEARCH_USER_NAME", "ai-search")


def text_search(
    query: str,
    top_k: int = 10,
    page: int = 1,
    ttl: int = 0,
    timeout: int = 2000,
    base_url: Optional[str] = None,
    authorization: Optional[str] = None,
    user_name: Optional[str] = None,
    task_type: str = "eval",
    task_desc: str = "evaluation"
) -> Dict[str, Any]:
    """
    Run a synchronous text search through an OpenAI-compatible internal search service.
    
    Args:
        query: Search query.
        top_k: Number of results.
        page: Search result page.
        ttl: Cache TTL in seconds.
        timeout: Search timeout in milliseconds.
        base_url: Search API URL, defaults to MT_SEARCH_BASE_URL.
        authorization: Search token, defaults to MT_SEARCH_AUTHORIZATION.
        user_name: Traffic source label.
        task_type: Usage scenario.
        task_desc: Task description.
        
    Returns:
        Search response dictionary.
    """
    base_url = base_url or DEFAULT_BASE_URL
    authorization = authorization or DEFAULT_AUTHORIZATION
    user_name = user_name or DEFAULT_USER_NAME
    if not base_url or not authorization:
        raise ValueError("MT_SEARCH_BASE_URL and MT_SEARCH_AUTHORIZATION are required for text_search().")

    headers = {
        'Authorization': authorization,
        'Content-Type': 'application/json'
    }
    


    payload = {
        "query": query,
        "topK": top_k,
        "ttl": ttl,
        "timeout": timeout,
        "sources": [sources_now],
        "searchAccessAuth": {
            "userName": user_name,
            "taskType": task_type,
            "taskDesc": task_desc
        }
    }
    
    if page > 1:
        payload["serperSearchParam"] = {"page": page}

    try:
        response = requests.post(
            base_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        response = response.json()

        return response
    except Exception as e:
        return {
            "status": -1,
            "message": f"Request failed: {str(e)}",
            "data": None
        }



async def mt_search_async(
    query: str,
    top_k: int = 20,
    page: int = 1,
    ttl: int = 0,
    timeout: int = 20000,
    base_url: Optional[str] = None,
    authorization: Optional[str] = None,
    user_name: Optional[str] = None,
    task_type: str = "eval",
    task_desc: str = "evaluation"
) -> Dict[str, Any]:
    """
    Run an asynchronous text search.
    
    Args:
        query: Search query.
        top_k: Number of results.
        page: Search result page.
        ttl: Cache TTL in seconds.
        timeout: Search timeout in milliseconds.
        base_url: Search API URL, defaults to MT_SEARCH_BASE_URL.
        authorization: Search token, defaults to MT_SEARCH_AUTHORIZATION.
        user_name: Traffic source label.
        task_type: Usage scenario.
        task_desc: Task description.
        
    Returns:
        Search results.
    """
    base_url = base_url or DEFAULT_BASE_URL
    authorization = authorization or DEFAULT_AUTHORIZATION
    user_name = user_name or DEFAULT_USER_NAME
    if not base_url or not authorization:
        raise ValueError("MT_SEARCH_BASE_URL and MT_SEARCH_AUTHORIZATION are required for mt_search_async().")

    headers = {
        'Authorization': authorization,
        'Content-Type': 'application/json'
    }
    


    payload = {
        "query": query,
        "topK": top_k,
        "ttl": ttl,
        "timeout": timeout,
        "sources": [sources_now],
        "searchAccessAuth": {
            "userName": user_name,
            "taskType": task_type,
            "taskDesc": task_desc
        }
    }
    
    if page > 1:
        payload["serperSearchParam"] = {"page": page}
    
    # logger.info(f"[MT_TEXT_SEARCH] Starting request to {base_url} for query: '{query}'")
    
    start_time = asyncio.get_event_loop().time()
    results=[]
    
    try:
        client_timeout = aiohttp.ClientTimeout(total=50.0, connect=30.0)
        
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                base_url,
                json=payload,
                headers=headers
            ) as response:
                # logger.info(f"[MT_TEXT_SEARCH] Received response status: {response.status} {response.json()}")

                if response.status == 200:
                    data = await response.json()
                    # print(f"[DEBUG ] {data=}")
                    if data.get('status') == 0:
                        data_field = data.get('data', {})
                        raw_context = data_field.get('results', [])

  

                        for context in raw_context:
                            # logger.info(f"[MT_TEXT_SEARCH] returns keys: {context.keys()} {context=}")

                            results.append({
                                "title": context.get("title", ""),
                                "url": context.get("url", ""),
                                # "url":"",
                                "snippet": context.get("snippet", ""),
                                # "source": context.get("source", ""),
                                # "publish_time": context.get("publish_time", ""),
                                # "website": context.get("website", "")
                            })
                    else:
                        api_error = data.get('message', 'Unknown API error')
                        logger.error(f"[MEITUAN_API] API returned failure: {api_error}")
                        raise Exception(f"Meituan API Error: {api_error}")
                else:
                    error_text = await response.text()
                    logger.error(f"[MEITUAN_API] HTTP Error {response.status}: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
    
    except aiohttp.ClientConnectorError as e:
        logger.error(f"[MEITUAN_API] Connection failed: {str(e)}. Is the server  reachable?")
        raise Exception(f"Connection failed to ") from e
    except asyncio.TimeoutError as e:
        logger.error(f"[MEITUAN_API] Request timed out after 30s. {str(e)}")
        raise Exception("Request timed out") from e
    except Exception as e:
        logger.error(f"[MEITUAN_API] Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise Exception("Request Unexpected out") from e

    finally:
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"[MEITUAN_API] Request finished in {elapsed:.2f}s. Results: {len(results)}")

    return results

if __name__ == "__main__":
    result = text_search(query="VistaHop benchmark")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # result = text_search(query="VistaHop benchmark", top_k=5)
    # print(json.dumps(result, ensure_ascii=False, indent=2))

    async def test_async_search():
        print("\n=== Testing async search ===")
        result = await mt_search_async(query="VistaHop benchmark", top_k=5)
        print(f"Search result count: {len(result)}")
        for i, item in enumerate(result, 1):
            print(f"\n{i}. {item['title']}")
            print(f"   URL: {item['url']}")
            print(f"   Snippet: {item['snippet'][:100]}...")
            print(f"{result[i]=}")
    
    asyncio.run(test_async_search())
    

"""
1. Inspect cache write logs:
grep -i "cache" logs/web_search_server_*.log | tail -20

2. Inspect cache database:
sqlite3 ./search_cache/search_cache.db "SELECT COUNT(*), MAX(created_at) FROM cache_entries;"


"""
