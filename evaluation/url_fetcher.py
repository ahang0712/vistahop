import requests
import os
import logging
from typing import Optional
from bs4 import BeautifulSoup

# ==========================================
# Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SKIP_EXTENSIONS = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.mp4', '.mp3', '.exe')
PROXY_URL = (
    os.environ.get("http_proxy")
    or os.environ.get("HTTP_PROXY")
    or os.environ.get("https_proxy")
    or os.environ.get("HTTPS_PROXY")
)
PROXIES = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

class URLFetchError(Exception):
    pass

# ==========================================
# Fetch helpers
# ==========================================

def fetch_url_by_requests(url: str, timeout_sec: int = 20) -> Optional[str]:
    """
    Fallback 1: fetch page text with requests.
    """
    if any(url.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        logger.debug(f"[Requests] Skipping non-HTML resource: {url}")
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        logger.debug(f"[Requests] Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=timeout_sec, proxies=PROXIES)
        response.raise_for_status()

        response.encoding = 'utf-8'

        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type:
            logger.debug(f"[Requests] Non-HTML content type ({content_type}), skipping: {url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        raw_content = soup.get_text(separator='\n', strip=True)

        if not raw_content:
            logger.warning(f"[Requests] Empty content: {url}")
            return None
            
        logger.info(f"[Requests] Fetched content (length: {len(raw_content)}): {url}")
        return raw_content

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "Unknown"
        if status in [403, 404, 412]:
            logger.warning(f"[Requests] Target rejected request ({status}): {url}")
        else:
            logger.error(f"[Requests] HTTP error ({status}): {url} - {e}")
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"[Requests] Timeout ({timeout_sec}s): {url}")
        return None
    except Exception as e:
        logger.error(f"[Requests] Unexpected error: {e} | URL: {url}")
        return None

def fetch_by_jina(url: str, timeout_sec: int = 20) -> Optional[str]:
    """
    Fallback 2: fetch content with Jina AI Reader.
    """
    jina_api_key = os.environ.get("JINA_API_KEY", "")
    
    headers = {
        "User-Agent": "FastAPI-Fetcher/1.0",
        "Authorization": f"Bearer {jina_api_key}" if jina_api_key else ""
    }
    
    jina_url = f"https://r.jina.ai/{url}"
    
    try:
        logger.debug(f"[Jina] Fetching: {jina_url}")
        response = requests.get(jina_url, headers=headers, timeout=timeout_sec)
        
        if response.status_code == 200:
            content = response.text.strip()
            if content:
                logger.info(f"[Jina] Fetched content (length: {len(content)}): {url}")
                return content
            else:
                logger.warning(f"[Jina] Empty content: {url}")
                return None
        else:
            logger.warning(f"[Jina] HTTP {response.status_code}: {response.text[:100]}... | URL: {url}")
            return None
            
    except Exception as e:
        logger.error(f"[Jina] Request error: {e} | URL: {url}")
        return None

def fetch_url_content_remote_sync(url: str, server_ip: str = "localhost", port: int = 7000, timeout_sec: int = 20) -> Optional[str]:
    """
    Optional remote fetch service.
    """
    api_url = f"http://{server_ip}:{port}/fetch"
    payload = {
        "url": url,
        "timeout": timeout_sec
    }

    try:
        response = requests.post(api_url, json=payload, timeout=timeout_sec + 5.0)
        
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", "Unknown error")
            except:
                detail = response.text[:100]
            raise URLFetchError(f"Server error: {detail}")
        
        result = response.json()
        content = result.get("content")
        
        if content:
            return content
        return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"Remote fetch service ({server_ip}:{port}) connection failed: {e}")
        return None
    except Exception as e:
        logger.warning(f"Remote fetch service error: {e}")
        return None

# ==========================================
# Public API
# ==========================================
def fetch_url_content(url: str, timeout_sec: int = 20) -> Optional[str]:
    """
    Fetch strategy:
    1. Optional remote service when URL_FETCHER_REMOTE_HOST is set.
    2. Local requests fetch.
    3. Jina AI Reader.
    """
    logger.info(f"Start fetching URL: {url}")

    remote_host = os.environ.get("URL_FETCHER_REMOTE_HOST")
    remote_port = int(os.environ.get("URL_FETCHER_REMOTE_PORT", "7000"))
    if remote_host:
        content = fetch_url_content_remote_sync(url, server_ip=remote_host, port=remote_port, timeout_sec=timeout_sec)
        if content:
            logger.info("[Level 1] Remote service fetch succeeded")
            return content
        logger.warning("[Level 1] Remote service failed; trying Level 2 (requests)")

    content = fetch_url_by_requests(url, timeout_sec=timeout_sec)
    if content:
        logger.info("[Level 2] Requests fetch succeeded")
        return content

    logger.warning("[Level 2] Requests failed; trying Level 3 (Jina)")

    content = fetch_by_jina(url, timeout_sec=timeout_sec)
    if content:
        logger.info("[Level 3] Jina fetch succeeded")
        return content

    logger.error("All fetch methods failed")
    return None

if __name__ == "__main__":
    test_url = "https://www.baidu.com"
    
    print(f"\n--- Testing URL fetch fallback logic (URL: {test_url}) ---\n")
    
    result = fetch_url_content(test_url, timeout_sec=10)
    
    if result:
        print("\n" + "="*50)
        print("Fetch succeeded. Content preview (first 300 chars):")
        print("="*50)
        print(result[:300])
        print("...")
        print("="*50)
    else:
        print("\nFetch failed.")
