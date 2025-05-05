from research_agent.state import Section
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

async def fetch_favicon_and_title(url: str) -> dict:
    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    api_endpoint = "https://chat.makaicare.com/api/url-meta"
    params = {"url": url}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(api_endpoint, params=params)
            resp.raise_for_status()
            # We expect your API to return the dict {url, title, favicon}
            data = resp.json()
    except httpx.HTTPError as e:
        print(f"[fetch][ERROR] calling metadata API: {e}")
        # Fallback to empty values or re-raise, as you prefer
        return {"url": url, "title": "", "favicon": ""}

    print(f"[fetch][INFO] metadata API response: {data}")
    return data