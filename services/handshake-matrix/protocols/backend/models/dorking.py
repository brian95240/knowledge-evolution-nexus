# backend/models/dorking.py
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime

class DorkingQuery(BaseModel):
    id: str
    query_template: str
    target_domain: str
    last_run: Optional[datetime]
    
class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    snippet: str
    discovered_at: datetime
    
class DorkingResults(BaseModel):
    query_id: str
    results: List[SearchResult]
    run_timestamp: datetime

def parse_search_results(html_content: str) -> List[SearchResult]:
    """Basic parser for search results (example structure)"""
    from bs4 import BeautifulSoup
    
    results = []
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Assuming a basic structure like:
    # <div class="result">
    #   <h3><a href="url">title</a></h3>
    #   <div class="snippet">text</div>
    # </div>
    
    for result in soup.find_all('div', class_='result'):
        title_elem = result.find('h3')
        url_elem = title_elem.find('a') if title_elem else None
        snippet_elem = result.find('div', class_='snippet')
        
        if all([title_elem, url_elem, snippet_elem]):
            results.append(
                SearchResult(
                    title=title_elem.text.strip(),
                    url=url_elem['href'],
                    snippet=snippet_elem.text.strip(),
                    discovered_at=datetime.utcnow()
                )
            )
    
    return results