import asyncio
import sys
import time
import typing as tp
from functools import lru_cache

from .proxy import LazyProxy
from .utils import get_logger

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup
    import urllib.parse

except ImportError:
    print(
        "Please install required libraries: `pip install selenium beautifulsoup4 pydantic webdriver-manager lxml`"
    )
    sys.exit(1)

from pydantic import BaseModel, Field, field_validator

logger = get_logger(__name__)


class WebSearchResult(BaseModel):
    title: str
    content: str = ""
    link: str
    rank: int

    @field_validator("title", "content", mode="before")
    @classmethod
    def clean_text(cls, v:str):
        return v.strip() if v else ""


class WebSearchTool(LazyProxy[webdriver.Chrome]):
    """
    Web Search Tool for retrieving search results from Google
    Supports multi-page search with configurable parameters
    """

    # Search configuration
    query: str = Field(..., description="Search query string")
    max_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of search results"
    )
    pages: int = Field(
        default=1, ge=1, le=5, description="Number of search pages to query"
    )
    country: str = Field(default="US", description="Two-letter country code")
    language: str = Field(default="en", description="Two-letter language code")
    safe_search: bool = Field(default=True, description="Enable safe search")
    timeout: int = Field(default=10, description="Timeout for page loading in seconds")

    @lru_cache(maxsize=1)
    def __load__(self) -> webdriver.Chrome:
        """Set up and configure Chrome WebDriver with optimized settings"""
        options = Options()
        arguments = [
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--blink-settings=imagesEnabled=false",
            "--disable-extensions",
            "--disable-infobars",
            "--window-size=1280,720",
            "--disable-browser-side-navigation",
            "--disable-features=NetworkService",
            "--disable-features=VizDisplayCompositor",
            "--ignore-certificate-errors",
            "--disk-cache-size=33554432",  # 32MB disk cache
            "--js-flags=--max_old_space_size=512",  # Limit JS memory
        ]

        for arg in arguments:
            options.add_argument(arg)

        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("prefs", {
            "profile.default_content_setting_values.images": 2,  # Disable images
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,  # Disable notifications
            "profile.managed_default_content_settings.javascript": 1,  # Enable JavaScript
        })
        
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(self.timeout)
        return driver

    def _build_search_url(self, page: int = 0) -> str:
        """Build a Google search URL with various parameters"""
        base_url = "https://www.google.com/search"
        params = {
            "q": self.query,
            "hl": self.language,
            "gl": self.country,
            "start": str(page * 10),
            "nfpr": "1",  # No spelling corrections
            "filter": "0",  # Don't omit similar results
        }

        if self.safe_search:
            params["safe"] = "active"

        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def _extract_search_results(
        self, page_source: str, page: int
    ) -> list[WebSearchResult]:
        """Extract search results from page source with improved parsing"""
        soup = BeautifulSoup(page_source, "lxml")
        results: list[WebSearchResult] = []
        
        search_results = soup.select("div.g")
        
        for i, result_div in enumerate(search_results):
            if len(results) >= self.max_results:
                break

            try:
                # Find link and title
                link_elem = result_div.select_one("a[href]")
                if not link_elem:
                    continue
                    
                # Get and clean link
                link = link_elem["href"]
                if link.startswith("/url?q="):
                    link = link.split("/url?q=")[1].split("&sa=")[0]
                    
                # Skip non-http links
                if not (link.startswith("http://") or link.startswith("https://")):
                    continue
                
                # Extract title - more robust approach
                title_elem = result_div.select_one("h3")
                title = title_elem.get_text(strip=True) if title_elem else ""
                
                # Extract snippet - more robust approach
                content_elem = result_div.select_one("div.VwiC3b") or result_div.select_one("span.aCOpRe")
                content = content_elem.get_text(strip=True) if content_elem else ""
                
                # Create result only if we have at least title and link
                if title and link:
                    results.append(
                        WebSearchResult(
                            title=title,
                            link=link,
                            content=content,
                            rank=(page * 10) + i + 1,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error processing result {i}: {str(e)}")
                continue

        return results

    async def run(self) -> tp.AsyncGenerator[str, None]:
        """Execute web search with improved concurrency and error handling"""
        try:
            results = await asyncio.to_thread(self._sync_run)
            
            for result in results:
                yield result.model_dump_json()
                
        except asyncio.CancelledError:
            logger.info("Search task was cancelled")
            raise
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            yield f'{{"error": "Search failed: {str(e)}"}}'

    def _sync_run(self) -> list[WebSearchResult]:
        """Synchronous version of the search with better resource management"""
        driver = None
        all_results: list[WebSearchResult] = []
        
        try:
            driver = self.__load__()
            
            for page in range(self.pages):
                try:
                    # Construct and navigate to search URL
                    search_url = self._build_search_url(page)
                    driver.get(search_url)

                    # Wait for search results to load with more robust selector
                    WebDriverWait(driver, self.timeout).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.g, div.yuRUbf"))
                    )

                    # Extract results for this page
                    page_results = self._extract_search_results(driver.page_source, page)
                    all_results.extend(page_results)

                    # Stop if we've reached max results
                    if len(all_results) >= self.max_results:
                        break
                        
                    # Small delay between pages to avoid rate limiting
                    if page < self.pages - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.warning(f"Error on page {page}: {str(e)}")
                    continue  # Try next page rather than aborting

            # Truncate results to max_results
            return all_results[:self.max_results]

        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    logger.warning(f"Error closing WebDriver: {str(e)}")