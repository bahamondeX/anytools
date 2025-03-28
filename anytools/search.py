import sys
import typing as tp
import asyncio
from .utils import get_logger

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager

except ImportError:
    print(
        "Please install required libraries: `pip install selenium beautifulsoup4 pydantic webdriver-manager lxml`"
    )
    sys.exit(1)

from .tool import Tool
from pydantic import BaseModel, Field, validator  # type: ignore


logger = get_logger(__name__)


class WebSearchResult(BaseModel):
    title: str
    content: tp.Optional[str] = None
    link: str
    rank: tp.Optional[int] = None

    @validator("title", "content", pre=True, always=True)  # type: ignore
    def clean_text(cls, v):  # type: ignore
        return v.strip() if v else ""  # type: ignore


class WebSearchTool(Tool[webdriver.Chrome]):
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

    def __load__(self) -> webdriver.Chrome:
        """
        Set up and configure Chrome WebDriver

        :return: Configured WebDriver instance
        """
        options = Options()
        arguments = [
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--blink-settings=imagesEnabled=false",
        ]

        for arg in arguments:
            options.add_argument(arg)  # type: ignore

        options.add_experimental_option("excludeSwitches", ["enable-automation"])  # type: ignore
        options.add_experimental_option("useAutomationExtension", False)  # type: ignore
        options.add_argument(  # type: ignore
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def _build_search_url(self, page: int = 0) -> str:
        """
        Build a Google search URL with various parameters

        :param page: Page number to search (0-indexed)
        :return: Constructed search URL
        """
        import urllib.parse

        base_url = "https://www.google.com/search"

        params = {
            "q": self.query,
            "hl": self.language,
            "gl": self.country,
            "start": str(page * 10),
        }

        if self.safe_search:
            params["safe"] = "on"

        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def _extract_search_results(
        self, page_source: str, page: int
    ) -> list[WebSearchResult]:
        """
        Extract search results from page source

        :param page_source: HTML page source
        :param page: Current page number
        :return: List of WebSearchResult objects
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(page_source, "lxml")

        results: list[WebSearchResult] = []
        links = soup.select("div.yuRUbf")
        contents = soup.select("div.VwiC3b")

        for i, (link_div, content_div) in enumerate(zip(links, contents)):
            if len(results) >= self.max_results:
                break

            try:
                link_elem = link_div.find("a", href=True)
                if not link_elem:
                    continue

                # Clean URL (simplified for this example)
                link: str = link_elem["href"]  # type: ignore
                if link.startswith("/url?q="):  # type: ignore
                    link = link.split("/url?q=")[1].split("&sa=")[0]  # type: ignore

                title = link_div.get_text(strip=True)
                content = content_div.get_text(strip=True) if content_div else ""

                results.append(
                    WebSearchResult(
                        title=title,
                        link=link,
                        content=content,
                        rank=(page * 10) + i + 1,
                    )
                )
            except Exception as e:
                logger.warning(f"Error processing result {i}: {e}")

        return results

    async def run(self) -> tp.AsyncGenerator[str, None]:
        """
        Execute web search and yield results as a string

        :return: Async generator of search result strings
        """

        def _sync_run():
            """Synchronous version of the search for use with ThreadPoolExecutor"""
            driver = None
            try:
                driver = self.__load__()
                all_results: list[WebSearchResult] = []

                for page in range(self.pages):
                    # Construct and navigate to search URL
                    search_url = self._build_search_url(page)
                    driver.get(search_url)

                    # Wait for search results to load
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.yuRUbf"))
                    )

                    # Extract results for this page
                    page_results = self._extract_search_results(
                        driver.page_source, page
                    )
                    all_results.extend(page_results)

                    # Stop if we've reached max results
                    if len(all_results) >= self.max_results:
                        break

                # Truncate results to max_results
                all_results = all_results[: self.max_results]

                return all_results

            except Exception as e:
                logger.error(f"Search failed: {e}")
                raise Exception("Search failed")

            finally:
                if driver:
                    driver.quit()

        results = await asyncio.to_thread(_sync_run)

        for result in results:
            yield result.model_dump_json() + "\n\n"
