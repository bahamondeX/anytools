import asyncio
import sys
import typing as tp
from abc import ABC, abstractmethod
from functools import lru_cache

import typing_extensions as tpe
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextDelta, ToolParam, ToolUseBlock
from pydantic import Field

from .proxy import LazyProxy
from .tool import Tool
from .utils import get_logger, get_random_int

AnthropicModels: tpe.TypeAlias = tp.Literal[
    "claude-3-7-sonnet-20250219",  # $75 • $150
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]

logger = get_logger()


class AnthropicTool(Tool, LazyProxy[AsyncAnthropic], ABC):
    """
    An abstract base class representing a tool that can be used in chat completions.

    This class combines functionality from Pydantic's BaseModel, LazyProxy, and ABC to create
    a flexible and extensible tool structure for use with groq's chat completion API.
    """

    @lru_cache(maxsize=1)
    def __load__(self):
        return AsyncAnthropic()

    @classmethod
    @lru_cache
    def tool_param(cls) -> ToolParam:
        logger.info("Retrieving %s `tool_param`.", cls.__name__)
        return ToolParam(
            input_schema=cls.model_json_schema(),
            name=cls.__name__,
            description=cls.__doc__ or "",
            cache_control={"type": "ephemeral"},
        )

    @abstractmethod
    def run(self) -> tp.AsyncGenerator[str, None]: ...


try:
    import urllib.parse

    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager

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
    def clean_text(cls, v: str):
        return v.strip() if v else ""


class WebSearchTool(AnthropicTool):
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
    def __call__(self) -> webdriver.Chrome:
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
            options.add_argument(arg)  # type: ignore

        options.add_experimental_option(  # type: ignore
            "excludeSwitches", ["enable-automation", "enable-logging"]
        )
        options.add_experimental_option("useAutomationExtension", False)  # type: ignore
        options.add_experimental_option(  # type: ignore
            "prefs",
            {
                "profile.default_content_setting_values.images": 2,  # Disable images
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2,  # Disable notifications
                "profile.managed_default_content_settings.javascript": 1,  # Enable JavaScript
            },
        )

        options.add_argument(  # type: ignore
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

    def _extract_search_results_generator(
        self, page_source: str, page: int
    ) -> tp.Generator[WebSearchResult, None, None]:
        """Extract search results from page source and yield them one by one"""
        soup = BeautifulSoup(page_source, "lxml")
        result_count = 0
        found_links = set[str]()  # Track found links to avoid duplicates

        # Look for search results using multiple possible selectors
        search_results = soup.select("div.MjjYud")

        for i, result_div in enumerate(search_results):
            if result_count >= self.max_results:
                return

            try:
                # Find link container (either g, yuRUbf, or direct a)
                link_container = result_div.select_one(
                    "div.g"
                ) or result_div.select_one("div.yuRUbf")

                # Find link element
                link_elem = None
                if link_container:
                    link_elem = link_container.select_one("a[href]")
                else:
                    link_elem = result_div.select_one("a[href]")

                if not link_elem:
                    continue

                # Get and clean link
                link = link_elem.get("href", "")
                if link.startswith("/url?q="):
                    link = link.split("/url?q=")[1].split("&sa=")[0]

                # Skip non-http links
                if not (link.startswith("http://") or link.startswith("https://")):
                    continue

                # Skip duplicates
                if link in found_links:
                    continue
                found_links.add(link)

                # Extract title - more robust approach
                title_elem = result_div.select_one("h3")
                title = title_elem.get_text(strip=True) if title_elem else ""

                # Extract snippet with more comprehensive selector approach
                # Try multiple paths to find content
                content = ""

                # First approach: Try VwiC3b class which is common for snippet content
                content_elem = result_div.select_one("div.VwiC3b")
                if content_elem:
                    content = content_elem.get_text(strip=True)

                # Second approach: Try content in kb0PBd section (which often contains snippets)
                if not content:
                    kb_section = result_div.select_one("div.kb0PBd.A9Y9g")
                    if kb_section:
                        # Look for nested content
                        desc_div = kb_section.select_one(
                            "div.VwiC3b"
                        ) or kb_section.select_one("div[class*='p4wth']")
                        if desc_div:
                            content = desc_div.get_text(strip=True)

                # Third approach: Look for aCOpRe spans which may contain descriptions
                if not content:
                    content_spans = result_div.select("span.aCOpRe")
                    if content_spans:
                        content = " ".join(
                            [span.get_text(strip=True) for span in content_spans]
                        )

                # Last resort: Try to get any text content from the result div
                if not content:
                    # Exclude title from the text extraction
                    title_text = title_elem.get_text(strip=True) if title_elem else ""
                    all_text = result_div.get_text(strip=True)
                    if all_text and title_text:
                        # Remove title from all text to get remaining content
                        content = all_text.replace(title_text, "", 1).strip()

                # Create result only if we have at least title and link
                if title and link:
                    result = WebSearchResult(
                        title=title,
                        link=link,
                        content=content[:500],  # Limit content length
                        rank=(page * 10) + i + 1,
                    )
                    result_count += 1
                    # Yield result immediately
                    yield result

            except Exception as e:
                logger.warning(f"Error processing result {i}: {str(e)}")
                continue

        # If we didn't find enough results, try an alternative selector
        if result_count < min(self.max_results, 5) and soup is not None:
            logger.info("Using alternative selectors to find more results")
            try:
                # Try a broader selector
                alt_results = soup.select("div.g")
                for i, result_div in enumerate(alt_results):
                    if result_count >= self.max_results:
                        return

                    # Similar logic as above but simplified
                    link_elem = result_div.select_one("a[href]")
                    if not link_elem:
                        continue

                    link = link_elem.get("href", "")
                    if link.startswith("/url?q="):
                        link = link.split("/url?q=")[1].split("&sa=")[0]

                    if not (link.startswith("http://") or link.startswith("https://")):
                        continue

                    # Skip duplicates
                    if link in found_links:
                        continue
                    found_links.add(link)

                    title_elem = result_div.select_one("h3")
                    title = title_elem.get_text(strip=True) if title_elem else ""

                    content = ""
                    content_elem = result_div.select_one(
                        "[class*='VwiC3b']"
                    ) or result_div.select_one("[class*='p4wth']")
                    if content_elem:
                        content = content_elem.get_text(strip=True)

                    if title and link:
                        result = WebSearchResult(
                            title=title,
                            link=link,
                            content=content[:500],
                            rank=(page * 10) + result_count + 1,
                        )
                        result_count += 1
                        yield result
            except Exception as e:
                logger.warning(f"Error in alternative parsing: {str(e)}")

    def _create_driver_pool(self, size: int = 3):
        """Create a pool of WebDriver instances for parallel content fetching"""
        drivers: list[webdriver.Chrome] = []
        for _ in range(size):
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
            ]
            for arg in arguments:
                options.add_argument(arg)

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.set_page_load_timeout(self.timeout)
            drivers.append(driver)
        return drivers

    async def run(self):
        """Execute web search yielding results as they are found"""
        main_driver = None
        content_drivers: list[webdriver.Chrome] = []
        content_tasks: list[asyncio.Task[WebSearchResult]] = []
        content_semaphore = asyncio.Semaphore(3)  # Limit parallel content fetches
        results_queue = asyncio.Queue[WebSearchResult]()
        found_results: list[WebSearchResult] = []

        try:
            # Start driver pools
            main_driver = await asyncio.to_thread(self)
            content_drivers = await asyncio.to_thread(self._create_driver_pool, 3)

            # Start search task
            search_task = asyncio.create_task(
                self._search_and_enqueue_results(main_driver, results_queue)
            )

            # Process results as they come in
            result_count = 0
            content_enrichment_tasks: list[asyncio.Task[WebSearchResult | None]] = []

            while True:
                try:
                    # Wait for a result or until search is complete
                    result = await asyncio.wait_for(results_queue.get(), timeout=0.5)
                    result_count += 1
                    found_results.append(result)

                    # Yield the basic result immediately
                    yield f"data: {result.model_dump_json()}\n\n"

                    # Start content enrichment in the background
                    driver_idx = len(content_enrichment_tasks) % len(content_drivers)
                    task = asyncio.create_task(
                        self._fetch_content_with_semaphore(
                            content_semaphore,
                            result.link,
                            driver_idx,
                            content_drivers,
                            result,
                        )
                    )
                    content_enrichment_tasks.append(task)

                    # Process completed content tasks
                    for task in list(content_tasks):
                        if task.done():
                            content_tasks.remove(task)
                            try:
                                updated_result = task.result()
                                if updated_result:
                                    yield f"data: {updated_result.model_dump_json()}\n\n"
                            except Exception as e:
                                logger.warning(
                                    f"Error processing content task: {str(e)}"
                                )

                except asyncio.TimeoutError:
                    # Check if search is complete
                    if search_task.done():
                        if search_task.exception():
                            logger.error(f"Search error: {search_task.exception()}")
                            yield f'data: {{"error": "Search failed: {str(search_task.exception())}"}}\n\n'

                        # No more results coming, wait for remaining content tasks
                        if not content_enrichment_tasks:
                            break

                        # Process any remaining content tasks
                        for task in asyncio.as_completed(content_enrichment_tasks):
                            try:
                                updated_result = await task
                                if updated_result:
                                    yield f"data: {updated_result.model_dump_json()}\n\n"
                            except Exception as e:
                                logger.warning(
                                    f"Error processing content task: {str(e)}"
                                )

                        # All done
                        break

            # If no results found at all
            if result_count == 0:
                yield 'data: {"error": "No search results found"}\n\n'

        except asyncio.CancelledError:
            logger.info("Search task was cancelled")
            raise
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            yield f'data: {{"error": "Search failed: {str(e)}"}}\n\n'
        finally:
            # Clean up drivers
            if main_driver:
                try:
                    main_driver.quit()
                except:
                    pass

            for driver in content_drivers:
                try:
                    driver.quit()
                except:
                    pass

    async def _search_and_enqueue_results(
        self, driver: webdriver.Chrome, queue: asyncio.Queue[WebSearchResult]
    ):
        """Search and enqueue results as they are found"""
        try:
            result_count = 0

            for page in range(self.pages):
                if result_count >= self.max_results:
                    break

                try:
                    # Construct and navigate to search URL
                    search_url = self._build_search_url(page)
                    await asyncio.to_thread(driver.get, search_url)

                    # Wait for search results to load
                    await asyncio.to_thread(
                        WebDriverWait(driver, self.timeout).until,
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.g, div.yuRUbf, div.MjjYud")
                        ),
                    )

                    # Extract and enqueue results for this page as they're found
                    page_source = await asyncio.to_thread(lambda: driver.page_source)

                    # Convert the generator to an async operation
                    for result in await asyncio.to_thread(
                        lambda: list(
                            self._extract_search_results_generator(page_source, page)
                        )
                    ):
                        await queue.put(result)
                        result_count += 1

                        if result_count >= self.max_results:
                            break

                    # Small delay between pages to avoid rate limiting
                    if page < self.pages - 1 and result_count < self.max_results:
                        await asyncio.sleep(0.2)

                except Exception as e:
                    logger.warning(f"Error on page {page+1}: {str(e)}")
                    continue  # Try next page

        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise

    async def _fetch_content_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        url: str,
        driver_idx: int,
        drivers: list[webdriver.Chrome],
        result: WebSearchResult,
    ):
        """Fetch content with semaphore to limit concurrent requests"""
        async with semaphore:
            try:
                content = await asyncio.to_thread(
                    self._fetch_website_content_optimized, url, drivers[driver_idx]
                )
                if content:
                    result.content = content
                return result
            except Exception as e:
                logger.warning(f"Error fetching content for {url}: {str(e)}")
                return None

    def _fetch_website_content_optimized(
        self, url: str, driver: webdriver.Chrome
    ) -> str:
        """Optimized version of content fetching"""
        try:
            # Navigate to URL
            driver.get(url)

            # Wait for page to load (shorter timeout)
            try:
                WebDriverWait(driver, self.timeout / 2).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except:
                pass  # Continue even if timeout - we might have partial content

            # Get metadata and content using streamlined approach
            content = ""

            # Fast metadata extraction
            try:
                # Execute JavaScript to get multiple metadata elements at once
                metadata = driver.execute_script(
                    """
                    return {
                        title: document.title || "",
                        metaDesc: document.querySelector('meta[name="description"]')?.content || "",
                        ogDesc: document.querySelector('meta[property="og:description"]')?.content || ""
                    }
                """
                )

                if metadata["title"]:
                    content += f"Title: {metadata['title']}\n\n"
                if metadata["metaDesc"] and len(metadata["metaDesc"]) > 50:
                    content += f"Description: {metadata['metaDesc']}\n\n"
                elif metadata["ogDesc"] and len(metadata["ogDesc"]) > 50:
                    content += f"Summary: {metadata['ogDesc']}\n\n"
            except:
                pass

            # Faster content extraction - single JavaScript call
            try:
                main_content = driver.execute_script(
                    """
                    // Try to find main content container
                    const selectors = [
                        "main", "article", "#content", ".content", '[role="main"]',
                        "section", ".main-content", ".post-content", ".entry-content",
                        "#main", ".main", ".body", ".post", ".entry"
                    ];
                    
                    // Try each selector
                    for (const selector of selectors) {
                        const elements = document.querySelectorAll(selector);
                        for (const el of elements) {
                            const text = el.textContent.trim();
                            if (text && text.length > 200) {
                                return text;
                            }
                        }
                    }
                    
                    // Fallback to paragraphs if no main content found
                    const paragraphs = document.querySelectorAll('p');
                    let pContent = '';
                    let count = 0;
                    
                    for (const p of paragraphs) {
                        const text = p.textContent.trim();
                        if (text && text.length > 30) {
                            pContent += text + "\\n";
                            count++;
                            if (count >= 8) break; // Limit to 8 paragraphs
                        }
                    }
                    
                    return pContent || document.body.textContent.trim();
                """
                )

                if main_content:
                    # Clean up content
                    lines = [
                        line.strip()
                        for line in main_content.split("\n")
                        if len(line.strip()) > 20
                    ]
                    cleaned_content = "\n".join(
                        lines[:12]
                    )  # Limit to 12 substantial lines

                    content += f"Content: {cleaned_content}\n"
            except:
                pass

            # Truncate if too long, but keep structured
            if len(content) > 2000:
                return content[:900] + "\n...\n" + content[-900:]

            return content.strip() or "No meaningful content could be extracted."

        except Exception as e:
            logger.warning(f"Error fetching content from {url}: {str(e)}")
            return f"Failed to fetch content: {str(e)}"


class AnthropicAgent(AnthropicTool):
    model: AnthropicModels = Field(default="claude-3-7-sonnet-20250219")
    messages: list[MessageParam] = Field(default_factory=list)
    tools: list[ToolParam] = Field(default_factory=list)
    max_tokens: int = Field(default_factory=get_random_int)
    _tool_classes_cache: tp.ClassVar[tp.Optional[dict[str, tp.Type[AnthropicTool]]]] = (
        None  # Class variable for caching
    )

    @classmethod
    def _get_subclasses(cls):
        if cls._tool_classes_cache is None:
            cls._tool_classes_cache = {
                cls.__name__: cls for cls in AnthropicTool.__subclasses__()
            }
        return cls._tool_classes_cache

    async def run(self) -> tp.AsyncGenerator[str, None]:
        client = self.__load__()

        # Use iterative approach instead of recursive
        processing_stack = [True]  # Start with processing the main request
        original_tools = self.tools.copy()

        while processing_stack:
            processing_stack.pop()  # Process the current item

            async with client.messages.stream(
                model=self.model,
                tools=self.tools,
                messages=self.messages,
                max_tokens=self.max_tokens,
            ) as response_stream:
                tool_classes = self._get_subclasses()

                async for raw_content_block in response_stream:
                    if raw_content_block.type == "content_block_stop":
                        if isinstance(raw_content_block.content_block, ToolUseBlock):
                            logger.info(
                                "Executing tool %s",
                                raw_content_block.content_block.name,
                            )
                            # Use list to collect chunks for efficiency
                            content_chunks: list[str] = []
                            tool_name = raw_content_block.content_block.name
                            tool_input = raw_content_block.content_block.input

                            async for chunk in (
                                tool_classes[tool_name].model_validate(tool_input).run()
                            ):
                                yield chunk
                                content_chunks.append(chunk)

                            content = "".join(content_chunks)
                            self.messages.append({"role": "user", "content": content})
                            self.tools = []
                            # Add to processing stack instead of recursion
                            processing_stack.append(True)
                            break  # Break from current stream to handle the new item in stack

                    elif raw_content_block.type == "content_block_delta":
                        if (
                            isinstance(raw_content_block.delta, TextDelta)
                            and raw_content_block.delta.text
                        ):
                            yield raw_content_block.delta.text

            # Restore original tools after processing a sub-request
            if not processing_stack:
                self.tools = original_tools
