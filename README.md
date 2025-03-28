# AnyTools

![Cover](/cover.webp)

## Introduction

**AnyTools** is a Python library designed to provide a uniform interface for interacting with tool object engines from various Large Language Model (LLM) providers. It emphasizes structured input and output, a self-describing `json_schema` embedded into the `Tool` base class, and integrations with providers such as `groq`, `openai`, and `mistralai`.

This project aims to improve **developer experience (DX)** and promote **Clean Code** practices.

## Key Features

- **Uniform interface**: Enables seamless interaction with different tool implementations.
- **Native integrations**: Direct support for `groq`, `openai`, and `mistralai`.
- **JSON Schema support**: Ensures structured tool descriptions.
- **Flexible loading mechanism**: Allows custom logic API providers.
- **Built-in `WebSearch` tool**: A milestone example of integration.
- **Streaming response support**: Asynchronous execution using `run`.

## Installation

Install **AnyTools** via `pip`:

```sh
pip install anytools git+https://github.com/bahamondeX/anytools.git
```

## Basic Usage

Example of creating a custom tool to split text into sentences using `spaCy`:

```python
import typing as tp
from anytools import Tool
import spacy

class SentenceChunker(Tool[spacy.language.Language]):
    """Chunks large English or Spanish text into sentences."""
    lang: tp.Literal["en", "es"]
    text: str

    def __load__(self):
        if self.lang == "en":
            return spacy.load("en_core_web_sm")
        return spacy.load("es_core_news_sm")

    async def run(self):
        for chunk in self.__load__()(self.text).sents:
            yield chunk.text
```

### Asynchronous Execution

```python
import asyncio

async def main():
    tool = SentenceChunker(lang="en", text="This is a test. It contains two sentences.")
    async for sentence in tool.run():
        print(sentence)

asyncio.run(main())
```

Expected output:
```sh
This is a test.
It contains two sentences.
```

## Architecture

**AnyTools** is built around a generic `Tool` base class, which serves as a container for specialized tools. Implementing a custom tool requires:

1. Defining input attributes as instance variables.
2. Implementing the `__load__()` method to initialize required resources.
3. Implementing the `run()` method to execute the tool's logic.

## Integrations

**AnyTools** currently supports integrations with the following LLM providers:

- **OpenAI**: Enables tools compatible with `openai.tools`.
- **Groq**: Allows tools in processing pipelines using `groq`.
- **MistralAI**: Support for `MistralAI` models.
- **WebSearch**: Built-in tool for performing web searches.

## Roadmap

Planned future developments include:
- Support for **additional LLM providers**.
- Implementation of extra tools such as **automatic translation**, **text summarization**, etc.
- Enhanced customization for tool loading mechanisms.

## Contributing

To contribute to **AnyTools**, follow these steps:

1. Fork the repository.
2. Create a new branch (`feature/new-tool`).
3. Implement your feature and write tests.
4. Submit a Pull Request with a detailed description.

Your contributions are welcome to help improve **AnyTools**!

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

