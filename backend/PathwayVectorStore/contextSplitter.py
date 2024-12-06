# Copyright © 2024 Pathway

"""
A library of text spliiters - routines which slit a long text into smaller chunks.
"""

import pathway as pw
from pathway.optional_import import optional_imports

import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from typing import List
from tqdm import tqdm
import pandas as pd
import re
from .config import (
    DOCUMENT_CONTEXT_PROMPT,
    CHUNK_CONTEXT_PROMPT,
    FINAL_CHUNK_CONTEXT_PROMPT
)
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


class ContextualRetrievalSplitter(pw.UDF):
    """
    Splits a given string or a list of strings into chunks based on token
    count.

    This splitter tokenizes the input texts and splits them into smaller parts ("chunks")
    ensuring that each chunk has a token count between `min_tokens` and
    `max_tokens`. It also attempts to break chunks at sensible points such as
    punctuation marks.

    All arguments set default which may be overridden in the UDF call

    Args:
        min_tokens: minimum tokens in a chunk of text.
        max_tokens: maximum size of a chunk in tokens.
        encoding_name: name of the encoding from `tiktoken`.

    Example:

    >>> from pathway.xpacks.llm.splitters import TokenCountSplitter
    >>> import pathway as pw
    >>> t  = pw.debug.table_from_markdown(
    ...     '''| text
    ... 1| cooltext'''
    ... )
    >>> splitter = TokenCountSplitter(min_tokens=1, max_tokens=1)
    >>> t += t.select(chunks = splitter(pw.this.text))
    >>> pw.debug.compute_and_print(t, include_id=False)
    text     | chunks
    cooltext | (('cool', pw.Json({})), ('text', pw.Json({})))
    """

    def __init__(
        self
    ):
        super().__init__()

        self.pages_chunk_size = 3000
        self.chunk_size = 1000

        self.page_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.pages_chunk_size,
            chunk_overlap=200
        )

        self.chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=200
        )

    @staticmethod
    def _get_chunk_summary(doc: str, chunk: str) -> str:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a useful assistant"},
                {
                    "role": "user",
                    "content": (
                        DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)
                        + CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)
                    ),
                },
            ],
        )
        return completion.choices[0].message.content
    
    def process_page(self, page, metadata) -> List[str]:
        page_chunks = self.chunk_splitter.split_text(page)
        context = [self._get_chunk_summary(page, chunk) for chunk in page_chunks]
        
        res = [FINAL_CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk, doc_content=cxt, file_name=metadata) 
            for chunk, cxt in zip(page_chunks, context)]
        
        # logger.debug(f"Page chunks: {res}")
        return res
    
    @staticmethod
    def _clean_text(text: str) -> str:
        # Remove HTML tags using regex
        text_cleaned = re.sub(r'<[^>]*>', '', text)
        # Replace \xa0 with a space and newlines with a space
        text_cleaned = text_cleaned.replace('\xa0', ' ').replace('\n', ' ')
        # Replace multiple spaces with a single space
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
        return text_cleaned
    
    def __wrapped__(self, txt: str, **kwargs) -> list[tuple[str, dict]]:
        data = txt.split("<THIS_IS_A_SPLITTER>")

        if len(data) > 1:
            txt = data[0]
            metadata = data[1]
        else:
            txt = data[0]
            metadata = ""
        
        txt = self._clean_text(txt)
        pages = self.page_splitter.split_text(txt)

        chunks = []

        with ThreadPoolExecutor() as executor:
            future_to_page = [executor.submit(self.process_page, page, metadata) for page in pages]

            chunks_temp = [future.result() for future in tqdm(
                                                as_completed(future_to_page), 
                                                total=len(pages), desc="Chunking and getting context")]
            
        for page_chunks in chunks_temp:
            chunks.extend(page_chunks)

        # logger.debug(f"Chunks: {chunks}")
        logger.debug(f"Chunks length: {len(chunks)}")
        logger.debug(f"chunk type - {type(chunks)}")
        logger.debug(f"chunk type - {type(chunks[0])}")

        return [
            (chunk, {})
            for chunk in chunks
        ]

    def __call__(self, text: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Split given strings into smaller chunks.

        Args:
            - messages (ColumnExpression[str]): Column with texts to be split
            - **kwargs: override for defaults set in the constructor
        """
        return super().__call__(text, **kwargs)
