import json
import nltk
import logging
import threading
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, TypeAlias, cast

import jmespath
import requests

import pathway as pw
import pathway.xpacks.llm.parsers
import pathway.xpacks.llm.splitters
from pathway.stdlib.indexing import default_usearch_knn_document_index
from pathway.stdlib.indexing.data_index import _SCORE, DataIndex
from pathway.stdlib.ml.classifiers import _knn_lsh
from pathway.xpacks.llm._utils import _coerce_sync

from pathway.xpacks.llm._utils import _unwrap_udf

import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import List
from openai import OpenAI
import re
from dotenv import load_dotenv
load_dotenv()

if TYPE_CHECKING:
    import langchain_core.documents
    import langchain_core.embeddings
    import llama_index.core.schema

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


DOCUMENT_SUMMARY_PROMPT = '''
You are an expert document summarizer. Your task is to process the provided document and create a concise and accurate summary. The summary should focus on the main ideas, key points, and any significant details while maintaining clarity and coherence. This summary will be used as input for a dynamic AI agent to perform further tasks.  

**Instructions**:  

1. Carefully read and analyze the document provided below.  
2. Extract the most critical information, themes, and insights.  
3. Ensure that the summary is no longer than 200-250 words (or adjust based on user needs).  
4. Write the summary in a way that is easy to understand, avoiding unnecessary technical jargon unless essential.  

**Input Document**:  

{doc_content}

**Output Summary**:  

Please provide your output in the following format:  

- **Main Theme**: (A one-sentence description of the overarching idea or purpose of the document.)  
- **Key Points**:  
  1. [First critical point]  
  2. [Second critical point]  
  3. [And so on...]  
- **Significant Insights**: (Optional) Highlight any additional information or implications worth noting.

'''


class VectorStoreServerModified:
    """
    Builds a document indexing pipeline and starts an HTTP REST server for nearest neighbors queries.

    Args:
        - docs: pathway tables typically coming out of connectors which contain source documents.
        - embedder: callable that embeds a single document
        - parser: callable that parses file contents into a list of documents
        - splitter: callable that splits long documents
        - doc_post_processors: optional list of callables that modify parsed files and metadata.
            any callable takes two arguments (text: str, metadata: dict) and returns them as a tuple.
    """

    def __init__(
        self,
        *docs: pw.Table,
        embedder: Callable[[str], list[float] | Coroutine] | pw.UDF,
        parser: Callable[[bytes], list[tuple[str, dict]]] | pw.UDF | None = None,
        splitter: Callable[[str], list[tuple[str, dict]]] | pw.UDF | None = None,
        doc_post_processors: (
            list[Callable[[str, dict], tuple[str, dict]] | pw.UDF] | None
        ) = None,
        save_doc_summary: bool = False,
        save_doc_path: str = "",
        store_meta_data_in_chunk: bool = True
    ):
        self.docs = docs
        self.doc_data = ""
        self.parser: Callable[[bytes], list[tuple[str, dict]]] = _unwrap_udf(
            parser if parser is not None else pathway.xpacks.llm.parsers.ParseUtf8()
        )
        self.doc_post_processors = []
        self.save_doc_summary = save_doc_summary
        self.save_doc_path = save_doc_path
        self.store_meta_data_in_chunk = store_meta_data_in_chunk

        if self.save_doc_summary:
            if not os.path.exists(self.save_doc_path):
                raise FileNotFoundError(f"File {self.save_doc_path} does not exist")
            if not self.save_doc_path.endswith(".txt"):
                raise ValueError(f"File {self.save_doc_path} should be a .txt file")
        
        if doc_post_processors:
            self.doc_post_processors = [
                _unwrap_udf(processor)
                for processor in doc_post_processors
                if processor is not None
            ]

        self.splitter = _unwrap_udf(
            splitter
            if splitter is not None
            else pathway.xpacks.llm.splitters.null_splitter
        )
        if isinstance(embedder, pw.UDF):
            self.embedder = embedder
        else:
            self.embedder = pw.udf(embedder)

        # detect the dimensionality of the embeddings
        self.embedding_dimension = len(_coerce_sync(self.embedder.__wrapped__)("."))
        logging.debug("Embedder has dimension %s", self.embedding_dimension)

        self._graph = self._build_graph()
        self.input_files = 0

        if self.save_doc_summary:
            self.remove_doc_summary_in_file()

    @classmethod
    def from_langchain_components(
        cls,
        *docs,
        embedder: "langchain_core.embeddings.Embeddings",
        parser: Callable[[bytes], list[tuple[str, dict]]] | None = None,
        splitter: "langchain_core.documents.BaseDocumentTransformer | None" = None,
        **kwargs,
    ):
        """
        Initializes VectorStoreServer by using LangChain components.

        Args:
            - docs: pathway tables typically coming out of connectors which contain source documents
            - embedder: Langchain component for embedding documents
            - parser: callable that parses file contents into a list of documents
            - splitter: Langchaing component for splitting documents into parts
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "Please install langchain_core: `pip install langchain_core`"
            )

        generic_splitter = None
        if splitter:
            generic_splitter = lambda x: [  # noqa
                (doc.page_content, doc.metadata)
                for doc in splitter.transform_documents([Document(page_content=x)])
            ]

        async def generic_embedded(x: str) -> list[float]:
            res = await embedder.aembed_documents([x])
            return res[0]

        return cls(
            *docs,
            embedder=generic_embedded,
            parser=parser,
            splitter=generic_splitter,
            **kwargs,
        )

    @classmethod
    def from_llamaindex_components(
        cls,
        *docs,
        transformations: list["llama_index.core.schema.TransformComponent"],
        parser: Callable[[bytes], list[tuple[str, dict]]] | None = None,
        **kwargs,
    ):
        """
        Initializes VectorStoreServer by using LlamaIndex TransformComponents.

        Args:
            - docs: pathway tables typically coming out of connectors which contain source documents
            - transformations: list of LlamaIndex components. The last component in this list
                is required to inherit from LlamaIndex `BaseEmbedding`
            - parser: callable that parses file contents into a list of documents
        """
        try:
            from llama_index.core.base.embeddings.base import BaseEmbedding
            from llama_index.core.ingestion.pipeline import run_transformations
            from llama_index.core.schema import BaseNode, MetadataMode, TextNode
        except ImportError:
            raise ImportError(
                "Please install llama-index-core: `pip install llama-index-core`"
            )
        try:
            from llama_index.legacy.embeddings.base import (
                BaseEmbedding as LegacyBaseEmbedding,
            )

            legacy_llama_index_not_imported = True
        except ImportError:
            legacy_llama_index_not_imported = False

        def node_transformer(x: str) -> list[BaseNode]:
            return [TextNode(text=x)]

        def node_to_pathway(x: list[BaseNode]) -> list[tuple[str, dict]]:
            return [
                (node.get_content(metadata_mode=MetadataMode.NONE), node.extra_info)
                for node in x
            ]

        if transformations is None or not transformations:
            raise ValueError("Transformations list cannot be None or empty.")

        if not isinstance(transformations[-1], BaseEmbedding) and (
            legacy_llama_index_not_imported
            or not isinstance(transformations[-1], LegacyBaseEmbedding)
        ):
            raise ValueError(
                f"Last step of transformations should be an instance of {BaseEmbedding.__name__}, "
                f"found {type(transformations[-1])}."
            )

        embedder = cast(BaseEmbedding, transformations.pop())

        async def embedding_callable(x: str) -> list[float]:
            embedding = await embedder.aget_text_embedding(x)
            return embedding

        def generic_transformer(x: str) -> list[tuple[str, dict]]:
            starting_node = node_transformer(x)
            final_node = run_transformations(starting_node, transformations)
            return node_to_pathway(final_node)

        return VectorStoreServerModified(
            *docs,
            embedder=embedding_callable,
            parser=parser,
            splitter=generic_transformer,
            **kwargs,
        )
    

    @staticmethod
    def _clean_text(text: str) -> str:
        # Remove HTML tags using regex
        text_cleaned = re.sub(r'<[^>]*>', '', text)
        # Replace \xa0 with a space and newlines with a space
        text_cleaned = text_cleaned.replace('\xa0', ' ').replace('\n', ' ')
        # Replace multiple spaces with a single space
        text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
        return text_cleaned
    
    def remove_doc_summary_in_file(self):
        with open(self.save_doc_path, "w") as f:
            f.write("")

    def get_doc_summary(self, text: str) -> str:
        text = self._clean_text(text)
        prompt = DOCUMENT_SUMMARY_PROMPT.format(doc_content=text)
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        ).choices[0].message.content
        return response
    
    def save_doc_summary_in_file(self, text: str) -> None:
        # Generate the document summary
        text = self.get_doc_summary(text)
        logger.debug(f"summary - {text}")
        self.input_files += 1

        # Read the existing file content if it exists
        content = ""
        if os.path.exists(self.save_doc_path):
            with open(self.save_doc_path, "r") as f:
                content = f.read().strip()
        
        # Append the new document summary to the content
        key = "<document>"
        new_entry = f"{key}\n: {text}\n"
        logger.debug("Appending new document summary")
        content = content + "\n\n\n" + new_entry
        content += "</document>\n"

        # Save the updated content back to the text file
        with open(self.save_doc_path, "a") as f:  # Open in append mode
            f.write(content)


    def _build_graph(self) -> dict:
        """
        Builds the pathway computation graph for indexing documents and serving queries.
        """
        docs_s = self.docs
        if not docs_s:
            raise ValueError(
                """Please provide at least one data source, e.g. read files from disk:

pw.io.fs.read('./sample_docs', format='binary', mode='static', with_metadata=True)
"""
            )
        if len(docs_s) == 1:
            (docs,) = docs_s
        else:
            docs: pw.Table = docs_s[0].concat_reindex(*docs_s[1:])  # type: ignore

        @pw.udf
        def parse_doc(data: bytes, metadata) -> list[pw.Json]:
            rets = self.parser(data)
            metadata = metadata.value
            return [dict(text=ret[0], metadata={**metadata, **ret[1]}) for ret in rets]  # type: ignore

        parsed_docs = docs.select(data=parse_doc(docs.data, docs._metadata)).flatten(
            pw.this.data
        )

        @pw.udf
        def post_proc_docs(data_json: pw.Json) -> pw.Json:
            data: dict = data_json.value  # type:ignore
            text = data["text"]
            metadata = data["metadata"]

            for processor in self.doc_post_processors:
                text, metadata = processor(text, metadata)

            return dict(text=text, metadata=metadata)  # type: ignore

        parsed_docs = parsed_docs.select(data=post_proc_docs(pw.this.data))
        
        @pw.udf
        def split_doc(data_json: pw.Json) -> list[pw.Json]:
            logger.debug(f"splitting doc")
            data: dict = data_json.value  # type:ignore
            text = data["text"]
            metadata = data["metadata"]

            if self.save_doc_summary:
                self.save_doc_summary_in_file(text)

            if self.store_meta_data_in_chunk:
                riched_text = text + "<THIS_IS_A_SPLITTER>" + metadata["name"]
                rets = self.splitter(riched_text)
            else:
                rets = self.splitter(text)

            return [
                dict(text=ret[0], metadata={**metadata, **ret[1]})  # type:ignore
                for ret in rets
            ]
        
        chunked_docs = parsed_docs.select(data=split_doc(pw.this.data)).flatten(
            pw.this.data
        )

        chunked_docs += chunked_docs.select(text=pw.this.data["text"].as_str())
        logger.debug(f"chunks formed and contexted{chunked_docs}")

        knn_index = default_usearch_knn_document_index(
            chunked_docs.text,
            chunked_docs,
            dimensions=self.embedding_dimension,
            metadata_column=chunked_docs.data["metadata"],
            embedder=self.embedder,
        )

        parsed_docs += parsed_docs.select(
            modified=pw.this.data["metadata"]["modified_at"].as_int(),
            indexed=pw.this.data["metadata"]["seen_at"].as_int(),
            path=pw.this.data["metadata"]["path"].as_str(),
        )

        stats = parsed_docs.reduce(
            count=pw.reducers.count(),
            last_modified=pw.reducers.max(pw.this.modified),
            last_indexed=pw.reducers.max(pw.this.indexed),
            paths=pw.reducers.tuple(pw.this.path),
        )
        return locals()

    class StatisticsQuerySchema(pw.Schema):
        pass

    class QueryResultSchema(pw.Schema):
        result: pw.Json

    class InputResultSchema(pw.Schema):
        result: list[pw.Json]

    @pw.table_transformer
    def statistics_query(
        self, info_queries: pw.Table[StatisticsQuerySchema]
    ) -> pw.Table[QueryResultSchema]:
        stats = self._graph["stats"]

        # VectorStore statistics computation
        @pw.udf
        def format_stats(counts, last_modified, last_indexed) -> pw.Json:
            if counts is not None:
                response = {
                    "file_count": counts,
                    "last_modified": last_modified,
                    "last_indexed": last_indexed,
                }
            else:
                response = {
                    "file_count": 0,
                    "last_modified": None,
                    "last_indexed": None,
                }
            return pw.Json(response)

        info_results = info_queries.join_left(stats, id=info_queries.id).select(
            result=format_stats(stats.count, stats.last_modified, stats.last_indexed)
        )
        return info_results

    class FilterSchema(pw.Schema):
        metadata_filter: str | None = pw.column_definition(
            default_value=None, description="Metadata filter in JMESPath format"
        )
        filepath_globpattern: str | None = pw.column_definition(
            default_value=None, description="An optional Glob pattern for the file path"
        )

    InputsQuerySchema: TypeAlias = FilterSchema

    @staticmethod
    def merge_filters(queries: pw.Table):
        @pw.udf
        def _get_jmespath_filter(
            metadata_filter: str, filepath_globpattern: str
        ) -> str | None:
            ret_parts = []
            if metadata_filter:
                metadata_filter = (
                    metadata_filter.replace("'", r"\'")
                    .replace("`", "'")
                    .replace('"', "")
                )
                ret_parts.append(f"({metadata_filter})")
            if filepath_globpattern:
                ret_parts.append(f"globmatch('{filepath_globpattern}', path)")
            if ret_parts:
                return " && ".join(ret_parts)
            return None

        queries = queries.without(
            *VectorStoreServerModified.FilterSchema.__columns__.keys()
        ) + queries.select(
            metadata_filter=_get_jmespath_filter(
                pw.this.metadata_filter, pw.this.filepath_globpattern
            )
        )
        return queries

    @pw.table_transformer
    def inputs_query(
        self, input_queries: pw.Table[InputsQuerySchema]
    ) -> pw.Table[InputResultSchema]:
        docs = self._graph["docs"]
        # TODO: compare this approach to first joining queries to dicuments, then filtering,
        # then grouping to get each response.
        # The "dumb" tuple approach has more work precomputed for an all inputs query
        all_metas = docs.reduce(metadatas=pw.reducers.tuple(pw.this._metadata))

        input_queries = self.merge_filters(input_queries)

        @pw.udf
        def format_inputs(
            metadatas: list[pw.Json] | None, metadata_filter: str | None
        ) -> list[pw.Json]:
            metadatas = metadatas if metadatas is not None else []
            assert metadatas is not None
            if metadata_filter:
                metadatas = [
                    m
                    for m in metadatas
                    if jmespath.search(
                        metadata_filter, m.value, options=_knn_lsh._glob_options
                    )
                ]

            return metadatas

        input_results = input_queries.join_left(all_metas, id=input_queries.id).select(
            all_metas.metadatas, input_queries.metadata_filter
        )
        input_results = input_results.select(
            result=format_inputs(pw.this.metadatas, pw.this.metadata_filter)
        )
        return input_results

    class RetrieveQuerySchema(pw.Schema):
        query: str = pw.column_definition(
            description="Your query for the similarity search",
            example="Pathway data processing framework",
        )
        k: int = pw.column_definition(
            description="The number of documents to provide", example=2
        )
        metadata_filter: str | None = pw.column_definition(
            default_value=None, description="Metadata filter in JMESPath format"
        )
        filepath_globpattern: str | None = pw.column_definition(
            default_value=None, description="An optional Glob pattern for the file path"
        )
    
    class RetrieveQueryAllChunksSchema(pw.Schema):
        query: str = pw.column_definition(
            description="Your query for the similarity search",
            example="Pathway data processing framework",
        )
        metadata_filter: str | None = pw.column_definition(
            default_value=None, description="Metadata filter in JMESPath format"
        )
        filepath_globpattern: str | None = pw.column_definition(
            default_value=None, description="An optional Glob pattern for the file path"
        )

    @pw.table_transformer
    def retrieve_query(
        self, retrieval_queries: pw.Table[RetrieveQuerySchema]
    ) -> pw.Table[QueryResultSchema]:
        knn_index: DataIndex = self._graph["knn_index"]

        # Relevant document search
        retrieval_queries = self.merge_filters(retrieval_queries)

        retrieval_results = retrieval_queries + knn_index.query_as_of_now(
            retrieval_queries.query,
            number_of_matches=retrieval_queries.k,
            collapse_rows=True,
            metadata_filter=retrieval_queries.metadata_filter,
        ).select(
            result=pw.coalesce(pw.right.data, ()),  # replace None results with []
            score=pw.coalesce(pw.right[_SCORE], ()),
        )

        retrieval_results = retrieval_results.select(
            result=pw.apply_with_type(
                lambda x, y: pw.Json(
                    sorted(
                        [{**res.value, "dist": -score} for res, score in zip(x, y)],
                        key=lambda x: x["dist"],  # type: ignore
                    )
                ),
                pw.Json,
                pw.this.result,
                pw.this.score,
            )
        )

        return retrieval_results
    
    
    @pw.table_transformer
    def retrieve_query_all_chunks(
        self, retrieval_queries: pw.Table[RetrieveQueryAllChunksSchema]
    ) -> pw.Table[QueryResultSchema]:
        knn_index: DataIndex = self._graph["knn_index"]

        # Relevant document search
        retrieval_queries = self.merge_filters(retrieval_queries)

        retrieval_results = retrieval_queries + knn_index.query_as_of_now(
            retrieval_queries.query,
            number_of_matches=3000,
            collapse_rows=True,
            metadata_filter=retrieval_queries.metadata_filter,
        ).select(
            result=pw.coalesce(pw.right.data, ()),  # replace None results with []
            score=pw.coalesce(pw.right[_SCORE], ()),
        )

        retrieval_results = retrieval_results.select(
            result=pw.apply_with_type(
                lambda x, y: pw.Json(
                    sorted(
                        [{**res.value, "dist": -score} for res, score in zip(x, y)],
                        key=lambda x: x["dist"],  # type: ignore
                    )
                ),
                pw.Json,
                pw.this.result,
                pw.this.score,
            )
        )

        return retrieval_results
    
    class EmptySchema(pw.Schema):
        pass

    class getDocumentTextSchema(pw.Schema):
        result : pw.Json

    @pw.table_transformer
    def get_document_text(
        self,
        inp : pw.Table[EmptySchema]
    ) -> pw.Table[getDocumentTextSchema]:
        
        content = ""
        if self.save_doc_summary:
            with open(self.save_doc_path, "r") as f:
                content = f.read().strip()        
        
        res = {
            "text": [content]
        }

        inp = inp.select(
            result = res
        )

        return inp

    @property
    def index(self) -> DataIndex:
        return self._graph["knn_index"]

    def run_server(
        self,
        host,
        port,
        threaded: bool = False,
        with_cache: bool = True,
        cache_backend: (
            pw.persistence.Backend | None
        ) = pw.persistence.Backend.filesystem("./Cache"),
        **kwargs,
    ):
        """
        Builds the document processing pipeline and runs it.

        Args:
            - host: host to bind the HTTP listener
            - port: to bind the HTTP listener
            - threaded: if True, run in a thread. Else block computation
            - with_cache: if True, embedding requests for the same contents are cached
            - cache_backend: the backend to use for caching if it is enabled. The
              default is the disk cache, hosted locally in the folder ``./Cache``. You
              can use ``Backend`` class of the
              [`persistence API`](/developers/api-docs/persistence-api/#pathway.persistence.Backend)
              to override it.
            - kwargs: optional parameters to be passed to :py:func:`~pathway.run`.

        Returns:
            If threaded, return the Thread object. Else, does not return.
        """

        webserver = pw.io.http.PathwayWebserver(host=host, port=port, with_cors=True)

        # TODO(move into webserver??)
        def serve(route, schema, handler, documentation):
            queries, writer = pw.io.http.rest_connector(
                webserver=webserver,
                route=route,
                methods=("GET", "POST"),
                schema=schema,
                autocommit_duration_ms=50,
                delete_completed_queries=False,
                documentation=documentation,
            )
            writer(handler(queries))

        serve(
            "/v1/retrieve",
            self.RetrieveQuerySchema,
            self.retrieve_query,
            pw.io.http.EndpointDocumentation(
                summary="Do a similarity search for your query",
                description="Request the given number of documents from the "
                "realtime-maintained index.",
                method_types=("GET",),
            ),
        )
        serve(
            "/v1/retrieve_all_chunks",
            self.RetrieveQueryAllChunksSchema,
            self.retrieve_query_all_chunks,
            pw.io.http.EndpointDocumentation(
                summary="Do a similarity search for your query and return all chunks",
                description="Request all the chunks from the "
                "realtime-maintained index.",
                method_types=("GET",),
            ),
        )
        serve(
            "/v1/statistics",
            self.StatisticsQuerySchema,
            self.statistics_query,
            pw.io.http.EndpointDocumentation(
                summary="Get current indexer stats",
                description="Request for the basic stats of the indexer process. "
                "It returns the number of documents that are currently present in the "
                "indexer and the time the last of them was added.",
                method_types=("GET",),
            ),
        )
        serve(
            "/v1/inputs",
            self.InputsQuerySchema,
            self.inputs_query,
            pw.io.http.EndpointDocumentation(
                summary="Get indexed documents list",
                description="Request for the list of documents present in the indexer. "
                "It returns the list of metadata objects.",
                method_types=("GET",),
            ),
        )
        serve(
            "/v1/get_doc_text",
            self.EmptySchema,
            self.get_document_text,
            pw.io.http.EndpointDocumentation(
                summary="Get the text of the document",
                description="Request for the text of the document",
                method_types=("GET",),
            ),
        )
        def run():
            if with_cache:
                if cache_backend is None:
                    raise ValueError(
                        "Cache usage was requested but the backend is unspecified"
                    )
                persistence_config = pw.persistence.Config(
                    cache_backend,
                    persistence_mode=pw.PersistenceMode.UDF_CACHING,
                )
            else:
                persistence_config = None

            pw.run(
                monitoring_level=pw.MonitoringLevel.NONE,
                persistence_config=persistence_config,
                **kwargs,
            )

        if threaded:
            t = threading.Thread(target=run, name="VectorStoreServer")
            t.start()
            return t
        else:
            run()

    def __repr__(self):
        return f"VectorStoreServer({str(self._graph)})"
