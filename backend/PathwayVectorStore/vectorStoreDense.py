import os
from loguru import logger
import pathway as pw
from pathway.xpacks.llm import parsers
from pathway.xpacks.llm.embedders import OpenAIEmbedder
from .vectorStore import VectorStoreServerModified
from .contextSplitter import ContextualRetrievalSplitter
from dotenv import load_dotenv
load_dotenv()


def make_dense_vector_store_server(
    source, 
    port: int,
    save_doc_summary: bool, 
    save_doc_path: str
) -> None:

    # table_args = {
    #     "parsing_algorithm": "pymupdf"
    # }

    # parser = parsers.OpenParse(table_args=table_args)\
    parser = parsers.ParseUnstructured(
        mode='single'
    )

    embedder = OpenAIEmbedder()
    # embedder = SentenceTransformerEmbedder(model="intfloat/e5-large-v2")
    splitter = ContextualRetrievalSplitter()

    vector_server = VectorStoreServerModified(
        source,
        embedder=embedder,
        splitter=splitter,  # no need to use splitter for with openai embedder
        parser=parser,
        save_doc_summary=save_doc_summary,
        save_doc_path=save_doc_path,
        store_meta_data_in_chunk=True,
    )

    # export TESSDATA_PREFIX=/usr/share/tesseract-orc
    # export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

    vector_server.run_server(
        host="127.0.0.1",
        port=port,
    )

if __name__=="__main__":

    table = pw.io.gdrive.read(
        object_id="1PKCELu34EgxIEp-tdZz2wpxAdIXF-e_e",
        service_user_credentials_file="./uploaded_files/credentials2.json",
        mode = "streaming",
        with_metadata = True
    )

    # pw.run()

    make_dense_vector_store_server(
        table,
        port=8765,
        save_doc_summary=True,
        save_doc_path="./document_data/document_summary.txt"
    )
