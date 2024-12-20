SPLADE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
DOCUMENT_CONTEXT_PROMPT = """
    DOCUMENT:
    {doc_content}
"""

CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    
    CHUNK:
    {chunk_content}

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
"""

FINAL_CHUNK_CONTEXT_PROMPT = """
    CHUNK:
    {chunk_content}
    -----------------

    CONTEXT:
    {doc_content}

    FILE NAME:
    {file_name}

    ----------------------------------
"""
