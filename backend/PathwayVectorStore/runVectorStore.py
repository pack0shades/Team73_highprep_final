import pathway as pw
import threading
import time
from .vectorRetriever import VectorStoreRetriever
from .vectorStoreDense import make_dense_vector_store_server
from .vectorStoreSparse import make_sparse_vector_store_server

def run_vector_store(
    credential_path: str,
    object_id: str
)-> None:
    
    # Read the table from Google Drive
    table = pw.io.gdrive.read(
        object_id=object_id,
        service_user_credentials_file=credential_path,
        mode = "streaming",
        with_metadata = True
    )

    # Start the servers in parallel
    t1 = threading.Thread(
        target=make_dense_vector_store_server,
        args=(table, 8765, True, "./document_data/document_summary.txt")
    )

    t2 = threading.Thread(
        target=make_sparse_vector_store_server,
        args=(table, 8766, False, "")
    )

    t1.start()
    t2.start()

    print ("Both servers initiated")

    #--------------------------check-1-------------------------------------

    while True:
        try:
            client = VectorStoreRetriever("localhost", 8765)
            num_input_files = client.get_num_input_files()
            if num_input_files > 0:
                break
        except:
            pass


    while True:
        try:
            client = VectorStoreRetriever("localhost", 8766)
            num_input_files = client.get_num_input_files()
            if num_input_files > 0:
                break
        except:
            pass

    #--------------------------check-2-------------------------------------
    
    while True:
        try:
            client = VectorStoreRetriever("localhost", 8765)
            num_input_files = client.get_num_input_files()
            if num_input_files > 0:
                break
        except:
            pass


    while True:
        try:
            client = VectorStoreRetriever("localhost", 8766)
            num_input_files = client.get_num_input_files()
            if num_input_files > 0:
                break
        except:
            pass

    return ""

if __name__ == "__main__":
    run_vector_store(
        credential_path="./uploaded_files/credentials2.json",
        object_id="",
    )
