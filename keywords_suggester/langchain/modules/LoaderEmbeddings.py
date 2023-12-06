
from ast import Str
from loaders.DIRLoader import DIRLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def InitChromaDocsFromPath(path):
    loader = DIRLoader(path,metadata_columns=["user","category","created_at"],content_column="content")
    docs = loader.load()

    #print(docs[0])
    print(len(docs))

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
        add_start_index = True,
    )

    all_splits = text_splitter.split_documents(docs)

    print(all_splits)


    def split_list(input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]
            
    split_docs_chunked = split_list(all_splits, 166)

    return split_docs_chunked


def ProcessChunksFromLocal(path):
    loader = DIRLoader(path,metadata_columns=["user","category","created_at"],content_column="content")
    docs = loader.load()

    #print(docs[0])
    #print(len(docs))

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
        add_start_index = True,
    )

    all_splits = text_splitter.split_documents(docs)

    return all_splits