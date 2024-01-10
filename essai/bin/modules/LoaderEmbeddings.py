
from essai.bin.loaders.DIRLoader import DIRLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


#FARE CONFIG SETTING


def SpliText():
    textSplitter = RecursiveCharacterTextSplitter(
            chunk_size = 256,
            chunk_overlap  = 64,
            length_function = len,
            add_start_index = True,
        )
    
    return textSplitter

def InitChromaDocsFromPath(path):
    loader = DIRLoader(path,metadata_columns=["user","category","created_at_year","created_at_month","created_at_day"],content_column="content")
    docs = loader.load()

    text_splitter = SpliText()

    all_splits = text_splitter.split_documents(docs)

    #print(all_splits)
    def split_list(input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]
            
    split_docs_chunked = split_list(all_splits, 166)
    return split_docs_chunked


def ProcessChunksFromLocal(path):
    loader = DIRLoader(path,metadata_columns=["user","category","created_at_year","created_at_month","created_at_day"],content_column="content")
    docs = loader.load()


    text_splitter = SpliText()

    all_splits = text_splitter.split_documents(docs)

    return all_splits