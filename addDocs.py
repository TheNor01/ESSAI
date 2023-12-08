
from pydoc import doc
from keywords_suggester.bin.modules.LoaderEmbeddings import ProcessChunksFromLocal
from keywords_suggester.config import settings
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
import os
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
import csv
#print(langchain_chroma._persist_directory)

"""

https://stackoverflow.com/questions/76482987/chroma-database-embeddings-none-when-using-get

{'ids': ['46c66dae-594c-11ee-92b8-0a925a2cd92a'], 
'embeddings': None, 
'metadatas': [{'category': 'sciences', 'row': 0, 'source': 'keywords_suggester/data_transformed/dataset/sciences/18.csv', 
'start_index': 0, 'user': 'd80b7'}], 
'documents': ['Video of Crazy-faced cats don’t win the adoption game\tCrazy-faced cats don’t win the adoption']}



#collection = persistent_client.get_collection("langchain")
collection = langchain_chroma._collection
print(collection.count())


chunks = ProcessChunksFromLocal("keywords_suggester/data_transformed_upload/dataset/dummy")

langchain_chroma.add_documents(documents=chunks)
print(langchain_chroma._collection.count())


print(langchain_chroma._persist_directory)
langchain_chroma.persist()

print("COLLECTION PERSISTED")




for chunk in chunks[0:1]:
    print(chunk.page_content)
    query_result = embed_model.embed_query(chunk.page_content)
    print(query_result[0:3])
    print(len(query_result))




    collection.add(
        #embeddings=query_result,
        documents= [chunk.page_content],
        metadatas= [chunk.metadata],
        ids = ["111111"]
    )



print("There are", langchain_chroma._collection.count(), "in the collection")

result = collection.get(
    ids=["111111"],
)

print(result)
"""


if __name__ == '__main__':

    settings.init()
    persist_directory = settings.persist_directory+"init_dataset_small"+"/"
    embed_model = settings.embed_model

    collection_name_local = "TestCollection"
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

    print("SCANNING UPLOAD DIR.. --> "+settings.upload_directory)
    files = os.listdir(settings.upload_directory)
    print(files)

    SELECTED_UPLOAD = "simpleUpload.csv"
    CHOOSED_FILE = [k for k in files if SELECTED_UPLOAD in k]

    print(CHOOSED_FILE) #csv is without category
    
    #csv file content|USER
    #dir -> category/.csvfile
    #LOAD A simple csv file user,text or structured --> if not structured bertopic will categorize them.
    
    #PROCESS TO BERTOPIC
    documents = []
    with open(os.path.join(settings.upload_directory,SELECTED_UPLOAD)) as file_obj: 
        reader_obj = csv.reader(file_obj,delimiter="|") 
        for row in reader_obj: 
            documents.append(row[1])
            print(row[1])

    BERT = BertTopicClass(restore=1)
    topics, probs = BERT.main_model.transform(documents)

    print(topics)


    print(BERT.main_model.get_topic_info())

