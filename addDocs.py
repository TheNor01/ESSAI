
from datetime import datetime
from pydoc import doc
from keywords_suggester.bin.modules.LoaderEmbeddings import ProcessChunksFromLocal
from keywords_suggester.config import settings
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
import os
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
import csv
from langchain.schema import Document
import hashlib
import pandas as pd
import numpy as np
#print(langchain_chroma._persist_directory)

"""

#https://medium.com/data-reply-it-datatech/bertopic-topic-modeling-as-you-have-never-seen-it-before-abb48bbab2b2
#https://maartengr.github.io/BERTopic/getting_started/topicrepresentation/topicrepresentation.html#optimize-labels



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
    BERT = BertTopicClass(restore=1)
    topic_info = BERT.main_model.get_topic_info()
    
    dict_topic_name = dict(zip(topic_info['Topic'], topic_info['Name']))
    
    documents_list,topics_list,users_list,ids_list = [],[],[],[]
    CREATED_TIME_NOW =  datetime.now().strftime("%Y-%m-%d")
    
    
    
    with open(os.path.join(settings.upload_directory,SELECTED_UPLOAD)) as file_obj: 
        reader_obj = csv.reader(file_obj,delimiter="|") 
        next(reader_obj, None) # SKIP HEADERS
        for row in reader_obj: 
            local_doc = row[1]
            ids_list.append(hashlib.md5(local_doc.encode()))
            documents_list.append(local_doc)
            users_list.append(row[0])
            print(local_doc)
            
            topics, probs = BERT.main_model.transform(local_doc)
            print(topics) #topics di zero dovrebbe essere il massimo
            print(probs)
            #index = np.argmax(probs)
            max_topic = topics[0]
            
            #print(probs.shape)
            #print(probs)
            #topic_mapped = [dict_topic_name[key] for key in topics]
            topic_mapped = dict_topic_name[max_topic] 
            #print(topic_mapped)
            topics_list.append(topic_mapped)

    upload_df = pd.DataFrame(zip(documents_list, topics_list, users_list,ids_list),columns=['content','category', 'user','ids'])
    upload_df['created_at']=CREATED_TIME_NOW
    #create column metadata as dict
    
    print(upload_df)
    #["user","category","created_at"]
    metadata_df = upload_df[["user","category","created_at"]]
    
    metadata_dict = metadata_df.to_dict(orient='records')
    
    print(metadata_dict)

    #LOAD FULL DF OR SPLIT BY CHUNK AND FOLLOW STANDARD FLOW? how handle multiple ids same text (chunks)
    
    DOCS_TO_UPLOAD =  [Document(page_content=d,metadata=m) for d,m in zip(documents_list, metadata_dict)]
    
    print(DOCS_TO_UPLOAD)
    print("UPLOADING... "+str(len(DOCS_TO_UPLOAD)))
    
    ChromaDB.AddDocsToCollection(DOCS_TO_UPLOAD)
    
  
    #BERT.main_model.visualize_topics().show()
    #BERT.main_model.visualize_barchart(top_n_topics=10).show()
    
    #i have to upload them into CHROMADB

