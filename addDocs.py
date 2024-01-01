
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
from keywords_suggester.bin.transformersCustom.ConvertAndFormatDataset import build_dataframe_from_csv_uploaded,clean_text
from langchain.document_loaders import UnstructuredURLLoader

#print(langchain_chroma._persist_directory)

"""

#https://medium.com/data-reply-it-datatech/bertopic-topic-modeling-as-you-have-never-seen-it-before-abb48bbab2b2
#https://maartengr.github.io/BERTopic/getting_started/topicrepresentation/topicrepresentation.html#optimize-labels
#https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html#example

"""


if __name__ == '__main__':


    #TODO AGGIUMGERE PICCOLA SEZIONE STREAMING
    #https://faust.readthedocs.io/en/latest/


    #TODO capire se conviene fare prima la preview e poi classificare o al contrario


    settings.init()
    persist_directory = settings.persist_directory+settings.init_dataset+"/"
    embed_model = settings.embed_model

    collection_name_local = settings.collection_name
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

    print("SCANNING UPLOAD DIR.. --> "+settings.upload_directory)
    files = os.listdir(settings.upload_directory)
    print(files)

    #SELECTED_UPLOAD = "simpleUpload.csv" #no category
    SELECTED_UPLOAD = "simpleUpload.csv" #no category
    CHOOSED_FILE = [k for k in files if SELECTED_UPLOAD in k][0]

    print(CHOOSED_FILE) #csv is without category
    
    #csv file content|USER
    #dir -> category/.csvfile
    #LOAD A simple csv file user,text or structured --> if not structured bertopic will categorize them.
    
    #PROCESS TO BERTOPIC
    BERT_NAME = settings.bert_name
    BERT = BertTopicClass(BERT_NAME,restore=1)

    #TODO possiamo fare la preview con BERTOPIC prima di aggiungere effettivamente gli utenti alla collezione CHROMA


    df = pd.read_csv(os.path.join(settings.upload_directory,SELECTED_UPLOAD),delimiter="|")
    #df = df.drop('CATEGORY', axis=1)

    texts = df['CONTENT'].tolist() #dovrebbero esserci un numero di sample pari a K hdbscan --> Edit. servono molti sample
    clean_texts = list(map(clean_text, texts))
    print("CSV UPLOAD LENGHT: ->"+str(len(clean_texts)))
    min_similarity_topics = 0.8
    
    
    #PREVIEW
    BERT.PreviewMerge(clean_texts,min_similarity_topics) #if category is not present


    upload_df = build_dataframe_from_csv_uploaded(BERT,CHOOSED_FILE)
    print(upload_df)

    """
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
            topics_list.append(topic_mapped)

    upload_df = pd.DataFrame(zip(documents_list, topics_list, users_list,ids_list),columns=['content','category', 'user','ids'])
    upload_df['created_at']=CREATED_TIME_NOW
    #create column metadata as dict
    """
    

    metadata_df = upload_df[["user","category","created_at"]]
    metadata_dict = metadata_df.to_dict(orient='records')
    #print(metadata_dict)

    documents_list = upload_df["content"].to_list()
    #print(documents_list)
    #LOAD FULL DF OR SPLIT BY CHUNK AND FOLLOW STANDARD FLOW? how handle multiple ids same text (chunks)
    
    DOCS_TO_UPLOAD =  [Document(page_content=d,metadata=m) for d,m in zip(documents_list, metadata_dict)]
    
    #print(DOCS_TO_UPLOAD)
    print("UPLOADING... "+str(len(DOCS_TO_UPLOAD)))
    
    #i have to upload them into CHROMADB
    ChromaDB.AddDocsToCollection(DOCS_TO_UPLOAD)
    

    


