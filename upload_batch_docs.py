
from datetime import datetime
from pydoc import doc
from essai.bin.modules.LoaderEmbeddings import ProcessChunksFromLocal
from essai.config import settings
from essai.bin.modules.ChromaSingle import ChromaClass
import os
from essai.bin.modules.BertSingle import BertTopicClass
import csv
from langchain.schema import Document
import hashlib
import pandas as pd
import numpy as np
from essai.bin.transformersCustom.ConvertAndFormatDataset import build_dataframe_from_csv_uploaded,clean_text
from langchain.document_loaders import UnstructuredURLLoader

#print(langchain_chroma._persist_directory)

"""

#https://medium.com/data-reply-it-datatech/bertopic-topic-modeling-as-you-have-never-seen-it-before-abb48bbab2b2
#https://maartengr.github.io/BERTopic/getting_started/topicrepresentation/topicrepresentation.html#optimize-labels
#https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html#example

"""


if __name__ == '__main__':



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
    

    


