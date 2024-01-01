
from datetime import datetime
from pydoc import doc
from keywords_suggester.config import settings
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
import os
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
import csv
from langchain.schema import Document
import pandas as pd
import numpy as np
from keywords_suggester.bin.modules.ScrapingSingle import ScrapingHTML


#print(langchain_chroma._persist_directory)

"""

#https://medium.com/data-reply-it-datatech/bertopic-topic-modeling-as-you-have-never-seen-it-before-abb48bbab2b2
#https://maartengr.github.io/BERTopic/getting_started/topicrepresentation/topicrepresentation.html#optimize-labels
#https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html#example

"""


if __name__ == '__main__':


    #TODO AGGIUMGERE PICCOLA SEZIONE STREAMING
    #https://faust.readthedocs.io/en/latest/


    settings.init()
    persist_directory = settings.persist_directory+settings.init_dataset+"/"
    embed_model = settings.embed_model

    collection_name_local = settings.collection_name
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

    
    #PROCESS TO BERTOPIC
    BERT_NAME = settings.bert_name
    BERT = BertTopicClass(BERT_NAME,restore=1)


    Scraping = ScrapingHTML("2023-31-12","english")
    links = ["https://www.ansa.it/sito/notizie/sport/2023/12/31/ciclismo-australia-rohan-dennis-arrestato-per-la-morte-della-moglie_f44d310f-1106-4921-840f-5007151e7472.html"]
    content = Scraping.StartRequest(links)

    print(content)





    

    


