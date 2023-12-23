
import numpy as np
from sklearn.datasets import fetch_20newsgroups
#from datasets import load_dataset
from keywords_suggester.config import settings


from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings

if __name__ == "__main__":

    SAVE=1

    #TODO move collection to settings

    settings.init()
    persist_directory = settings.persist_directory+"init_dataset"+"/"
    embed_model = settings.embed_model
    collection_name_local = "default"
    ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)


    collection=ChromaDB.CLIENT.get(include=["documents","embeddings","metadatas"])

    computed_document = collection["documents"]
    computed_embeddings = collection["embeddings"]

    print(ChromaDB.CLIENT._collection.name)
    print(len(computed_embeddings))
    print(len(computed_document))


    computed_document_array=np.array([np.array(xi) for xi in computed_embeddings])
    print(computed_document_array.shape)

    BERT_NAME = settings.bert_name

    BERT = None
    if(SAVE==1):
        #main_model = BERTopic(embedding_model=embeded_model,verbose=True)
        BERT = BertTopicClass(BERT_NAME)
        BERT.UpdateDocuments(computed_document)
        topics, prob = BERT.main_model.fit_transform(documents=computed_document,embeddings=computed_document_array)#,embeddings=computed_document_array)

        #print(BERT.main_model.embedding_model)

        similar_topics, similarity = BERT.FindSimilarTopics("food", top_n=5)
        print(BERT.main_model.get_topic(similar_topics[1]))
        
        BERT.PersistModel()
        #BERT.save("./keywords_suggester/models_checkpoint/bert", serialization="pytorch", save_ctfidf=True, save_embedding_model=embeded_model)
    else:
        print("RESTORING MODEL BERT ...")
        BERT = BertTopicClass(BERT_NAME,restore=1)


    print(BERT.main_model.get_topic_info())


    