
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings
from sentence_transformers import SentenceTransformer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.RetriverSingle import RetrieverSingle
import pickle


settings.init()
persist_directory = settings.persist_directory+"init_dataset"+"/"
embed_model = settings.embed_model
collection_name_local = settings.collection_name
ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

#collection=ChromaDB.CLIENT.get(include=["documents","metadatas"])



#BERT_NAME = settings.bert_name
#BERT = BertTopicClass(BERT_NAME,restore=1)

#candidate_topics = ["war", "politics", "sports"]
#labels = BERT.SuggestLabels("US officials - from Mr Biden to Secretary of State Antony Blinken and Defence Secretary Lloyd Austin - have continually affirmed what they present as Israel's right to self-defence, and declared that a military operation which stops short of removing Hamas from power would only guarantee more attacks.",candidate_topics)
#print(labels)


#BERT.TopicOverTime(collection)


#BERT.ChangeLabelMeaning(dictToChange={-1: "outliers"})
#print(BERT.TopicInfo())

#BERT.Genereate_WC(0)
#BERT.GenerateTopioChart()

ChromaDB.HistogramUsersTopics()
#BERT.Genereate_WC(11)