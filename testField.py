
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings
from sentence_transformers import SentenceTransformer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.RetriverSingle import RetrieverSingle
import pickle
from keywords_suggester.bin.modules.LLModel import LLModel

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

mygpt = LLModel(ChromaDB)

#docs = mygpt.SelfQuery("I want to know what article user 4a2bd reads")
#docs = mygpt.SelfQuery("Give me some food documents created in year 2024")
#docs = mygpt.SelfQuery("Based on his documents, create a sample text for the user dc16c")
#docs = mygpt.SelfQuery("What are some documents about food which contains the word chicken")
#docs = mygpt.StructuredQuery("give me sports documents with creation date equals to 2023-06-02. You have to treat date as string")

#docs = mygpt.retriever.invoke("I want to know what article user 4a2bd reads")
#pretty_print_docs(docs)


docs = mygpt.RagQA("Give me 5 healthy food")