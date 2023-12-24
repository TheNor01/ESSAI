
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings
from sentence_transformers import SentenceTransformer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.RetriverSingle import RetrieverSingle

settings.init()
persist_directory = settings.persist_directory+"init_dataset"+"/"
embed_model = settings.embed_model
collection_name_local = settings.collection_name
ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)



#BERT_NAME = settings.bert_name
#BERT = BertTopicClass(BERT_NAME,restore=1)

#candidate_topics = ["war", "politics", "sports"]
#labels = BERT.SuggestLabels("US officials - from Mr Biden to Secretary of State Antony Blinken and Defence Secretary Lloyd Austin - have continually affirmed what they present as Israel's right to self-defence, and declared that a military operation which stops short of removing Hamas from power would only guarantee more attacks.",candidate_topics)
#print(labels)



#BERT.ChangeLabelMeaning(dictToChange={-1: "outliers"})
#print(BERT.TopicInfo())

"""
topic = "healthy food"
similar_topics, similarity = BERT.FindSimilarTopics(topic, top_n=5)

print("SIMILAR TOPICS TO ->"+topic.upper())
#print(BERT.main_model.get_topic(similar_topics[1]))

for sim in similar_topics:
    print(BERT.main_model.get_topic(sim))
    print("========== \n")
"""

#[[1, 2][3, 4]]
#BERT.ManualMergeTopics([[11,16]])
#BERT.ReduceTopics(50)


ChromaDB.HistogramUsersTopics()
#BERT.Genereate_WC(11)