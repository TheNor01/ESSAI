
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



BERT_NAME = settings.bert_name
BERT = BertTopicClass(BERT_NAME,restore=1)

print(BERT.TopicInfo())

BERT.VisualizeTopics()

similar_topics, similarity = BERT.FindSimilarTopics("food", top_n=5)
print(BERT.main_model.get_topic(similar_topics[1]))