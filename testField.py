
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.config import settings
from sentence_transformers import SentenceTransformer
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass


BERT_NAME = "test1"
BERT = BertTopicClass(BERT_NAME,restore=1)


settings.init()
persist_directory = settings.persist_directory+"init_dataset_small"+"/"
embed_model = settings.embed_model
collection_name_local = "TestCollection"
ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)


embeded_model = SentenceTransformer('all-mpnet-base-v2')
collection=ChromaDB.CLIENT.get(include=["documents","embeddings","metadatas"])

computed_document = collection["documents"]
computed_metadata = collection["metadatas"]

#print(computed_document)
print(type(computed_metadata))

timestamps = [ x["created_at"] for x in computed_metadata]
#print(timestamps)

print(len(computed_document))
print(len(timestamps))

BERT.TopicOverTime(computed_document,timestamps)