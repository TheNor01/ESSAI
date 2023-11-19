from bertopic import BERTopic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
#https://maartengr.github.io/BERTopic/getting_started/representation/llm.html#langchain


persist_directory = "keywords_suggester/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

collection=vectordb.get(include=["documents","embeddings"])

computed_document = collection["documents"]
computed_embeddings = collection["embeddings"]


#print(computed_document)

topic_model = BERTopic(embedding_model=embed_model,verbose=True)

topics, prob = topic_model.fit_transform(computed_document)

print(topic_model.get_topic_info())

print(topic_model.get_topic(0))
print(topic_model.get_document_info(computed_document))

