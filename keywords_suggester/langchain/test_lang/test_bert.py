from bertopic import BERTopic
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
#https://maartengr.github.io/BERTopic/getting_started/representation/llm.html#langchain

SAVE=1
persist_directory = "keywords_suggester/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

collection=vectordb.get(include=["documents","embeddings","metadatas"])

computed_document = collection["documents"]
computed_embeddings = collection["embeddings"]

print(len(computed_document))

#print(computed_document)
if(SAVE==1):
    topic_model = BERTopic(embedding_model=embed_model,verbose=True)
    topic_model.save("./keywords_suggester/models_checkpoint/bert", serialization="pytorch", save_ctfidf=True, save_embedding_model=embed_model)
else:
    topic_model = BERTopic.load("./keywords_suggester/models_checkpoint/bert")

topics, prob = topic_model.fit_transform(computed_document)




print(topic_model.get_topic_info()) # we can access the frequent topics that were generated:

print(topic_model.get_topic(0)) #topic_model.get_topic(1, full=True)
print(topic_model.get_document_info(computed_document))

topic_model.visualize_topics()
#topic_model.visualize_barchart()
similar_topics, similarity = topic_model.find_topics("motor", top_n=5)

#https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html#saveload-bertopic-model
#https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#finding-similar-topics-between-models
#https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#multimodal-data
#https://maartengr.github.io/BERTopic/getting_started/topicreduction/topicreduction.html
#https://maartengr.github.io/BERTopic/getting_started/semisupervised/semisupervised.html
#https://maartengr.github.io/BERTopic/faq.html
"""

# Fine-tune topic representations with GPT
openai.api_key = "sk-..."
representation_model = OpenAI(model="gpt-3.5-turbo", chat=True)
topic_model = BERTopic(representation_model=representation_model)


or
conviene per avere una cosa più custom
model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    top_n_words=5,
    language='english',
    calculate_probabilities=True,
    verbose=True
)
"""
