from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Create topics
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs[0:10_000])


similar_topics, similarity = topic_model.find_topics("motor", top_n=5)
print(topic_model.get_topic(similar_topics[0]))