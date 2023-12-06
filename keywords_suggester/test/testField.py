
from bin.modules.ChromaSingle import ChromaClass
from langchain.embeddings import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
persist_directory = "keywords_suggester/index_storage_lang"

singleton_instance_1 = ChromaClass(persist_directory,embed_model)
singleton_instance_2 = ChromaClass(persist_directory,embed_model)


print(singleton_instance_1 is singleton_instance_2)  # True