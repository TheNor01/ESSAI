
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import chromadb
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import TFIDFRetriever
from langchain.document_loaders import DirectoryLoader


persist_directory = "keywords_suggester/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


"""

loader = DirectoryLoader('keywords_suggester/data/dataset/automotive', glob="**/*.txt")
docs = loader.load()

retrieverTF = TFIDFRetriever.from_documents(docs)
result = retrieverTF.get_relevant_documents("tesla")

print(len(result))
"""

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)


#https://python.langchain.com/docs/integrations/retrievers/
#https://python.langchain.com/docs/modules/data_connection/retrievers/time_weighted_vectorstore

#raccomandation system: vespa,weaviate

question = "i love art"
docs = vectordb.similarity_search(question,k=5)

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)


exit()
model_url = "https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


"""


template = 
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]

prompt = PromptTemplate(template=template, input_variables=["prompt"])
"""


llm = LlamaCpp(
    # You can pass in the URL to a GGML model to download it automatically
    model_path="resources/llama-2-7b-chat.Q4_0.gguf",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 30}), return_source_documents=True)

query = "How to repair a car?"
result = qa({"query": query})
#first successful try

print(result["result"])
print(result["source_documents"])