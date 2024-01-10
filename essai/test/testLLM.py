
from langchain.chains import RetrievalQA
import chromadb
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings

model_url = "https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
persist_directory = "essai/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

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
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 30}), return_source_documents=True)

query = "How many documents there are?"
result = qa({"query": query})