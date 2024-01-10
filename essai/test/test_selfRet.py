

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

persist_directory = "essai/index_storage_lang"

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embed_model)

collection=vectordb.get()
print(collection.keys())
print(len(collection["ids"])) #lui ha 834 chunks rispetto ai doc di partenza
print((collection["metadatas"][0])) #lui ha 834 chunks

metadata_field_info=[
    AttributeInfo(
        name="category",
        description="The categories of the document", 
        type="string", 
    ),
    AttributeInfo(
        name="user",
        description="user of document", 
        type="string", 
    )
]

document_content_description = "A simple document"
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


llm = LlamaCpp(
    # You can pass in the URL to a GGML model to download it automatically
    model_path="resources/llama-2-7b-chat.Q4_0.gguf",
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    temperature=0.2,
    max_tokens=2000,
    top_p=1,
    n_ctx=2048,
    n_gpu_layers=0,
    callback_manager=callback_manager, 
    verbose=True, # Verbose is required to pass to the callback manager
)


retriever = SelfQueryRetriever.from_llm(llm, vectordb, document_content_description, metadata_field_info, verbose=True)

#out = retriever.get_relevant_documents("What are some documents about automotive")
#print(len(out))

out2= retriever.get_relevant_documents("People who like soccers")
print(len(out2))