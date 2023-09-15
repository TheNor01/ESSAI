#https://gpt-index.readthedocs.io/en/latest/getting_started/concepts.html#retrieval-augmented-generation-rag

from llama_index import SimpleDirectoryReader,Document,set_global_service_context
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter
from llama_index import VectorStoreIndex
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms import LlamaCPP
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings

from llama_index import ServiceContext

# Make our printing look nice
from llama_index.schema import MetadataMode

from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler



required_exts = [".txt"]
filename_fn = lambda filename: {'file_name': filename} #metadata

# Load in Documents
documents = SimpleDirectoryReader('keywords_suggester/data/dataset/automotive',required_exts=required_exts,recursive=True,file_metadata=filename_fn,filename_as_id=True).load_data()
print(f"Loaded {len(documents)} docs")

metadata_template = "{key}: {value},"

for doc in documents:
    doc.set_content("text")
    doc.metadata_template = metadata_template
    doc.metadata= {"category" : "auto"}

print("INFO SINGLE")
print(documents[5].get_content(metadata_mode=MetadataMode.ALL))

# Parse the Documents into Nodes
#You can choose to define Nodes and all its attributes directly
#You may also choose to “parse” source Documents into Nodes through our NodeParser classes


text_splitter = SentenceSplitter(
  separator=" ",
  chunk_size=200,
  chunk_overlap=5,
)


print("PARSING")
parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
nodes = parser.get_nodes_from_documents(documents)

print(f"Parsed {len(nodes)} docs")


#LLM SECTION

template = """I will give you a question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])



model_url = "https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.8,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    #model_kwargs={"n_gpu_layers": 1},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

service_context = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)

#set as global
set_global_service_context(service_context)



#response = llm.complete("Hello! Can you tell me a poem about cats and dogs?")
#print(response.text)
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

index.storage_context.persist(persist_dir="keywords_suggester/index_storage")


query_engine = index.as_query_engine()
response = query_engine.query("Tell me something about car")
print(response)
