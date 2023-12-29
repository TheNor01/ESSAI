 #LLM SECTION
    
#https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever

#https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
#https://github.com/langchain-ai/langchain/discussions/9645
#self query use llm https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query
#https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed


#serve il self query

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import GPT4All
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.LLModel import LLModel
from keywords_suggester.config import settings



def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

llm = GPT4All(
    model="./keywords_suggester/storage/llm/mistral-7b-openorca.Q4_0.gguf",
    max_tokens=2048
)



# Run
settings.init()
persist_directory = settings.persist_directory+"init_dataset_small"+"/"
embed_model = settings.embed_model
collection_name_local = "TestCollection"

ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

mygpt = LLModel(ChromaDB)

docs = mygpt.SelfQuery("give me documents with category sport")
#docs = mygpt.StructuredQuery("give me sports documents with creation date equals to 2023-06-02. You have to treat date as string")

pretty_print_docs(docs)
