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

import logging

logging.basicConfig(level = logging.INFO)

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content + "Metadata:" + str(d.metadata) for i, d in enumerate(docs)]))


# Run
settings.init()
persist_directory = settings.persist_directory+settings.init_dataset+"/"
embed_model = settings.embed_model
collection_name_local = settings.collection_name

ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

mygpt = LLModel(ChromaDB)

#docs = mygpt.SelfQuery("what food is healthy? Return more than 10 results")
#docs = mygpt.SelfQuery("Give me some food documents created in year 2024")
#docs = mygpt.SelfQuery("Based on his documents, create a sample text for the user dc16c")
#docs = mygpt.SelfQuery("What are some documents about food which contains the word chicken")
#docs = mygpt.StructuredQuery("give me sports documents with creation date equals to 2023-06-02. You have to treat date as string")

docs = mygpt.RagQA("What food is healthy?")

pretty_print_docs(docs)



#mygpt.SummarizeContent()
