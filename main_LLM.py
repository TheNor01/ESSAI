 #LLM SECTION
    
#https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever

#https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
#https://github.com/langchain-ai/langchain/discussions/9645
#self query use llm https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query
#https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed


#serve il self query

from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.LLModel import LLModel
from keywords_suggester.config import settings

import logging

logging.basicConfig(level = logging.INFO)

def pretty_print_docs(docs):
    if(isinstance(docs,str)):
        print(docs)
    if(isinstance(docs,dict)):
        for i in docs:
            print (i, docs[i])
    else:
        #print(docs)
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content + "Metadata:" + str(d.metadata) for i, d in enumerate(docs)]))


# Run
settings.init()
persist_directory = settings.persist_directory+settings.init_dataset+"/"
embed_model = settings.embed_model
collection_name_local = settings.collection_name

ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

mygpt = LLModel(ChromaDB)

#docs = mygpt.SelfQuery("I want to know what article user 4a2bd reads")
#docs = mygpt.SelfQuery("Give me some food documents created in year 2024")
#docs = mygpt.SelfQuery("Based on his documents, create a sample text for the user dc16c")
#docs = mygpt.SelfQuery("What are some documents about food which contains the word chicken")
#docs = mygpt.StructuredQuery("give me sports documents with creation date equals to 2023-06-02. You have to treat date as string")

#docs = mygpt.retriever.invoke("I want to know what article user 4a2bd reads")
#pretty_print_docs(docs)


#docs = mygpt.RagQA("Tell me something about volkswagen")

docs = mygpt.SummarizeContent()
pretty_print_docs(docs)



#mygpt.SummarizeContent()
