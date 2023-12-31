
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import GPT4All

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
    load_query_constructor_runnable
)

#https://maartengr.github.io/BERTopic/api/representation/langchain.html#bertopic.representation._langchain.LangChain

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance



@singleton
class LLModel():
    def __init__(self,ChromaSingle):

        self.chroma = ChromaSingle
        self.document_content_description = "A document article read by an user with his metadata section"


        #TODO fix date filter created_at
        self.metadata_field_info = [
                            AttributeInfo(
                                name="created_at",
                                description="The creation date of document, should have format yyyy-mm-day",
                                type="string",
                            ),
                            AttributeInfo(
                                name="category",
                                description="The category of document",
                                type="string",
                            ),
                            AttributeInfo(
                                name="source",
                                description="The document the chunk is from",
                                type="string",
                            ),
                            AttributeInfo(
                                name="user",
                                description="The user who reads the documents",
                                type="string",
                            ),
        ]

        prompt = get_query_constructor_prompt(
            self.document_content_description,
            self.metadata_field_info,
        )

        print(prompt.format(query="{query}"))

        self.llm = GPT4All(
            model="./keywords_suggester/storage/llm/mistral-7b-openorca.Q4_0.gguf",
            max_tokens=2048
        )

        self.chain = load_query_constructor_runnable(
                self.llm, self.document_content_description, self.metadata_field_info
                )
        
        #result = self.chain.invoke({"query": "give me documents with category sports or category food"})
        #print(result)

        #output_parser = StructuredQueryOutputParser.from_components()
        #self.query_constructor = prompt | self.llm | output_parser

    def SelfQuery(self,query):

        #https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/#constructing-from-scratch-with-lcel

        #https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
        #https://python.langchain.com/docs/expression_language/cookbook/retrieval
        #https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
        #https://python.langchain.com/docs/use_cases/question_answering/

        """
        
        retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.chroma.CLIENT,
            self.document_content_description,
            self.metadata_field_info,
            #structured_query_translator=ChromaTranslator(),
            verbose=True
        )
        
        """
        retriever = SelfQueryRetriever(
            query_constructor=self.chain,
            vectorstore=self.chroma.CLIENT,
            structured_query_translator=ChromaTranslator(),
            verbose=True
        )

        docs = retriever.invoke(query)
        #docs = retriever.get_relevant_documents(query)
        if(len(docs)==0):
            print("NO DATA")
        return docs
    
    def StructuredQuery(self,query):
        retriever = SelfQueryRetriever(
            query_constructor=self.query_constructor,
            vectorstore=self.chroma.CLIENT,
            structured_query_translator=ChromaTranslator(),
        )   

        docs = retriever.invoke(query)
        return docs
        
        