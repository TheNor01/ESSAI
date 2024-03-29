
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.chains.query_constructor.base import (
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


       
        self.metadata_field_info = [
                            AttributeInfo(
                                name="created_at_year",
                                description="The creation year of document",
                                type="integer",
                            ),
                            AttributeInfo(
                                name="created_at_month",
                                description="The creation month of document",
                                type="integer",
                            ),
                            AttributeInfo(
                                name="created_at_day",
                                description="The creation day of document",
                                type="integer",
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

        self.llm = GPT4All(
            model="./essai/storage/llm/mistral-7b-openorca.Q4_0.gguf",
            max_tokens=4096
        )


        chain = load_query_constructor_runnable(
                self.llm, self.document_content_description, self.metadata_field_info
                )
    
        compressor = LLMChainExtractor.from_llm(self.llm)

        self.retriever = SelfQueryRetriever(
            query_constructor=chain,
            vectorstore=self.chroma.CLIENT,
            structured_query_translator=ChromaTranslator(),
            verbose=True
        )

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.chroma.CLIENT.as_retriever()
        )

        self.prompt=hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

        self.rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)


        self.runnable = (
        {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
        | self.prompt
        | self.llm
        | StrOutputParser()
        )

    
    #Used to QA special documents, such as user profile in order to retriver answer according to a question
    def RagQA(self,question):


        # Build prompt
        prompt = hub.pull("rlm/rag-prompt")
        #SummarizePrompt = PromptTemplate.from_template("Summarize this content:\n\n{context}")

        
        #BUILD SELF RETRIVER
        #retriever = self.chroma.CLIENT.as_retriever()
        retriever = self.compression_retriever
               
        #output = qa_chain.invoke(question)
        output = self.rag_chain_with_source.invoke(question)
        
        print(type(output))

        if(len(output)==0):
            print('NO DATA')
            exit()

        return output


    def SummarizeContent(self,question): #summerize documents - map reduce in order to fill context
        prompt = PromptTemplate.from_template(
         "Summarize the main themes in these retrieved docs: {docs}"
        )


        # Chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        chain = {"docs": format_docs} | prompt | self.llm | StrOutputParser()

        # Run
        question = question
        docs = self.chroma.CLIENT.similarity_search(question)
        output = chain.invoke(docs)

        print(output)