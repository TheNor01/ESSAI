
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import GPT4All
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel,RunnableBranch
from langchain.chains import RetrievalQA
from langchain_core.pydantic_v1 import BaseModel
from typing import Literal
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from operator import itemgetter
from langchain_core.prompts import format_document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
            model="./keywords_suggester/storage/llm/mistral-7b-openorca.Q4_0.gguf",
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
            base_compressor=compressor, base_retriever=self.retriever
        )



    def SelfQuery(self,query):

        
        pass
        #docs = retriever.invoke(query)
        #docs = retriever.get_relevant_documents(query)

        #LIMIT https://github.com/langchain-ai/langchain/issues/13961
        #if(len(docs)==0):
        #    print("NO DATA")
        #return docs
    
    #Used to QA special documents, such as user profile in order to retriver answer according to a question
    def RagQA(self,question):


        # Build prompt
        prompt = hub.pull("rlm/rag-prompt")
        #SummarizePrompt = PromptTemplate.from_template("Summarize this content:\n\n{context}")


        #BUILD SELF RETRIVER
        retriever = self.compression_retriever
               
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)
        
        output = rag_chain_with_source.invoke(question)

        if(len(output)):
            print('NO DATA')
            return

        return output


        """
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            #retriever=self.chroma.CLIENT.as_retriever(search_kwargs={"filter": {"user":'dc16c'}}
            retriever = retriever,
            return_source_documents=True,
            chain_type="map-reduce",
            chain_type_kwargs={"prompt": prompt},
        )

        #stuff as default https://python.langchain.com/docs/modules/chains/document/stuff

        question = question
        result = qa_chain({"query": question})
        # Check the result of the query
        print(result["result"])
        # Check the source document from where we 
        print(result["source_documents"][0])
        """
        
    #Come usare? Oppure https://python.langchain.com/docs/expression_language/cookbook/embedding_router --> prendere dai topic?
    def RouterPrompt(self,question):
        physics_template = """You are a very smart physics professor. \
        You are great at answering questions about physics in a concise and easy to understand manner. \
        When you don't know the answer to a question you admit that you don't know.

        Here is a question:
        {input}"""
        physics_prompt = PromptTemplate.from_template(physics_template)

        math_template = """You are a very good mathematician. You are great at answering math questions. \
        You are so good because you are able to break down hard problems into their component parts, \
        answer the component parts, and then put them together to answer the broader question.

        Here is a question:
        {input}"""
        
        math_prompt = PromptTemplate.from_template(math_template)
        general_prompt = PromptTemplate.from_template(
            "You are a helpful assistant. Answer the question as accurately as you can.\n\n{input}"
        )

        prompt_branch = RunnableBranch(
            (lambda x: x["topic"] == "math", math_prompt),
            (lambda x: x["topic"] == "physics", physics_prompt),
            general_prompt,
        )


        class TopicClassifier(BaseModel):
            "Classify the topic of the user question"

            topic: Literal["math", "physics", "general"]
            "The topic of the user question. One of 'math', 'physics' or 'general'."

        classifier_function = convert_pydantic_to_openai_function(TopicClassifier)

        llm =  self.llm.bind(
                functions=[classifier_function], function_call={"name": "TopicClassifier"}
                )
        parser = PydanticAttrOutputFunctionsParser(
            pydantic_schema=TopicClassifier, attr_name="topic"
        )

        classifier_chain = llm | parser

        final_chain = (
            RunnablePassthrough.assign(topic=itemgetter("input") | classifier_chain)
            | prompt_branch
            | self.llm()
            | StrOutputParser()
        )

        out = final_chain.invoke(
            {
                "input":  question
            }
        )

        print(out)


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