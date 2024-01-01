
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import GPT4All
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel,RunnableBranch
from langchain.chains import RetrievalQA
from langchain_core.pydantic_v1 import BaseModel
from typing import Literal
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

        """
        prompt = get_query_constructor_prompt(
            self.document_content_description,
            self.metadata_field_info,
        )

        print(prompt.format(query="{query}"))
        """


        self.llm = GPT4All(
            model="./keywords_suggester/storage/llm/mistral-7b-openorca.Q4_0.gguf",
            max_tokens=2048
        )


    def SelfQuery(self,query):

        #https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/#constructing-from-scratch-with-lcel

        #https://python.langchain.com/docs/expression_language/cookbook/prompt_llm_parser
        #https://python.langchain.com/docs/expression_language/cookbook/retrieval
        #https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
        #https://python.langchain.com/docs/use_cases/question_answering/

        chain = load_query_constructor_runnable(
                self.llm, self.document_content_description, self.metadata_field_info
                )
    
        retriever = SelfQueryRetriever(
            query_constructor=chain,
            vectorstore=self.chroma.CLIENT,
            structured_query_translator=ChromaTranslator(),
            verbose=True
        )

        docs = retriever.invoke(query)
        #docs = retriever.get_relevant_documents(query)

        #LIMIT https://github.com/langchain-ai/langchain/issues/13961
        if(len(docs)==0):
            print("NO DATA")
        return docs
    
    """
    def ContextQuestionQuery(self,query):

        template = "Answer the question based only on the following context:
                    {context}

                    Question: {question}
                    "
    
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
             {"context": self.chroma.CLIENT.as_retriever() ,"question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        retriever = SelfQueryRetriever(
            query_constructor=self.query_constructor,
            vectorstore=self.chroma.CLIENT,
            structured_query_translator=ChromaTranslator(),
        )   

        docs = retriever.invoke(query)
        return docs
    """
    
    #
    def RagChainWithSource(self,question):
        # Build prompt
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
        qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.chroma.CLIENT.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            
        )

        #stuff as default https://python.langchain.com/docs/modules/chains/document/stuff

        question = question
        result = qa_chain({"query": question})
        # Check the result of the query
        print(result["result"])
        # Check the source document from where we 
        print(result["source_documents"][0])
        
    #Come usare? Oppure https://python.langchain.com/docs/expression_language/cookbook/embedding_router
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


    def SummarizeContent(self):
        doc_prompt = PromptTemplate.from_template("{page_content}")

        chain = (
            {
                "content": lambda docs: "\n\n".join(
                    format_document(doc, doc_prompt) for doc in docs
                )
            }
            | PromptTemplate.from_template("Summarize the following content:\n\n{content}")
            | self.llm
            | StrOutputParser()
        )

        text = """The large earthquake struck just off the Noto Peninsula at a little after 4pm local time.
                It was very shallow, and the shaking was very severe, bringing down buildings in towns and villages along the coast.
                For several hours after the quake struck, authorities said the Sea of Japan coast could be hit by tsunamis of up to five metres.
                Tens of thousands of people were told to leave their homes and head for higher ground.
                It immediately brought back memories March 2011 when a 15m tsunami inflicted massive destruction along Japan’s north-east coast, killing nearly 20,000 people.
                The threat of a major tsunami has now passed, and the severe tsunami warning that was issued for much of the north-west coast has now been downgraded. But the damage is still severe.
                Older houses have been brought down, roads torn up, bridges and railways severely damaged.
                Some people are reported trapped under collapsed buildings and hospitals are reporting many injured."""

        docs = [
            Document(
                page_content=split,
                metadata={"source": "https://www.bbc.com/news/live/world-asia-67856144"},
            )
            for split in text.split()
        ]

        print(chain.invoke(docs))

        #then pass it to Router?