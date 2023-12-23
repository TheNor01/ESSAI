
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import GPT4All


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

        self.llm = GPT4All(
            model="./keywords_suggester/storage/llm/mistral-7b-openorca.Q4_0.gguf",
            max_tokens=2048
        )


    def SelfQuery(self,query):

        
        document_content_description = "A document article"
        retriever = SelfQueryRetriever.from_llm(
            self.llm,
            self.chroma.CLIENT,
            document_content_description,
            self.metadata_field_info,
            verbose=True
        )

        docs = retriever.get_relevant_documents(query)
        return docs