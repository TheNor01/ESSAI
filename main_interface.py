from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from keywords_suggester.bin.modules.LLModel import LLModel
from keywords_suggester.config import settings
from chainlit import on_message, on_chat_start
import chainlit as cl
from keywords_suggester.bin.prompts.prompts import load_query_gen_prompt,load_essai_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import logging

#https://docs.chainlit.io/integrations/langchain

logging.basicConfig(level = logging.INFO)

settings.init()
persist_directory = settings.persist_directory+settings.init_dataset+"/"
embed_model = settings.embed_model
collection_name_local = settings.collection_name

ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)

mygpt = LLModel(ChromaDB)

essai_prompt = load_essai_prompt()
query_gen_prompt = load_query_gen_prompt()
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(query_gen_prompt)

@on_chat_start
def init(): 
    llm = mygpt.llm
    memory = ConversationTokenBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000)
    
    docsearch = ChromaDB.CLIENT
    #retriever = docsearch.as_retriever(search_kwargs={"k": 10})
    retriever = mygpt.retriever

    messages = [SystemMessagePromptTemplate.from_template(essai_prompt)]
    # print('mem', user_session.get('memory'))
    messages.append(HumanMessagePromptTemplate.from_template("{question}"))
    prompt = ChatPromptTemplate.from_messages(messages)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=prompt)

    chain = ConversationalRetrievalChain(
            retriever=retriever,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            verbose=True,
            memory=memory,
            rephrase_question=False
        )
    cl.user_session.set("conversation_chain", chain)

    
@on_message
async def main(message: str):
        # Read chain from user session variable
        chain = cl.user_session.get("conversation_chain")

        # Run the chain asynchronously with an async callback
        res = await chain.arun({"question": message.content},callbacks=[cl.AsyncLangchainCallbackHandler()])

        print(res)
        # Send the answer and the text elements to the UI
        await cl.Message(content=res).send()
