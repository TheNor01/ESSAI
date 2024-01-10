from essai.bin.modules.ChromaSingle import ChromaClass
from essai.bin.modules.LLModel import LLModel
from essai.config import settings
from chainlit import on_message, on_chat_start
import chainlit as cl
from essai.bin.prompts.prompts import load_query_gen_prompt,load_essai_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.schema.runnable.config import RunnableConfig


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

@cl.on_chat_start
async def on_chat_start():
    runnable = mygpt.runnable
    cl.user_session.set("runnable", runnable)

    
@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    async with cl.Step(type="run", name="QA ESSAI"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

    await msg.send()

