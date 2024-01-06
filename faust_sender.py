import asyncio
from faust_producer import Content, crawler
from selenium import webdriver
from selenium.webdriver.support.events import EventFiringWebDriver, AbstractEventListener
from pynput.mouse import Listener
import time
from queue import Queue
import uuid
from keywords_suggester.config import settings
from keywords_suggester.bin.modules.BertSingle import BertTopicClass
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from langchain.schema import Document
from datetime import datetime
from keywords_suggester.bin.modules.LoaderEmbeddings import SpliText
#docker-compose -f zk-single-kafka-single.yml up -d
#docker-compose -f zk-single-kafka-single.yml ps

# ==== INIT  SECTION =====

domain = "https://www.ansa.it/"

user = uuid.uuid4().hex[:5]
settings.init()
persist_directory = settings.persist_directory+settings.init_dataset+"/"
embed_model = settings.embed_model

collection_name_local = settings.collection_name
ChromaDB = ChromaClass(persist_directory,embed_model,collection_name_local)
BERT_NAME = settings.bert_name
BERT_MODEL = BertTopicClass(BERT_NAME,restore=1)


topic_info = BERT_MODEL.main_model.get_topic_info()    
dict_topic_name = dict(zip(topic_info['Topic'], topic_info['Name']))


# ==== DRIVER SELENIUM SECTION =====
b = webdriver.Firefox()
b.maximize_window()

class EventListeners(AbstractEventListener):
    def before_navigate_to(self, url, driver):
        print("before_navigate_to %s" % url)

    def after_navigate_to(self, url, driver):
        print("after_navigate_to %s" % url)

    def before_click(self, element, driver):
        print("before_click %s" % element)

    def after_click(self, element, driver):
        print("after_click %s" %element)

d = EventFiringWebDriver(b,EventListeners())

queue = Queue()
d.get(domain)
d.implicitly_wait(20)
d.back()



# ==== FAUST  SECTION =====
async def send_value(visited_url) -> None:
    content = await crawler.ask(Content(url=visited_url))
    
    print("CONTENT: ",content[0:50])
    
    created_at =  datetime.now()
    created_at_day = created_at.day
    created_at_month = created_at.month
    created_at_year = created_at.year

    topics, _ = BERT_MODEL.main_model.transform(content)
    max_topic = topics[0]
    #print(topics)
    #print(BERT_MODEL.main_model.get_topic_info(max_topic))
    topic_mapped = dict_topic_name[max_topic] 
    
    custom_metadata = {
        'category' : topic_mapped,
        'user' : user,
        'created_at_year' : created_at_year,
        'created_at_month':  created_at_month,
        'created_at_day': created_at_day ,
    }

    print(custom_metadata)

    text_splitter = SpliText()
    DOCS_TO_UPLOAD =  [Document(page_content=content,metadata=custom_metadata)]
    chunks = text_splitter.split_documents(DOCS_TO_UPLOAD)

    #add to chroma
    ChromaDB.AddDocsToCollection(chunks)


def on_click(x, y, button, pressed):
    if pressed:
        print('Mouse clicked')
        time.sleep(2)
        queue.put(b.current_url)
        
if __name__ == '__main__':
        
    listener = Listener(on_click=on_click)
    listener.start()

    while True:
        urls = queue.get()
        if(urls != domain):
            print("Navigation to: %s " % urls)

            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_value(urls))
       