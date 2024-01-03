import asyncio
from faust_producer import Content, crawler
from selenium import webdriver
from selenium.webdriver.support.events import EventFiringWebDriver, AbstractEventListener
from pynput.mouse import Listener
import time
from queue import Queue

#docker-compose -f zk-single-kafka-single.yml up -d
#docker-compose -f zk-single-kafka-single.yml ps

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
d.get('https://www.ansa.it')
d.implicitly_wait(20)
d.back()


async def send_value(visited_url) -> None:
    content = await crawler.ask(Content(url=visited_url))
    print("CONTENT: ",content)


    #add to chroma

def on_click(x, y, button, pressed):
    if pressed:
        print('Mouse clicked')
        time.sleep(2)
        #print("Navigation to: %s " % b.current_url)
        queue.put(b.current_url)
        


if __name__ == '__main__':
    #with Listener(on_click=on_click) as listener:
    #    listener.start()
        
    listener = Listener(on_click=on_click)
    listener.start()

    while True:
        urls = queue.get()
        if(urls != "https://www.ansa.it"):
            print("Navigation to: %s " % urls)

            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_value(urls))
       