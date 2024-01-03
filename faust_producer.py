import faust
from keywords_suggester.bin.modules.ScrapingSingle import ScrapingHTML
import uuid

#docker-compose -f zk-single-kafka-single.yml up -d
#docker-compose -f zk-single-kafka-single.yml ps

class Content(faust.Record):
    url: str

app = faust.App(
    'content-app',
    broker='kafka://localhost:9092',
    value_type=Content
)
content_topic = app.topic('content')



Scraping = ScrapingHTML("2023-31-12","english")

@app.agent(content_topic)
async def crawler(content_traffic):
    async for Content in content_traffic:
        print("===================")
        print(Content.url)
        decoded_url= Content.url
        #print(decoded_url)
        user = uuid.uuid4().hex[:5]
        links = [decoded_url]
        content = Scraping.StartRequest(links)
        yield content
        print("===================")