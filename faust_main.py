import faust
from keywords_suggester.config import settings



settings.init()
app = faust.App(settings.collection_name, 
                broker='kafka://localhost:9092', 
                #store='rocksdb://'
                store='memory://'                    
                )
    
content_topic = app.topic('content')

#content_views = app.Table('content_views', default=int)

@app.agent(content_topic)
async def greet(traffic):
    async for content in traffic:
        print(content)


if __name__ == '__main__':
    app.main()