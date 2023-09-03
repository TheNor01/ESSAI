

import requests
import openai

openai.api_key = "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"


response = requests.get('https://localhost:9200/test/_mapping', verify="./docker-compose/ca/ca.crt", auth=('elastic', 'Pw_CLzJpkoCctLJl6-bV'))

data = response.json()
print(data)


ELASTIC_PASSWORD="Pw_CLzJpkoCctLJl6-bV"
from elasticsearch import Elasticsearch

client = Elasticsearch(
    "https://localhost:9200",
    ca_certs="./docker-compose/ca/ca.crt",
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# Successful response!
print(client.info())


def chat_with_chatgpt(prompt, model="gpt-3.5-turbo"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message

#user_prompt = "Given this elastic mapping" + str(data) + " return the first 10 documents where field1 is equal to Ferrari as DSL querys"
#chatbot_response = chat_with_chatgpt(user_prompt)
#print(chatbot_response)