from datasets import load_dataset,load_from_disk
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from transformers import AutoModel


#https://medium.com/llamaindex-blog/easily-finetune-llama-2-for-your-text-to-sql-applications-ecd53640e10d


#dataset = load_dataset("b-mc2/sql-create-context")
#dataset.save_to_disk("./resources/sqlContext")

#https://github.com/huggingface/autotrain-advanced

dataset = load_from_disk("./resources/sqlContext")

MODEL_PATH = "./CodeLlama-7b-Instruct"

model = AutoModel.from_pretrained(MODEL_PATH)


#print(dataset.data)
print(type(dataset.data))
print(dataset.column_names)

print(dataset['train'][0])


train_set = dataset['train']

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])




for sample in train_set:
    print(sample)
