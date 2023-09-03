# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, pipelinepy

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain


#pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

text = """   
can you generate a dsl language given a mapping index
"""


base_model = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
    load_in_8bit=True,
    device_map='auto',
)




tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=500,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """
Write a SQL Query given the table name {Table} and columns as a list {Columns} for the given question : 
{question}.
"""

prompt = PromptTemplate(template=template, input_variables=["Table","question","Columns"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

def get_llm_response(tble,question,cols):
    llm_chain = LLMChain(prompt=prompt, 
                         llm=local_llm
                         )
    response= llm_chain.run({"Table" : tble,"question" :question, "Columns" : cols})
    print(response)

tble = "employee"
cols = ["id","name","date_of_birth","band","manager_id"]
question = "Query the count of employees in band L6 with 239045 as the manager ID"
get_llm_response(tble,question,cols)