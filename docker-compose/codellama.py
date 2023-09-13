

from ctransformers import AutoModelForCausalLM,AutoTokenizer
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import CTransformers

"""
GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""

configDict = {
    "temperature" : 0.8,
    "top_p" : 0.9,
    "top_k": 30,
    "repetition_penalty" : 1.1,
    "max_new_tokens" :2048,
    #"gpu_layers" : 30,
    "repetition_penalty" : 1.15
}

#llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id="TheBloke/CodeLlama-7B-Instruct-GGUF", model_type="llama",hf=True)
#llm = CTransformers(model="resources/codellama-7b-instruct.ggmlv3.Q6_K.bin",model_type="llama",config=configDict)
llm = CTransformers(model="resources/codellama-7b-instruct.Q5_K_M.gguf",model_type="llama",config=configDict)


#tokenizer = AutoTokenizer.from_pretrained(llm)


template = """[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:{prompt}[/INST]"""

prompt = PromptTemplate(template=template, input_variables=["prompt"])

qa_chain = LLMChain(prompt=prompt,llm=llm)

#pipe = pipeline("text-generation", model=llm, tokenizer=tokenizer)
#print(pipe("Current state of deep learning and its potential impact on various industries", max_new_tokens=256))

response = qa_chain("Genere a java code that prints numbers from 1 to 100")["text"]
print(repr(response))