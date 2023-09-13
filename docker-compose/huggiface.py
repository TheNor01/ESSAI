# Use a pipeline as a high-level helper
from transformers import pipeline

from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from datasets import load_dataset,load_from_disk

from transformers import AutoTokenizer

import transformers
import torch

modelName = "codellama/CodeLlama-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(modelName)
model= LlamaForCausalLM.from_pretrained(modelName)

#model.save_pretrained('./codeLlama7b')


exit()

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float32,
    device_map="auto",
)

sequences = pipeline(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")




exit()


dataset = load_from_disk("./resources/sqlContext")


def tokenize_function(sample):
    return tokenizer(sample["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


