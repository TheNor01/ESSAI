# Use a pipeline as a high-level helper
from transformers import pipeline

from transformers import LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from datasets import load_dataset,load_from_disk

import transformers
import torch

tokenizer = LlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")




exit()


dataset = load_from_disk("./resources/sqlContext")


def tokenize_function(sample):
    return tokenizer(sample["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


