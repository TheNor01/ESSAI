from llama_index import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="keywords_suggester/index_storage")

# load index, need service context
index = load_index_from_storage(storage_context)