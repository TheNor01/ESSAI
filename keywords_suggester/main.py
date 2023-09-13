#https://gpt-index.readthedocs.io/en/latest/getting_started/concepts.html#retrieval-augmented-generation-rag

from llama_index import SimpleDirectoryReader,Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter
from llama_index import VectorStoreIndex


required_exts = [".txt"]
filename_fn = lambda filename: {'file_name': filename} #metadata

# Load in Documents
documents = SimpleDirectoryReader('./keywords_suggester/data/dataset/automotive/',required_exts=required_exts,recursive=True,file_metadata=filename_fn).load_data()
print(f"Loaded {len(documents)} docs")

# Parse the Documents into Nodes
#You can choose to define Nodes and all its attributes directly
#You may also choose to “parse” source Documents into Nodes through our NodeParser classes


text_splitter = SentenceSplitter(
  separator=" ",
  chunk_size=1024,
  chunk_overlap=20,
)

parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
nodes = parser.get_nodes_from_documents(documents)

print(f"Parsed {len(nodes)} docs")

#An Index is a data structure that allows us to quickly retrieve relevant context for a user query. For LlamaIndex, it’s the core foundation for retrieval-augmented generation (RAG) use-cases.
index = VectorStoreIndex.from_documents(nodes)

#Resuse nodes: Reusing Nodes across Index Structures


