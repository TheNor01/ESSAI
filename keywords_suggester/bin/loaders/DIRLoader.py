import os

from langchain.document_loaders import TextLoader
from keywords_suggester.bin.loaders.CSVLoader import CSVLoader

class DIRLoader:

    def __init__(self, dir_path, **kwargs):
        self.dir_path = dir_path
        self.kwargs = kwargs

    def load(self):
        docs = []
        for root, _, files in os.walk(self.dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path.endswith('.csv'):
                    loader = CSVLoader(file_path, encoding='utf-8',**self.kwargs)
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8',**self.kwargs)
                else:
                    print(f"Do not process the file: {file_path}")
                    continue
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
        return docs