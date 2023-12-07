from flask import Flask, render_template, request
import pandas as pd
from keywords_suggester.bin.modules.ChromaSingle import ChromaClass
from langchain.embeddings import HuggingFaceEmbeddings



app = Flask(__name__,template_folder='D:/ALDO/Studio/Uni/ESSAI/interface/templates')

@app.route('/')
def index():
    return render_template('menu.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Nessun file fornito"

        file = request.files['file']

        if file.filename == '':
            return "Nome file vuoto"

        if file:
            df = pd.read_csv(file)
            print(df.head())

            return "File caricato con successo"

    return render_template('upload.html')

@app.route('/create_chromadb', methods=['GET', 'POST'])
def create_chromadb():
    if request.method == 'POST':
        # Ottieni i dati dal modulo del form
        # Esegui le azioni necessarie per creare un oggetto ChromaDB

        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        persist_directory = "keywords_suggester/index_storage_lang"

        ChromaDB = ChromaClass(persist_directory,embed_model)

        return "ChromaDB creato con successo a" + ChromaDB.persist_directory

    return render_template('create_chromadb.html')

if __name__ == '__main__':
    app.run(debug=True)


