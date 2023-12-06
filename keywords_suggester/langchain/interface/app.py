from flask import Flask, render_template, request
import pandas as pd


app = Flask(__name__)

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
        return "ChromaDB creato con successo"

    return render_template('create_chromadb.html')

if __name__ == '__main__':
    app.run(debug=True)


