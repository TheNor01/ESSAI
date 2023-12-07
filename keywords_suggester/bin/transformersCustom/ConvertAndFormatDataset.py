import os
import csv
import uuid
from datetime import datetime
from keywords_suggester.bin.modules.Cleaner import clean_text

# Funzione per convertire un file di testo in un file CSV
def convert_to_csv(input_file_path, output_file_path,headers):

    with open(input_file_path, 'r',encoding="utf-8") as infile, open(output_file_path, 'w', newline='\n',encoding="utf-8") as outfile:
        lines = infile.readlines()
        content = ('\t'.join([line.strip() for line in lines])).replace('|','')

        #clean text
        content = clean_text(content)
        csv_writer = csv.writer(outfile,escapechar='\\',delimiter='|',quoting=csv.QUOTE_MINIMAL,quotechar='"')

        csv_writer.writerow(headers)
        folder_metadata = input_file_path.split('/')[-2]

        created_at_string = datetime.now().strftime("%Y-%m-%d")

        #truncate text to limit 

        user = uuid.uuid4().hex[:5]
        data = [content.strip(),user,folder_metadata,created_at_string] #created at in order to delete FROM DB, cleaning action with delete
        csv_writer.writerow(data)

# Funzione per elaborare ricorsivamente una directory
def process_directory(source_dir, destination_dir,headers):

    print("PROCESSING dataset :"+ source_dir)

    for root, _, files in os.walk(source_dir):
        for file_name in files:
            # Crea la struttura della cartella di destinazione se non esiste
            dest_folder = os.path.join(destination_dir, os.path.relpath(root, source_dir))
            os.makedirs(dest_folder, exist_ok=True)
            
            # Costruisci i percorsi completi per i file sorgenti e di destinazione
            source_file_path = os.path.join(root, file_name)
            dest_file_path = os.path.join(dest_folder, file_name.replace('.txt', '.csv'))
            
            # Converte il file di testo in un file CSV
            convert_to_csv(source_file_path, dest_file_path,headers)


