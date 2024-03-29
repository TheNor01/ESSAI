import os
import csv
from datetime import datetime
from essai.bin.modules.Cleaner import clean_text
from essai.config import settings
import hashlib
import pandas as pd
import random


user_data_file="./essai/storage/users/population.txt"

population = None
with open(user_data_file, "r") as population:
        users = [line.rstrip() for line in population]
        population = users

def build_dataframe_from_csv_uploaded(BERT_MODEL,SELECTED_UPLOAD):

    topic_info = BERT_MODEL.main_model.get_topic_info()
    
    dict_topic_name = dict(zip(topic_info['Topic'], topic_info['Name']))

    documents_list,topics_list,users_list,ids_list = [],[],[],[]
    created_at= datetime.now()
    created_at_day = created_at.day
    created_at_month = created_at.month
    created_at_year = created_at.year

    USE_BERT = 1
    with open(os.path.join(settings.upload_directory,SELECTED_UPLOAD),encoding="utf-8") as file_obj: 
        reader_obj = csv.reader(file_obj,delimiter="|") 

        #next(reader_obj, None) # SKIP HEADERS
        for count,row in enumerate(reader_obj): 
            
            #CHECK HEADER
            if(count==0): #FIRST ITER CHECK
                if not "CATEGORY".lower() in (item.lower() for item in row):
                    print("CATEGORY NOT DETECTED -> Using bertopic")
                else:
                    USE_BERT = 0
                    print("CATEGORY DETECTED")
                continue #next(reader_obj, None) # SKIP HEADERS
            
            local_doc = row[1]
            ids_list.append(hashlib.md5(local_doc.encode()).hexdigest()) #in realtà crea lui ID, posso togliere
            documents_list.append(local_doc)
            users_list.append(row[0])
            if(USE_BERT==1):
                topics, probs = BERT_MODEL.main_model.transform(local_doc)
                max_topic = topics[0]
                #print(topics)
                #print(BERT_MODEL.main_model.get_topic_info(max_topic))
                topic_mapped = dict_topic_name[max_topic] 
                topics_list.append(topic_mapped)

            else:
                #Assume Category is 3rd column mapped
                local_category = row[2]
                topics_list.append(local_category)

    upload_df = pd.DataFrame(zip(documents_list, topics_list, users_list,ids_list),columns=['content','category', 'user','ids'])
    upload_df['created_at_year']=created_at_year
    upload_df['created_at_month']=created_at_month
    upload_df['created_at_day']=created_at_day


    return upload_df




# Funzione per convertire un file di testo in un file CSV
def convert_to_csv(input_file_path, output_file_path,headers):

    input_file_path = input_file_path.replace("\\","/")

    #TODO USER deve venire dal file, non generato --> No perchè qua siamo nella fase di INIT, non necessario
    with open(input_file_path, 'r',encoding="utf-8",errors='ignore') as infile, open(output_file_path, 'w', newline='\n',encoding="utf-8") as outfile:
        lines = infile.readlines()
        content = ('\t'.join([line.strip() for line in lines])).replace('|','')

        #clean text
        content = clean_text(content)
        csv_writer = csv.writer(outfile,escapechar='\\',delimiter='|',quoting=csv.QUOTE_MINIMAL,quotechar='"')

        csv_writer.writerow(headers)
        folder_metadata = input_file_path.split('/')[-2] #TO FIX

        created_at= datetime.now()
        created_at_day = created_at.day
        created_at_month = created_at.month
        created_at_year = created_at.year

        user = random.choice(population)
        data = [content.strip(),user,folder_metadata,created_at_year,created_at_month,created_at_day] #created at in order to delete FROM DB, cleaning action with delete
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


