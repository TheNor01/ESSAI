
### MODULES
from bs4 import BeautifulSoup
from essai.bin.classi.ChangeHeaders import Headers

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import requests
import spacy
import os




def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class ScrapingHTML:
    
    def __init__(self,language):
        self.__headersObj = Headers()
        self.headers=None
        self.lang = language
        options = FirefoxOptions()
        options.add_argument("--headless")
        self.browser = webdriver.Firefox(options=options)
        self.maxTimeoutScraping = 30
        #self.News = GoogleScraping(period) passato tramite parametri
        if language=="english":
            self.nlp = spacy.load("en_core_web_lg",disable=['ner','lemmatizer'])

#----------------------------------------------------------
    def StartRequest(self,links):  
        self.headers = self.__headersObj.setHeader()  #Setta l'Header
        output = self.GetText(links)
        return output
#----------------------------------------------------------
    def ApplyFiltro(self,text):  #Regex per eliminare spazzatura
        if len(text) >=35:
            matches=re.findall('[.\s|[cC]ookie[s]*|©|@|[Cc]opyright|http[s]*:',text)
            if(len(matches) >=1 or text[0].isdigit()):
                return False
            else:
                return True
        else:
            return False
#----------------------------------------------------------               
    def DecomposeElementTrash(self, soup):
        try:
            soup.find('footer').decompose()
        except:
            pass
        for item in soup.select('div[class*="hidden"]'):
            item.decompose()
#----------------------------------------------------------
    def NplFilter(self,stringDoc):
        doc=self.nlp(stringDoc)
        stringReturn=""
        for w in doc:
            if w.pos_ in ("NOUN","ADJ","VERB","PROPN"):
                stringReturn+=w.text+" "
        stringReturn=stringReturn[:-1]
        return stringReturn 
#----------------------------------------------------------
    def WriteTitleDescKeyIntoFile(self, soup):

        local_str = ""
        #extract from meta: title,description,keywords
        #title
        try:
            title= soup.find('title')
            contentTitle=title.get_text().strip()
            
            Cleaned_contentTitle = self.NplFilter(contentTitle) #Applico la NLP ad ognuno
            local_str=local_str+"Titolo: "+ Cleaned_contentTitle+"\n"
            #file.write("Titolo: "+ Cleaned_contentTitle+"\n")
        except:
            local_str=local_str+("Titolo: none\n")

        #keywords
        keywordWords=[]
        try:
            keywordSite= soup.find('meta',attrs={'name':'keywords'}).attrs['content']
            contentKeywordSite=keywordSite.strip()
            keywordWords= contentKeywordSite.replace(","," ").replace("  "," ").split(' ')
            keywordWrite=set(keywordWords)
            #file.write("Keywords: ")
            local_str=local_str+"Keywords: "
            for key in keywordWrite:
                #file.write(key + " ")
                local_str=local_str+(key + " ")
            #file.write("\n")
        except:
            #file.write("Keywords: none\n")
            local_str=local_str+"Keywords: none\n"

        #description
        try:
            descrizione= soup.find('meta',attrs={'name':'description'}).attrs['content']
            contentDesc=descrizione.strip()
            Cleaned_contentDesc = self.NplFilter(contentDesc)
            #file.write("Descrizione: "+Cleaned_contentDesc+"\n")
            local_str=local_str+"Descrizione: "+Cleaned_contentDesc+"\n"

        except:
            try:
                descrizione= soup.find('meta',attrs={'property':'og:description'}).attrs['content']
                contentDesc=descrizione.strip()
                Cleaned_contentDesc = self.NplFilter(contentDesc)
                #file.write("Descrizione: "+Cleaned_contentDesc+"\n")
                local_str=local_str+("Descrizione: "+Cleaned_contentDesc+"\n")
            except:
                #file.write("Descrizione: none\n")
                local_str=local_str+("Descrizione: none\n")

        return local_str
#----------------------------------------------------------
    def GetCorrectLink(self,url):
        #necessary in case of cookie form
        try:
            req = requests.get(url,headers=self.headers, timeout=self.maxTimeoutScraping)
            soup = BeautifulSoup(req.content, 'html.parser')
            try:
                div=soup.find("div",{"class":"m2L3rb"})
                urlReturn=div.find("a")["href"]
            except:
                urlReturn=url
            return urlReturn
        except:
            return None
#----------------------------------------------------------
        
    def GetCorrectLinkUsingSelenium(self,url):
    #necessary in case of cookie form ----
        self.browser.get(url)

        try:
            WebDriverWait(self.browser, 3).until(EC.element_to_be_clickable((By.XPATH,'//*[@aria-label="Accetta tutto"]'))).click()
            print("accepted cookies")
            url = self.browser.current_url
            html = self.browser.page_source
            print(url)
        except Exception as e:
            url = self.browser.current_url
            html = self.browser.page_source
            print('no cookie button')
            try:
                #req = requests.get(url,headers=self.headers, timeout=self.maxTimeoutScraping)
                soup = BeautifulSoup(html, 'html.parser')
                try:
                    div=soup.find("div",{"class":"m2L3rb"})
                    urlReturn=div.find("a")["href"]
                except:
                    urlReturn=url
                return urlReturn
            except:
                return None

#----------------------------------------------------------
    def GetText(self,links) -> str:
        for index in range(len(links)):
            print("----------------------------------------------------------------")

            #Start scraping
            url=links[index]
            url=self.GetCorrectLinkUsingSelenium(url)
            
            if url == None:
                continue

            try:
                req = requests.get(url,headers=self.headers, timeout=self.maxTimeoutScraping)
            except :
                print("Expect Request URL!")
                continue

            if req.status_code!=200:
                print("code status error 200!")
                continue

            try:
                soup = BeautifulSoup(req.content, 'html.parser') # parser #,from_encoding="iso-8859-1"

            except :
                try:
                    soup = BeautifulSoup(req.content.decode('utf-8','ignore'), 'html.parser') # parser #,from_encoding="iso-8859-1"
                except:
                    print("Expect BeautifulSoap error")
                    continue
                        
            self.DecomposeElementTrash(soup)

            #Estrazione titolo descrizione e keywords
            local_str = self.WriteTitleDescKeyIntoFile(soup)

            #Estrazione contenuto
            tags = soup.find_all('p',class_=False)
            linesCounter=0
            
            s = ""
            for tag in tags:
                content=tag.get_text().strip()
                if( self.ApplyFiltro(content)==True ):  ##prendo il P
                    #print("CONTENT= ",content)
                    s = s + content
                    linesCounter=linesCounter+1

            local_str = local_str + s


            #discard if: file is empty; file size is over 20kb; file contains more than 100 lines; file don't contains text
            """
            if os.stat(pathname).st_size == 0 or os.stat(pathname).st_size > 20000 or control_if_Empty==True or linesCounter >=100: 
                print("-------------------------------")
                print("Rimuovo file: "+pathname)
                os.remove(pathname) 
                print("-------------------------------")
            

            else: 
                print("Salvo il file: "+pathname)
                savedFiles=savedFiles+1
            """
            return local_str
            #we want reach full articles for path
            #if savedFiles >= self.articoli: 
            #    print("Limite aricoli raggiunto")
            #   return
#----------------------------------------------------------