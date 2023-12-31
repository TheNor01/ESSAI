
import random

class Headers:
    def __init__(self,data_file="./keywords_suggester/storage/headers/headers.txt"):
        self.data_file=data_file
        self.file = None

    def __getAgents(self):
        self.file=open(self.data_file,"r")
        agents = (self.file).read()
        agents_list = agents.splitlines()

        return agents_list
        
    def __getHeader(self):
        
        agents_list=self.__getAgents()
        user_agent = random.choice(agents_list)
        return user_agent
   
    def getUserAgent(self):
        
        agents_list=self.__getAgents()
        user_agent = random.choice(agents_list)
        return user_agent
    
    def setHeader(self):
        user_agent = self.__getHeader()


        headers = {
            'User-Agent': user_agent,
            'Upgrade-Insecure-Requests': '1',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'DNT' : '1',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'it-IT',
            'Cookie':'CONSENT=YES+cb.20231214-17-p0.it+FX+917; '
            #'Cookie':"OTZ:5946257_48_52_123900_48_436380"
            }

            #,'referer':'https://www.google.com/'#QUI
        self.file.close()
        return headers
   
"""
if __name__ == "__main__":

    obj = Headers()
    obj.setHeader()
"""