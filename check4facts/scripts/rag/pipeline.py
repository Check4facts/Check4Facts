import os
import re
import ollama
from check4facts.scripts.rag.search_api import google_search
from check4facts.scripts.rag.harvester import *
from check4facts.scripts.rag.translate import *
from check4facts.scripts.rag.groq_api import *
from check4facts.scripts.rag.gemini_llm import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import time
import tiktoken
import numpy as np
os.environ["OLLAMA_MODE"] = "remote"


#app = Flask(__name__)

class pipeline:

    def __init__(self, query, n, model="gemma2:latest"):
        self.query = query
        self.n = n
        self.model = model
        self.harvested_urls= None

    #function to truncate info if the prompt token limit is exceeded (not used)
    def count_tokens(self, text, info):
        tokenizer = tiktoken.get_encoding(self.model)
        token_count = len(tokenizer.encode(text))
        if token_count > 8192:
            pass

    #method to remove 20% of text for cases when the input token limit is exceeded 
    def truncate_text(self, text):
        words = text.split()
        total_words = len(words)
        words_to_keep = int(total_words * 0.8)
        truncated_text = ' '.join(words[:words_to_keep])
        sentence_end = re.search(r'\.|\?|!$', truncated_text)  
        if sentence_end:
            truncated_text = truncated_text[:sentence_end.end()]
        
        return truncated_text


            


    

    #based on a claim, implement a searcher and a harvester 
    def retrieve_knowledge(self, max_sources):

        #scan the web for urls containing knowledge
        
        url_list = google_search(self.query, self.n + 3)
        if url_list is None:
            print('Could not find any results regarding the claim. Please try again or choose a different statement')
            return None
        else:
            #harvest the external urls using a harvester instance
            my_harvester = Harvester(list(url_list), self.query, timeout=1000, claim_id=0, max_sources = max_sources)
            df = my_harvester.run()
        

            #get the bodies of the top-n web sources that has the biggest "body_similarity" value
            try:
                result = df.nlargest(self.n, 'body_similarity')['similar_sentences'] 
                self.harvested_urls = df.nlargest(self.n, 'body_similarity')['url'].to_list()
            except Exception as e:
                print('Could not find relevant sources.')
                return None
            
        #print(result)
        return result
    
    def run_local_model(self, external_sources):
        #external_sources = self.retrieve_knowledge(max_sources)
        if(external_sources is not None):
            info = "\n\n".join(external_sources)

            print('\n')
            print('------------------Harvested URLs-------------------------')
            print(self.harvested_urls)
            print('----------------------------------------------------------')
            print('\n')
            print('------------------External knowledge----------------------')
            print(info)
            print('----------------------------------------------------------')

            input = f""" 
"Έχεις στη διάθεσή σου μια πληροφορία '{info}' και μία δήλωση: '{self.query}' της οποίας η ακρίβεια πρέπει "
    "να αξιολογηθεί. Χρησιμοποίησε μόνο την παρεχόμενη πληροφορία σε συνδυασμό με τις γνώσεις σου ώστε να αποφασίσεις "
    "εάν η δήλωση είναι ΑΛΗΘΗΣ, ΨΕΥΔΗΣ, ΜΕΡΙΚΩΣ-ΑΛΗΘΗΣ ή ΜΕΡΙΚΩΣ-ΨΕΥΔΗΣ.\n\n"
    "Πριν αποφασίσεις:\n\n"
    "1. Ανάλυσε με σαφήνεια τη δήλωση για να κατανοήσεις το περιεχόμενό της και να εντοπίσεις τα κύρια σημεία "
    "που πρέπει να αξιολογηθούν.\n"
    "2. Σύγκρινε τη δήλωση με την πληροφορία που έχεις, αξιολογώντας κάθε στοιχείο της δήλωσης ξεχωριστά.\n"
    "3. Χρησιμοποίησε τις γνώσεις σου ΜΟΝΟ σε συνδυασμό με την παρεχόμενη πληροφορία, αποφεύγοντας την αναφορά σε "
    "μη εξακριβωμένες πληροφορίες.\n\n"
    "Αποτέλεσμα: Δώσε μια ξεκάθαρη απάντηση επιλέγοντας μία από τις παρακάτω ετικέτες:\n\n"
    "- ΑΛΗΘΗΣ: Αν η δήλωση είναι απόλυτα επιβεβαιωμένη από την πληροφορία και τα στοιχεία σου.\n"
    "- ΨΕΥΔΗΣ: Αν η δήλωση διαψεύδεται ξεκάθαρα από την πληροφορία και τα στοιχεία σου.\n"
    "- ΜΕΡΙΚΩΣ-ΑΛΗΘΗΣ: Αν η δήλωση περιέχει κάποια σωστά στοιχεία, αλλά όχι απόλυτα ακριβή.\n"
    "- ΜΕΡΙΚΩΣ-ΨΕΥΔΗΣ: Αν η δήλωση περιέχει κάποια σωστά στοιχεία, αλλά περιέχει παραπλανητικές ή ανακριβείς πληροφορίες.\n\n"
    "Τέλος, εξήγησε τη λογική σου με σαφήνεια και επικεντρώσου στα δεδομένα που παρέχονται και στη δική σου γνώση. "
    "Απόφυγε περιττές λεπτομέρειες και προσπάθησε να είσαι ακριβής και περιεκτικός στην ανάλυσή σου."
    "Οι απαντήσεις σου πρέπει να έχουν την μορφή:"
            "Δήλωση: '{self.query}'"
            "Αποτέλεσμα δήλωσης:" 
            "Δικαιολόγηση:"
    """


            
        else:


            input = f""" 
                Πάρε μία βαθιά ανάσα και έλενξε παρακάτω δήλωση: '{self.query}' 
                Πες μου αν η δήλωση είναι αληθής, ψευδής, εν-μερει αληθής, η εν-μέρει ψευδής
                Δείξε μου διάφορες πηγές που δικαιολογούν την απάντησή σου.
                Οι απαντήσεις σου πρέπει να έχουν την μορφή:

                Δήλωση: '{self.query}'
                Αποτέλεσμα δήλωσης: 
                Πηγές που υποστηρίζουν αυτήν την αξιολόγηση:

                
                """

        
        while True:
            try:
                response = ollama.chat(model=self.model, messages=[{"role": "user", "content": translate_long_text(input, src_lang='el', target_lang='en')}])
                break
            except Exception as e:
                print(e)
                print("Retrying with smaller input....")
                info = self.truncate_text(info)
            
        answer = response['message']['content']
        answer = translate_long_text(answer, src_lang='en', target_lang='el')
        # print()
        print('----------------ANSWER----------------')
        print(answer)

        # Output the response from the model
        return answer
    
    def create_api(self, external_sources):
        #external_sources = self.retrieve_knowledge(max_sources)
        
        if(external_sources is not None):
            info = "\n\n".join(external_sources)

            print('\n')
            print('------------------Harvested URLs----------------------')
            print(self.harvested_urls)
            print('----------------------------------------------------------')


            print('\n')
            print('------------------External knowledge----------------------')
            print(info)
            print('----------------------------------------------------------')
        else: 
            print('No external sources found')
            info = None
        api = groq_api(info, self.query)

        response = api.run_api()
        
        return response
    

    def run_gemini(self, info, article_id):

        info = "\n\n".join(info)
        print('\n')
        print('------------------Harvested URLs----------------------')
        print(self.harvested_urls)
        print('----------------------------------------------------------')
        print('\n')
        print('------------------External knowledge----------------------')
        print(info)
        print('----------------------------------------------------------')


        gemini_instance  = gemini_llm(query=self.query, external_knowledge=info)
        answer = gemini_instance.google_llm(article_id)
        return answer



def run_pipeline(article_id, claim, num_of_web_sources):
    start_time = time.time()  

    if not isinstance(claim, str) or not isinstance(num_of_web_sources, int):
        print("Either claim is not a string or num_of_web_sources is not an integer.")
        return None
    


    pip = pipeline(str(claim) , int(num_of_web_sources))
    external_sources = pip.retrieve_knowledge(int(num_of_web_sources)+2)

    #make the model run on groq api
    groq_response = pip.create_api(external_sources)
    if groq_response:
        end_time = time.time()
        label_match = re.search(r"(?i)Result of the statement:\s*(.*?)\s*justification:", str(groq_response), re.DOTALL)
        label = label_match.group(1) if label_match else None
        justification_match = re.search(r"Justification:\s*(.*)", str(groq_response), re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else None
        groq_response['sources'] = pip.harvested_urls
        groq_response['label'] = re.sub(r"\s+$", "", label)
        groq_response['label'] = str(groq_response['label']).strip()
        label = re.sub(r"\s+$", "", label)
        groq_response['justification'] = justification
        groq_response['elapsed time'] = np.round((end_time-start_time) , 2)
        groq_response["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return groq_response

        
    #if the connections fails to be established or the response is empty, invoke the gemini LLM.
    gemini_response = pip.run_gemini(external_sources, article_id)
    # print('ANSWER IS: ')
    # print(gemini_response['response'])
    # print(gemini_response)
    if gemini_response:
        
        end_time = time.time()
        label_match = re.search(r"(?i)Result of the statement:\s*(.*?)\s*justification:", str(gemini_response), re.DOTALL)
        label = label_match.group(1) if label_match else None
        justification_match = re.search(r"Justification:\s*(.*)", str(gemini_response), re.DOTALL)
        justification = justification_match.group(1).strip() if justification_match else None
        gemini_response['sources'] = pip.harvested_urls
        gemini_response['label'] = re.sub(r"\s+$", "", label)
        gemini_response['justification'] = justification
        gemini_response['elapsed time'] = np.round((end_time-start_time) , 2)
        gemini_response["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return gemini_response


    #if the connections fails to be established or the response is empty, invoke the local LLM.
    print('Running local ollama model...')
    #run the model with ollama, max sources are the number of web sources plus 2
    answer =  pip.run_local_model(external_sources)
    if answer:
        end_time = time.time()
        answer['sources'] = pip.harvested_urls
        answer['elapsed time'] = np.round((end_time-start_time) , 2)
        answer["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        return answer  
    

    return {"error": "All llm invokation attempts failed"} 
        
    