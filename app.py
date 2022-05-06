import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import os
import requests
import openai
from cxrPrediction import run_model
from keras.models import load_model
import pandas as pd
import numpy as np
import os
import geocoder
import pickle
import warnings
from googletrans import Translator
import spacy
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#from transformers import AutoModelForCausalLM, AutoTokenizer
import random
#from model_generator import Generator
import pyttsx3

#from flask_ngrok import run_with_ngrok
#from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)
app.static_folder = 'static'
#run_with_ngrok(app)
#model and tokenizer initialization through HuggingFace
# tokenizer = AutoTokenizer.from_pretrained('Models/epochs_4/')
# model = AutoModelForCausalLM.from_pretrained('Models/epochs_4/')
# special_token = '<|endoftext|>'
translater = Translator()

# bot = Engine()
run_model = run_model()

def reply_api(userinput):
    try:
        r = requests.post(url='', json={"data": [userinput]})
        d= r.json()['data']
        return str(*d)
    except :
        return "Sorry I dont Know"

@app.route('/tts')
def tts(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    return 1
    

def docInfo():
    df=pd.read_csv("Dataset/doctor_data.csv")
    a=random.randint(0,len(df.index)-1)
    return str(df.loc[a].to_string(index = False))

def docotorNearMe():
    g = geocoder.ip('me')
    return f'https://www.google.com/maps/search/find+doctor+near+me/@{g.lat},{g.lng},14z'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    text = request.args.get('msg').strip()
    text=translater.translate(text,dest="en").text
    reply = reply_api(text.strip())
    openai.api_key = "apikey"
    response = openai.Completion.create(
    engine="text-davinci-001",
    prompt=reply,
    temperature=0,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    #return response['choices'][0]['text']
    reply=response['choices'][0]['text']
    #return reply
    #print(reply)
    # r = requests.post(url='https://hf.space/embed/jsylee/adverse-drug-reactions-ner/+/api/predict/', json={"data": [reply]})
    # r.json()
    # data=r.json()['data']
    # reply=f'''{data}'''
    # exp="['']"
    # for char in exp:
    #     reply=reply.replace(char,"")
    #blob = response['choices'][0]['text'] + "<br><br> <b>Note: Not satisy with the answer. Please enter your query again.</b><br><br>"+ docInfo()
    # with open('logs.txt', 'a', encoding = "UTF-8") as f:
    #     f.write(f'User: {text}\nDoctoBot: {response['choices'][0]['text']}\n')
    temp='''
    <div class="mapouter"><div class="gmap_canvas"><iframe width="220" height="400" id="gmap_canvas" src="https://maps.google.com/maps?q=doctor%20near%20me&t=k&z=13&ie=UTF8&iwloc=&output=embed" frameborder="0" scrolling="no" marginheight="0" marginwidth="0"></iframe><style>.mapouter{text-align:left;height:437px;width:528px;}</style><style>.gmap_canvas {overflow:hidden;background:none!important;height:437px;width:528px;}</style></div></div>
    '''
    lk=docotorNearMe()
    temp2='''
    <a href="lk"></a>
    '''
    blob = reply + "<br><br> <b>Note: Not satisy with the answer. Please enter your query again.</b><br><br>"+"Contact:<br>"+ docInfo()+"<br>"+temp
    #tts(reply)
    #return reply
    return blob

@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        # Getting the file from post request
        f = request.files['file']

        # Saving the file to ./uploads
        basepath = os.path.dirname(__file__)

        current_directory = os.getcwd()
        final_directory = os.path.join(current_directory, r'uploads')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)

        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        result = run_model.model_predict(img_path=file_path)

        return str(result)

    return None


if __name__ == "__main__":
    app.run(debug=True)
