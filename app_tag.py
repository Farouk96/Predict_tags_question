import os
!pip install nltk
import streamlit as st
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# preprocessing functions
def clean_text(text):
      text = text.lower()
      text = re.sub(r"what's", "what is ", text)
      text = re.sub(r"\'s", " ", text)
      text = re.sub(r"\'ve", " have ", text)
      text = re.sub(r"can't", "can not ", text)
      text = re.sub(r"n't", " not ", text)
      text = re.sub(r"i'm", "i am ", text)
      text = re.sub(r"\'re", " are ", text)
      text = re.sub(r"\'d", " would ", text)
      text = re.sub(r"\'ll", " will ", text)
      text = re.sub(r"\'scuse", " excuse ", text)
      text = re.sub(r"\'\n", " ", text)
      text = re.sub(r"\'\xa0", " ", text)
      text = re.sub('\s+', ' ', text)
      text= re.sub('nan',' ',text)
      text= re.sub('null',' ',text)
      text= re.sub('func',' ',text)
      text= re.sub(r'[0-9]', ' ', text) # remove numbers
      #text= re.sub(r'(?:^| )\w(?:$| )', ' ', text)
      text = text.strip(' ')
      return text

token=ToktokTokenizer()
punct = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
tags_features=['<python',
 '<javascript',
 '<java',
 '<reactjs',
 '<html',
 '<r',
 '<c#',
 '<android',
 '<python-3.x',
 '<pandas',
 '<node.js',
 '<sql',
 '<php',
 '<css',
 '<c++',
 '<flutter',
 '<arrays',
 '<c',
 '<django',
 '<angular',
 '<mysql',
 '<dataframe',
 '<typescript',
 '<jquery',
 '<swift',
 '<json',
 '<laravel',
 '<vue.js',
 '<ios',
 '<firebase',
 '<amazon-web-services',
 '<react-native',
 '<dart',
 '<postgresql',
 '<kotlin',
 '<azure',
 '<excel',
 '<numpy',
 '<spring-boot',
 '<sql-server',
 '<list',
 '<mongodb',
 '<docker',
 '<tensorflow',
 '<regex',
 '<spring',
 '<api',
 '<asp.net-core',
 '<oracle',
 '<vba',
 '<linux',
 '<string',
 '<swiftui',
 '<android-studio',
 '<loops',
 '<git',
 '<matplotlib',
 '<express',
 '<powershell',
 '<bash',
 '<selenium',
 '<wordpress',
 '<kubernetes',
 '<.net',
 '<ggplot2',
 '<database',
 '<algorithm',
 '<ruby-on-rails',
 '<function',
'<apache-spark',
 '<keras',
 '<web-scraping',
 '<dictionary',
 '<google-cloud-firestore',
 '<ruby',
 '<visual-studio-code',
 '<machine-learning',
 '<discord',
 '<pyspark',
 '<csv',
 '<visual-studio',
 '<ajax',
 '<for-loop',
 '<azure-devops',
 '<xcode',
 '<google-sheets',
 '<tkinter',
 '<macos',
 '<scala',
 '<if-statement',
 '<.net-core',
 '<react-hooks',
 '<windows',
 '<xml',
 '<elasticsearch',
 '<dplyr',
 '<discord.py',
 '<mongoose',
 '<bootstrap-4',
 '<opencv']
def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']
def clean_punct(text): 
    words=token.tokenize(text)
    punctuation_filtered = []
    regex = re.compile('[%s]' % re.escape(punct))
    remove_punctuation = str.maketrans(' ', ' ', punct)
    for w in words:
        if w in tags_features:
            punctuation_filtered.append(w)
        else:
            punctuation_filtered.append(regex.sub('', w))
  
    filtered_list = strip_list_noempty(punctuation_filtered)
        
    return ' '.join(map(str, filtered_list))

lemma=WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
    
    stop_words = set(stopwords.words("english"))
    
    words=token.tokenize(text)
    
    filtered = [w for w in words if not w in stop_words]
    
    return ' '.join(map(str, filtered))
####################################################################
st.title('Question_tags_suggestion')
title = st.text_input('Le titre')
qst= st.text_area('Posez votre question')

if title=='':
  st.write("Svp,ecrivez le titre.")
elif qst=='':
  st.write("Svp,ecrivez la question.")
else:
  #preprocessing a title & text
  lst=[title,qst]
  for j in lst:
    j = str(j)
    j = clean_text(j) 
    j = clean_punct(j) 
    j = lemitizeWords(j) 
    j = stopWordsRemove(j)
  #Download a vectorizer   
  with open('/content/drive/MyDrive/ P5_HADJ NACER_Farouk/Vectorizer question1','rb') as S:
    vectorizer1=pickle.load(S)
  with open('/content/drive/MyDrive/ P5_HADJ NACER_Farouk/Vectorizer question2','rb') as T:
    vectorizer2=pickle.load(T)   
  title=[title]
  qst=[qst]
  title= vectorizer2.transform(title) 
  qst= vectorizer1.transform(qst)
  X= hstack([qst,title])

  #Download a best model
  with open('/content/drive/MyDrive/ P5_HADJ NACER_Farouk/Best tager questions','rb') as f:
    model =pickle.load(f)
  st.subheader('Tags:')
  #Download a multi_label
  with open('/content/drive/MyDrive/ P5_HADJ NACER_Farouk/Multi_label','rb') as H:
    multi_label =pickle.load(H)
  pred =list(model.predict(X))
  R= 159-len(pred)
  for i in range(R):
    pred.append(0)
  pred= np.array(pred)
  #pred.reshape(pred.shape[1],1)
  st.write(multi_label.inverse_transform(pred))
  
