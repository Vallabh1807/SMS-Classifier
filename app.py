import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps = PorterStemmer()



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

#Loading the saved vectorizer and Naive Bayes Model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model(1).pkl','rb'))

 #Streamlit code
st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")
if st.button('Predict'):
    #1. preprocess
    transformed_sms = transform_text(input_sms)
    #2.vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #4. Dissplay
    if result == 1:
       st.header("Spam")
    else:
       st.header("Not Spam")
