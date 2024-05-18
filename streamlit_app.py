import time
import pandas as pd
import streamlit as st
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

SENTIMENT_ANALYSIS_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')
st.title('Streamlit - tłumacz')
st.header('Przetwarzanie języka naturalnego')
st.subheader('Do wyboru są dwie opcje')
st.text('1. Sprawdzenie wydźwięku emocjonalnego tekstu napisanego w języku angielskim')
st.text('2. Tłumaczenie tekstu z języka angielskiego na język niemiecki')

options = [
    'Wydźwięk emocjonalny tekstu w języku angielskim',
    'Tłumacz z języka angielskiego na język niemiecki'
]

option = st.selectbox('Opcje', options)

if option == options[0]:
    text = st.text_area(label='Tekst do analizy')
    if text:
        try:
            classifier = pipeline("sentiment-analysis", model=SENTIMENT_ANALYSIS_MODEL_NAME)
            answer = classifier(text)
            st.spinner()

            st.write(answer[0]['label'])
        except Exception as e:
            st.error('Wystąpił błąd przy tłumaczeniu - spróbuj ponownie')
            print(str(e))

elif option == options[1]:
    text_to_translate = st.text_area(label='Tekst do przetłumaczenia')
    if text_to_translate:
        try:
            AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
            AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
            translator = pipeline('translation_en_to_de', model=TRANSLATION_MODEL_NAME)
            result = translator(text_to_translate, max_length=100)[0]['translation_text']

            st.spinner()
            with st.spinner(text='Tłumaczenie w toku...'):
                time.sleep(2)
                st.success(result)
        except Exception as e:
            st.error('Wystąpił błąd przy tłumaczeniu - spróbuj ponownie')
            print(str(e))

st.write('Autor: Aleksander Łapiński, Indeks: s22064')