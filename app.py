import streamlit as st
from preprocessing import sub_preprocessing
from catboost import CatBoostClassifier, Pool
import pandas as pd
import nltk


print('Start app')
df_words = pd.read_csv('Oxford_classification.csv')
model = CatBoostClassifier()
model.load_model('catboostclassifier_model.cbm')
features = ['text_len',
            'sub_per_second',
            'phrases_lenght',
            'sub_per_second_frases',
            'sub_per_word',
            'num_sentence',
            'word_persentence',
            'A2',
            'B1',
            'B2',
            'C1',
            'B1+',
            'words_unique_count',
            'words_unique_per_second',
            'words_unique_part',
            'gerund',
             'gerund_per_sentence']

st.title('Классификация фильмов по сложности восприятия английского языка')


upload_file = st.file_uploader('Откройте файл субтитpов в формате .srt', type='srt')

def make_predict(data, model):
    """
    :param data:
    :param model:
    :return:
    """
    predict_pool = Pool(data=data)
    predict = model.predict(predict_pool)
    decode = {2:'A2',
              3:'B1',
              4:'B2',
              5:'C1'
             }

    return decode[predict[0][0]]

if upload_file:
    print(upload_file.name)
    df = sub_preprocessing(upload_file, df_words, df_idioms)
    st.header(f'Данный фильм имеет уровень **{make_predict(df[features], model)}** :sunglasses: по классификации CEFR')

