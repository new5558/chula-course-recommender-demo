import streamlit as st
import streamlit.components.v1 as components
from thai2transformers.preprocess import process_transformers
import requests
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import typing
from typing import List
import sentencepiece as spm

max_token_length = 508

def tokenize_transformers(text, sp: spm.SentencePieceProcessor):
  return sp.encode(process_transformers(text), out_type=str)

def replace_long_sentence(text, sp: spm.SentencePieceProcessor):
    return ''.join(tokenize_transformers(text, sp)[0:max_token_length]).replace('▁', '').strip()

def create_process_description_thai(sp: spm.SentencePieceProcessor):
    def process_description_thai(text):
        course_description_trimmed = replace_long_sentence(text, sp)
        course_description_processed = process_transformers(course_description_trimmed).strip()
        return course_description_processed
    return process_description_thai

def aggregate_features_array(array1: npt.NDArray[np.float64], array2: npt.NDArray[np.float64]):
    features = np.concatenate((array1, array2), axis=0)
    return features, np.linalg.norm(features, axis = 1) 

@st.cache()
def load_features_array_1() -> npt.NDArray[np.float64]:
    dict_data_0 = np.load('./course_features/features_array_0.npz')['arr_0']
    dict_data_1 = np.load('./course_features/features_array_1.npz')['arr_0']
    features = np.concatenate((dict_data_0, dict_data_1), axis=0)
    return features

@st.cache() 
def load_features_array_2() -> npt.NDArray[np.float64]:
    dict_data_2 = np.load('./course_features/features_array_2.npz')['arr_0']
    dict_data_3 = np.load('./course_features/features_array_3.npz')['arr_0']
    features = np.concatenate((dict_data_2, dict_data_3), axis=0)
    return features

    

@st.cache()
def load_texts_array() -> List[str]:
    texts = pd.read_csv('./course_features/texts_total.csv', index_col=0)['0']
    return texts.values.tolist()

@st.cache()
def load_course_dataframe(process_description_thai) -> pd.DataFrame:
    courses_df = pd.read_csv('./course_features/courses_recommender_demo.csv', index_col=0)
    courses_df['description_thai_key'] = courses_df['description_thai'].apply(process_description_thai)
    return courses_df[['course_no', 'course_name_thai', 'description_thai', 'description_thai_key']]

@st.cache()
def generate_courses_key_index(all_courses_key: List[str]) \
     -> typing.TypedDict('Key index', key=str, index=int):
    dict = {}
    for i in range(len(all_courses_key)):
        key = all_courses_key[i].strip()
        dict[key] = i

    return dict

@st.cache()
def load_sentence_pie():
    sp = spm.SentencePieceProcessor(model_file='model/sentencepiece.bpe.model')
    return sp


def main():
    if 'button_sent' not in st.session_state:
        st.session_state['button_sent'] = False
    title = "Chula Course Recommendation demo"
    
    st.set_page_config(page_title=title, page_icon="🎓", layout="wide")
    # _, center_column, __ = st.columns((1, 2, 1))
    # center_column.write("![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)  [Project's repo](https://github.com/new5558/chula-course-faculty-prediction-demo)")
    st.title(title)
    
    sp = load_sentence_pie()

    process_description_thai = create_process_description_thai(sp)
    all_courses_key = load_texts_array()
    courses_features_1 = load_features_array_1()
    courses_features_2 = load_features_array_2()
    all_courses_features, all_courses_features_norm = aggregate_features_array(courses_features_1, courses_features_2)
    courses_df = load_course_dataframe(process_description_thai)

    courses_key_index = generate_courses_key_index(all_courses_key)

    form = st.form(key='search')
    query = form.text_input('Course search query', '')
    submit = form.form_submit_button(label='Submit')

    if submit:
        st.session_state['button_sent'] = True

    
    if st.session_state['button_sent']:
        result = courses_df[courses_df.apply(lambda r: r.str.contains(query, case=False).any(), axis=1)] 

        option = st.selectbox('Search result', result['course_no'] + ": " + result['course_name_thai'], key=result['course_name_thai'])

        st.write('You selected:', option)
        course_no = option.split(':')[0]

        search_result = result[result['course_no'] == course_no]

        course_description_key = search_result['description_thai_key'].iloc[0]
        course_description = search_result['description_thai'].iloc[0]

        st.write(course_description)
        
        course_key = courses_key_index[course_description_key]
        
        query_vector = all_courses_features[course_key].reshape((1, -1))
        dot_product = np.dot(query_vector, all_courses_features.T)
        cosine_similarity = dot_product / (all_courses_features_norm * np.linalg.norm(query_vector, axis = 1) )
        scores = pd.Series(np.flip(np.sort(cosine_similarity[0])[-10:]), name = 'scores')

        max_course = 10
        best_indexes = np.flip(np.argsort(cosine_similarity[0])[-max_course:]).tolist()
        best_keys = [all_courses_key[i] for i in best_indexes]
        
        best_courses = pd.DataFrame()
        for key in best_keys:
            row = courses_df[courses_df['description_thai_key'] == key]
            best_courses = best_courses.append(row)
        
        best_courses = pd.concat([best_courses.reset_index(), scores], axis = 1)
        best_courses = best_courses.drop(['description_thai_key', 'index'], axis = 1)
        
        st.table(best_courses)

if __name__ == "__main__":
    main()