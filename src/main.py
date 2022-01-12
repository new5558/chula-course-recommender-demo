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

# model_path = './model'
# txt = ""

# def st_shap(plot, height=None):
#     shap_html = f"<head>{shap.initjs()}</head><body>{plot}</body>"
#     components.html(shap_html, height=height)

# def hash_tokenizer(_):
#     return model_path

# def get_label_from_text(text, label_map):
#     for key in label_map.keys():
#         value = label_map[key]
#         if value['name_th'] == text:
#             return key
#     return text

# def map_name_th(label_map):
#     def mapper(label):
#         if label in label_map.keys():
#             return label_map[label]['name_th']
#         else:
#             return label
#     return mapper

@st.cache()
def load_features_array() -> npt.NDArray[np.float64]:
    dict_data_0 = np.load('./course_features/features_array_0.npz')['arr_0']
    dict_data_1 = np.load('./course_features/features_array_1.npz')['arr_0']
    dict_data_2 = np.load('./course_features/features_array_2.npz')['arr_0']
    dict_data_3 = np.load('./course_features/features_array_3.npz')['arr_0']

    return np.concatenate((dict_data_0, dict_data_1, dict_data_2, dict_data_3), axis=0)

@st.cache()
def load_texts_array() -> List[str]:
    texts = pd.read_csv('./course_features/texts_total.csv', index_col=0)['0']
    return texts.values.tolist()

@st.cache()
def load_course_dataframe() -> pd.DataFrame:
    courses_df = pd.read_csv('./course_features/courses_recommender_demo.csv', index_col=0)
    return courses_df[['course_no', 'course_name_thai', 'description_thai']]

@st.cache()
def generate_courses_key_index(all_courses_key: List[str]) \
     -> typing.TypedDict('Key index', key=str, index=int):
    dict = {}
    for i in range(len(all_courses_key)):
        key = all_courses_key[i].strip()
        dict[key] = i

    return dict


# @st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1, hash_funcs={Tokenizer: hash_tokenizer, AddedToken: hash_tokenizer})
# def load_pipeline():
    
#     is_development = bool(os.environ.get('DEVELOPMENT'))
#     model_path = "./model" if is_development else 'new5558/wangchan-course'
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     pipeline = TextClassificationPipeline(model = model, tokenizer = tokenizer)

#     return pipeline, model.config



def main():
    if 'button_sent' not in st.session_state:
        st.session_state['button_sent'] = False
    title = "Chula Course Recommendation demo"
    
    st.set_page_config(page_title=title, page_icon="🎓", layout="wide")
    # _, center_column, __ = st.columns((1, 2, 1))
    # center_column.write("![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)  [Project's repo](https://github.com/new5558/chula-course-faculty-prediction-demo)")
    st.title(title)

    all_courses_key = load_texts_array()
    all_courses_features = load_features_array()
    courses_df = load_course_dataframe()

    courses_key_index = generate_courses_key_index(all_courses_key)
    # st.write(courses_key_index)

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
        st.write(course_no)

        course_description = result[result['course_no'] == course_no]['description_thai'].iloc[0]
        st.write(course_description)
        
        course_description_processed = process_transformers(course_description).strip()
        st.write(courses_key_index[course_description_processed])




    
    # pipeline, model_config = load_pipeline()
    # label_map = download_faculty_map()
    # txt = center_column.text_area('Course description thai', 'บทบาทของเทคโนโลยีสารสนเทศในการแก้ปัญหาทางธุรกิจ แนวคิดและพื้นฐานเทคนิคของศาสตร์เทคโนโลยีสารสนเทศ การวางแผน การพัฒนาและการจัดการของระบบประยุกต์สารสนเทศโดยคอมพิวเตอร์')    

    # if center_column.button(label="Predict Faculty"):
    #     st.session_state['button_sent'] = True

    #     processed_text = process_transformers(txt)

    #     prediction = pipeline(processed_text)[0]
    #     center_column.json(prediction)
    #     predicted_label = prediction['label']
    #     center_column.json(label_map[predicted_label])

    #     predicted_id = model_config.label2id[predicted_label]
        
    #     all_labels = model_config.label2id.keys()
    #     all_labels_text =  list(map(map_name_th(label_map), all_labels))
    #     selected_label_text = center_column.selectbox('Target Faculty for SHAP interpretation', all_labels_text, index = predicted_id)
    #     selected_label = get_label_from_text(selected_label_text, label_map)
    #     selected_id = model_config.label2id[selected_label]

    #     explainer = shap.Explainer(pipeline)
    #     shap_values = None
    #     with st.spinner(text="Interpreting model with SHAP. we don't have GPU so this will take a 1-2 minutes"):
    #         shap_values = explainer([processed_text])

        
    #     st_shap(shap.plots.text(shap_values[0, :, selected_id], display = False))

if __name__ == "__main__":
    main()