import streamlit as st
import streamlit.components.v1 as components
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from thai2transformers.preprocess import process_transformers
from tokenizers import Tokenizer, AddedToken
import shap
import requests
import os

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

# @st.cache()
# def download_faculty_map():
#     r = requests.get('https://raw.githubusercontent.com/DSCChula/chula-util/main/src/faculties.json')
#     label_map = r.json()
#     return label_map

# @st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1, hash_funcs={Tokenizer: hash_tokenizer, AddedToken: hash_tokenizer})
# def load_pipeline():
    
#     is_development = bool(os.environ.get('DEVELOPMENT'))
#     model_path = "./model" if is_development else 'new5558/wangchan-course'
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     pipeline = TextClassificationPipeline(model = model, tokenizer = tokenizer)

#     return pipeline, model.config


def main():
    title = "Chula Course Faculty prediction demo"
    
    # st.set_page_config(page_title=title, page_icon="🎓", layout="wide")
    # _, center_column, __ = st.columns((1, 2, 1))
    # center_column.write("![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)  [Project's repo](https://github.com/new5558/chula-course-faculty-prediction-demo)")
    # center_column.title(title)


    # if 'button_sent' not in st.session_state:
    #     st.session_state['button_sent'] = False
    
    # pipeline, model_config = load_pipeline()
    # label_map = download_faculty_map()
    # txt = center_column.text_area('Course description thai', 'บทบาทของเทคโนโลยีสารสนเทศในการแก้ปัญหาทางธุรกิจ แนวคิดและพื้นฐานเทคนิคของศาสตร์เทคโนโลยีสารสนเทศ การวางแผน การพัฒนาและการจัดการของระบบประยุกต์สารสนเทศโดยคอมพิวเตอร์')    

    # if center_column.button(label="Predict Faculty"):
    #     st.session_state['button_sent'] = True

    # if st.session_state['button_sent']:
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