import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import joblib
from sentence_transformers import SentenceTransformer, util
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode
from prompt_template import Template, GPT


# page config
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    #button.css-nqowgj.e1ewe7hr3{
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# data and model cache
@st.cache_data()
def load_data(url):
    data = pd.read_csv(url)
    return data

@st.cache_resource()
def load_sentenceBert():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource()
def load_embedding():
    return joblib.load('./embeddings/corpus_embedding.joblib')

# functions
def search(query, data):
    search_model = load_sentenceBert()
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    corpus_embedding = load_embedding()
    top_k = util.semantic_search(query_embedding, corpus_embedding, top_k=len(data))
    top_id = [i['corpus_id'] for i in top_k[0]]
    data = init_data.iloc[top_id]
    return data

def generate_random(data):
    random_list = []
    for i in range(0,len(data)):
        random_list.append(round(random.uniform(0.5, 1), 2))
    return random_list

def generate_random_boolean(data):
    random_list = []
    for i in range(0,len(data)):
        random_list.append(random.choice([0, 1]))
    return random_list

# initiate session state
if 'user_defined_facet_number' not in st.session_state:
    st.session_state['user_defined_facet'] = []
    st.session_state['user_defined_prompts'] = []
    st.session_state['user_defined_facet_number'] = 0
    st.session_state['GPT_filtered_data'] = pd.DataFrame([])
    st.session_state.value_watcher = []

# load data
TEST_URL = './user_test_data.csv'
original_data = load_data(TEST_URL)
if st.session_state['GPT_filtered_data'].empty:
    init_data = original_data
else:
    init_data = st.session_state['GPT_filtered_data'].sort_index()

df = init_data

# UI
st.subheader('Create and add new facet')
facet_name = st.text_input(f'**Facet name**: what is your new facet?', placeholder="propaganda")

prompts_1 = st.text_area(f'**Descriptions**: how you want LLMs to filter claims?', 
                                     key='free_form_customized', 
                                     placeholder='e.g., if the new facet aims to detect propaganda claims, describe how a propaganda clam is written')

query = st.text_input(f"**Representative examples**: which claims that match the new facet?", placeholder="search claims using keywords")
df = search(query, df)

# AgGrid version
edited_df = GridOptionsBuilder.from_dataframe(df[['tweet_text']])
# edited_df.configure_default_column(groupable=True)
# edited_df.configure_pagination(enabled=True)
edited_df.configure_column('tweet_text', editable=True, wrapText=True, autoHeight=True)
edited_df.configure_column('tweet_text', header_name='tweets', **{'width':1000})
edited_df.configure_selection(selection_mode="multiple", use_checkbox=True)
gridOptions = edited_df.build()
grid_table = AgGrid(df[['tweet_text']], 
                        update_mode=GridUpdateMode.SELECTION_CHANGED,
                        gridOptions = gridOptions,
                        fit_columns_on_grid_load=True,
                        height = 400,
                        width = '100%',
                        custom_css = {".ag-cell-value": {'line-height': '20px','padding': '10px'}, "#gridToolBar": {'display':'none'}},
                        # reload_data = False
                        )
prompts_2 = [i['tweet_text'] for i in grid_table['selected_rows']]
# st.write(prompts_2)

final_submission = st.button('Confirm and add new facet', type='primary')
if final_submission:
    if facet_name and (prompts_1 or prompts_2):
        if facet_name not in [items['facet_name'] for items in st.session_state['user_defined_facet']]:
            # record user input in session state
            st.session_state['user_defined_facet_number'] += 1
            t = st.session_state['user_defined_facet_number']
            t_facet = {'facet_name':facet_name}
            st.session_state['user_defined_facet'].append(t_facet)
            if prompts_1:
                st.session_state['user_defined_prompts'].append({'prompt':prompts_1})
            if prompts_2:
                st.session_state['user_defined_prompts'].append({'prompt':prompts_2})
            # request GPT
            # k = 100
            GPT_response_list = []
            progress_text = 'The GPT is processing your texts.'
            progress_bar = st.progress(0, text=progress_text)
            percent_complete = 0
            ## generate prompts
            for text in df['tweet_text']:
                prompt = Template(facet_name, text, prompts_1, prompts_2)
                # st.markdown(prompt.prompt())
                GPT_response = GPT(prompt.prompt())
                GPT_response_list.append(GPT_response.generate_1())
                progress_bar.progress(percent_complete, text=progress_text)
                percent_complete = percent_complete + 1/len(df)
            # for text in df['tweet_text']:
            #     progress_bar.progress(percent_complete, text=progress_text)
            #     percent_complete = percent_complete + 1/len(df)
            time.sleep(0.1)    
            progress_bar.progress(100, text='completed')
            # new_facet_answer = [item[0] for item in GPT_response_list] + generate_random_boolean(df)[k::]
            # new_facet_prob = [item[1] for item in GPT_response_list] + generate_random(df)[k::]
            new_facet_answer = [item[0] for item in GPT_response_list]
            new_facet_prob = [item[1] for item in GPT_response_list]
            df[[facet_name, facet_name + '_prob']] = list(zip(new_facet_answer, new_facet_prob))
            # generate random number for testing
            # df[facet_name] = generate_random_boolean(df)
            # df[facet_name + '_prob'] = generate_random(df)
            st.dataframe(df)
            st.session_state['GPT_filtered_data'] = df
            st.success('You have successfully added the new facet to the facet browsing, go back and play with it!', icon="✅")
        else:
            st.warning('You have already created this facet. Please give it a different name for the new facet.', icon="👻")
    else:
        st.error('Please complete both of the name and prompt for your new facet. The prompt could be either the description or the example.', icon="👻")

# st.write(st.session_state['user_defined_facet'])
# st.write(st.session_state['user_defined_prompts'])
# st.dataframe(st.session_state['GPT_filtered_data'])