import streamlit as st
import pandas as pd
import numpy as np
import string
import time
import random
import joblib
from sentence_transformers import SentenceTransformer, util
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode, DataReturnMode
from prompt_template import Template, GPT


# page config
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    # button.css-nqowgj.e1ewe7hr3{
    #     display: none;
    # }
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    # .css-lrlib {
    #     padding-top: 3rem;
    # }
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
    # path = Path(__file__).parents
    embedding_model = load_sentenceBert()
    corpus_embedding = embedding_model.encode(list(init_data['tweet_text']))
    return corpus_embedding

# functions
def similarity_search(query, data):
    search_model = load_sentenceBert()
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    corpus_embedding = load_embedding()
    top_k = util.semantic_search(query_embedding, corpus_embedding, top_k=len(data))
    top_id = [i['corpus_id'] for i in top_k[0]]
    new_id = []
    for i in top_id:
        if i in list(data.index): 
            new_id.append(i)
    # data = data.sort_values(by='index', key=lambda column: column.map(lambda e: new_id.index(e)))
    # st.write(new_id)
    data = init_data.iloc[new_id]
    return data

def boolean_search(query, data):
    data = data[data['tweet_text'].str.contains(query) == True]
    return data

def generate_random(data):
    random_list = []
    for i in range(0,len(data)):
        random_list.append(round(random.uniform(0.0, 1.0), 3))
    return random_list

def generate_random_boolean(data):
    random_list = []
    for i in range(0,len(data)):
        # random_list.append(random.choice([0, 1]))
        random_list.append(1)
    return random_list

# initiate session state
if 'select_search' not in st.session_state:
    st.session_state.before_example = []
    st.session_state.middle_example = []
    st.session_state.after_example = []

if 'user_defined_facet_number' not in st.session_state:
    st.session_state['user_defined_facet'] = []
    st.session_state['user_defined_prompts'] = []
    st.session_state['user_defined_facet_number'] = 0
    st.session_state['GPT_filtered_data'] = pd.DataFrame([])
    st.session_state.value_watcher = []
    
# load data
TEST_URL = './user_test_data_cleaned.csv'
original_data = load_data(TEST_URL)
if st.session_state['GPT_filtered_data'].empty:
    init_data = original_data
else:
    init_data = st.session_state['GPT_filtered_data'].sort_index()

df = init_data

# UI
st.subheader('Create and add new criterion')

st.info(f'You are going to use GPT-3 to create new criteria to filter claims. Please provide **:red[detailed descriptions of the new criteria]** to GPT-3 so that it helps to preprocess claims that are more likely to match the new criteria.')


facet_name = st.text_input(f'**Criterion name**: what is your new criterion?', placeholder="propaganda")

prompts_1 = st.text_area(f'**Descriptions**: how would you describe the new criterion?', 
                                     key='free_form_customized', 
                                     placeholder='e.g., if the new facet aims to detect propaganda claims, describe how a propaganda clam is written')

query = st.text_input(f"**Representative examples**: which claims that match the new criterion?", placeholder="search claims using keywords")
query_search = st.radio("xx", ('Boolean Search', 'Similarity Search'), horizontal=True, label_visibility='collapsed', key = 'select_search')
if query_search == 'Similarity Search' and query:
    df = similarity_search(query, df)
if query_search == 'Boolean Search' and query:
    df = boolean_search(query, df)
# df = search(query, df)

st.session_state.before_example = list(df['tweet_text'])

delect_js = JsCode("""
    function(e) {
        let api = e.api;        
        let sel = api.getSelectedRows();
        setTimeout(function(){
            api.applyTransaction({remove: sel})
        }, 500);
    };
    """)

# AgGrid version
col1, col2 = st.columns(2)
df_before = pd.DataFrame(st.session_state.before_example, columns=['tweet_text'])
with col1:
    edited_df = GridOptionsBuilder.from_dataframe(df_before)
    edited_df.configure_column('tweet_text', wrapText=True, autoHeight=True)
    edited_df.configure_column('tweet_text', header_name='tweets (click to add)', **{'width':1000})
    edited_df.configure_selection(selection_mode="single", use_checkbox=True)
    # edited_df.configure_grid_options(onRowSelected=delect_js)
    gridOptions = edited_df.build()
    grid_table = AgGrid(df_before, 
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            gridOptions = gridOptions,
                            fit_columns_on_grid_load=True,
                            height = 600,
                            width = '100%',
                            custom_css = {".ag-cell-value": {'line-height': '20px','padding': '10px'}, "#gridToolBar": {'display':'none'}},
                            allow_unsafe_jscode = True,
                                # reload_data = False
                            )
    # st.session_state.before_example = list(grid_table.data['tweet_text'])
    for i in grid_table.selected_rows:
        if i['tweet_text'] not in st.session_state.middle_example:
            st.session_state.middle_example.append(i['tweet_text'])

df_middle = pd.DataFrame(st.session_state.middle_example, columns=['tweet_text'])

with col2:
    df_middle['label'] = 1
    selected_df = GridOptionsBuilder.from_dataframe(df_middle)
    selected_df.configure_column('label', editable=True)
    selected_df.configure_column('tweet_text', wrapText=True, autoHeight=True, editable = True)
    selected_df.configure_column('tweet_text', header_name='selected tweets (click to remove)', **{'width':1000})
    selected_df.configure_selection(selection_mode="single", use_checkbox=True)
    selected_df.configure_grid_options(onRowSelected=delect_js)
    gridOptions = selected_df.build()
    selected_table = AgGrid(df_middle, 
                            gridOptions = gridOptions,
                            editable = True,
                            # reload_data = False,
                            fit_columns_on_grid_load=True,
                            height = 600,
                            width = '100%',
                            custom_css = {".ag-cell-value": {'font-size': '14px', 'line-height': '20px','padding': '10px'}, "#gridToolBar": {'display':'none'}},
                            allow_unsafe_jscode = True,
                            data_return_mode = DataReturnMode.AS_INPUT,
                            update_mode = GridUpdateMode.MODEL_CHANGED
                            )
    if selected_table.data.empty == False:
        st.session_state.after_example = list(selected_table.data['tweet_text'])
    # else:
    #     st.session_state.middle_example = []
        # st.experimental_rerun()
        
    # for i in selected_table.selected_rows:
    #     st.session_state.before_example.append(i['tweet_text'])
    # df_before = pd.DataFrame(st.session_state.before_example, columns=['tweet_text'])
    st.session_state.middle_example = st.session_state.after_example

# st.write(selected_table.data)

prompts_2 = []
for idx, row in selected_table.data.iterrows():
    prompts_2.append([row['tweet_text'], row['label']])
# st.write(prompts_2)
    # st.session_state.after_example = [] 
    # for i in selected_table.data:
    #     st.session_state.after_example.append(i['tweet_text'])
    # for i in selected_table.selected_rows:
    #     st.session_state.before_example.append(i['tweet_text'])
    
# prompts_2 = st.session_state.temp_example
final_submission = st.button('Confirm and add new criterion', type='primary')
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
            k = 20
            GPT_response_list = []
            progress_text = 'The GPT is processing your texts.'
            progress_bar = st.progress(0, text=progress_text)
            percent_complete = 0
            ## generate prompts
            for text in df['tweet_text'][0:k]:
                prompt = Template(facet_name, text, prompts_1, prompts_2)
                # st.markdown(prompt.prompt())
                GPT_response = GPT(prompt.prompt())
                GPT_response_list.append(GPT_response.generate_1())
                progress_bar.progress(percent_complete, text=progress_text)
                percent_complete = percent_complete + 1/len(df)
            for text in df['tweet_text'][k::]:
                progress_bar.progress(percent_complete, text=progress_text)
                percent_complete = percent_complete + 1/len(df)
            time.sleep(0.1)    
            progress_bar.progress(100, text='completed')
            new_facet_answer = [item[0] for item in GPT_response_list] + generate_random_boolean(df)[k::]
            new_facet_prob = [item[1] for item in GPT_response_list] + generate_random(df)[k::]
            # new_facet_answer = [item[0] for item in GPT_response_list]
            # new_facet_prob = [item[1] for item in GPT_response_list]
            df[[facet_name, facet_name + '_prob']] = list(zip(new_facet_answer, new_facet_prob))
            # generate random number for testing
            # df[facet_name] = generate_random_boolean(df)
            # df[facet_name + '_prob'] = generate_random(df)
            # st.dataframe(df)
            st.session_state['GPT_filtered_data'] = df
            st.success('You have successfully added the new facet to the facet browsing, go back and play with it!', icon="✅")
        else:
            st.warning('You have already created this facet. Please give it a different name for the new facet.', icon="👻")
    else:
        st.error('Please complete both of the name and prompt for your new facet. The prompt could be either the description or the example.', icon="👻")

# st.write(st.session_state['user_defined_facet'])
# st.write(st.session_state['user_defined_prompts'])
# st.dataframe(st.session_state['GPT_filtered_data'])