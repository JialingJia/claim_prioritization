import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer, util
import streamlit_toggle as tog
import streamlit.components.v1 as componentsvalue_watcher
import plotly.figure_factory as ff
import requests

# page config
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    [data-testid=stVerticalBlock]{
        gap: 0.5rem;
    }
    .css-1629p8f h4 {
        font-weight: 300;
    }
    label.css-1whk732.e10yodom0 {
        padding-top: 10px;
    }
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    # button.css-nqowgj.e1ewe7hr3{
    #     display: none;
    # }
    </style>
    """,unsafe_allow_html=True)

# data and model cache
@st.cache_data()
def load_data(url):
    data = pd.read_csv(url)
    return data

@st.cache_data(show_spinner=False)
def filter_data(data, facet):
    data = data.loc[(data[facet] == 1)]
    data = data.sort_values(by='weighted_score', ascending=False)
    return data

@st.cache_resource()
def load_sentenceBert():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource()
def load_embedding():
    # path = Path(__file__).parents
    return joblib.load('./embeddings/corpus_embedding.joblib')

# st.write()

# session state
if 'user_defined_facet_number' not in st.session_state:
    st.session_state['user_defined_facet'] = []
    st.session_state['user_defined_prompts'] = []
    st.session_state['user_defined_facet_number'] = 0
    st.session_state['GPT_filtered_data'] = pd.DataFrame([])
    st.session_state.selected_claims = []
    st.session_state.value_watcher = []

st.session_state.verifiable = True
st.session_state.false_info = True
st.session_state.interest_to_public = True
st.session_state.general_harm = True
st.session_state.attention_to_fact_check = True
# st.session_state.government_interest = True
# st.session_state['search'] = ''

if st.session_state['user_defined_facet']:
    for item in st.session_state['user_defined_facet']:
        new_facet = item['facet_name']
        st.session_state[new_facet] = True

# functions
def search(query, data):
    search_model = load_sentenceBert()
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    corpus_embedding = load_embedding()
    top_k = util.semantic_search(query_embedding, corpus_embedding, top_k=len(data))
    top_id = [i['corpus_id'] for i in top_k[0]]
    data = init_data.iloc[top_id]
    return data

def draw_graph(data, name, prob):
    df_fig = data[data[name] == 1]
    fig = ff.create_distplot([df_fig[prob]*10], group_labels=['x'], bin_size=.1, show_rug=False, show_curve=False, colors=['rgba(255, 75, 75, 0.65)'])
    fig.update_layout(showlegend=False, height=50, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False}, xaxis={'visible': False, 'showticklabels': False})
    if name + '_slider' in st.session_state:
        fig.add_vrect(x0=st.session_state[name + '_slider'][0]*10, x1=st.session_state[name + '_slider'][1]*10,fillcolor="rgba(255, 75, 75, 0.35)", opacity=0.5,layer="below", line_width=1.5, line_color="rgba(255, 75, 75, 0.7)")
        # fig.add_vline(st.session_state[name + '_slider'][0],line_color="rgba(255, 75, 75, 0.7)", line_width=1.5, line_dash='dot')
        # fig.add_vline(st.session_state[name + '_slider'][1],line_color="rgba(255, 75, 75, 0.7)", line_width=1.5, line_dash='dot')
    else:
        fig.add_vrect(x0=5.0, x1=10.0,fillcolor="rgba(255, 75, 75, 0.35)", opacity=0.5,layer="below", line_width=1.5, line_color="rgba(255, 75, 75, 0.7)")
        # fig.add_vline(0,line_color="rgba(255, 75, 75, 0.7)", line_width=1.5, line_dash='dot')
        # fig.add_vline(10,line_color="rgba(255, 75, 75, 0.7)", line_width=1.5, line_dash='dot')
    graph = st.plotly_chart(fig, theme='streamlit', config={'staticPlot': True}, use_container_width=True)
    return graph

def re_rank(data):
    data['weighted_score'] = (data['verifiable']*data['verifiable_numeric']*verifiable_weight_slider
                                        + data['false_info']*data['false_info_numeric']*false_info_weight_slider
                                        + data['interest_to_public']*data['interest_to_public_numeric']*interest_to_public_weight_slider
                                        + data['general_harm']*data['general_harm_numeric']*general_harm_weight_slider
                                        + data['attention_to_fact_check']*data['attention_to_fact_check_numeric']*attention_to_fact_check_weight_slider)
    if st.session_state['user_defined_facet']:
        for item in st.session_state['user_defined_facet']:
            new_facet = item['facet_name']
            new_facet_weight = new_facet + '_weight_slider'
            data['weighted_score'] = data['weighted_score'] + data[new_facet]*data[new_facet + "_prob"]*st.session_state[new_facet_weight]
    
    data = data.sort_values(by='weighted_score', ascending=False)
    return data

# def embedTweet(tw_url):
#     URL = "https://twitter.com/anyuser/status/" + tw_url
#     api = "https://publish.twitter.com/oembed?url={}".format(URL)
#     response = requests.get(api)
#     try:
#         res = response.json()["html"]
#         return components.html(res, height=800,scrolling=True)
#     except:
#         pass
        
def split_frame(data, rows):
    data = [data.loc[i : i + rows - 1, :] for i in range(0, len(data), rows)]
    return data

# load data
TEST_URL = './user_test_data.csv'
original_data = load_data(TEST_URL)
if st.session_state['GPT_filtered_data'].empty:
    init_data = original_data
else:
    init_data = st.session_state['GPT_filtered_data'].sort_index()

df_filter_data = init_data
# layout

# sidebar

with st.sidebar:

    st.markdown('## Preset')

    col1, col2 = st.columns([3.4, 1])
    with col1:
        st.markdown('#### Verifiable', help="The tweet contains a verifiable factual claim.")
    with col2:
        verifiable_select = tog.st_toggle_switch(label=None, key='verifiable_select', inactive_color='rgba(151, 166, 195, 1)', active_color='rgb(255, 75, 75)', track_color='rgba(151, 166, 195, 0.5)')
    verifiable_weight_slider = st.slider('Verifiable', key='verifiable_weight', min_value=0.0, value=0.0, max_value=1.0, label_visibility='collapsed')
    if verifiable_select:
        st.session_state.verifiable = False
        draw_graph(df_filter_data, 'verifiable', 'verifiable_numeric')
        if verifiable_weight_slider == 0.00:
            st.session_state.verifiable = True
        verifiable_slider = st.slider('Select a range of values',0.50, 1.00, (0.50, 1.00), format="%f",
                                    key='verifiable_slider', disabled=st.session_state.verifiable, label_visibility='collapsed')
        
    st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3.4, 1])
    with col1:
        st.markdown('#### False information', help="The tweet appears to contain false information")
    with col2:
        false_info_select = tog.st_toggle_switch(label=None, key='false_info_select', inactive_color='rgba(151, 166, 195, 1)', active_color='rgb(255, 75, 75)', track_color='rgba(151, 166, 195, 0.5)')
    false_info_weight_slider = st.slider('false_info', key='false_info_weight', min_value=0.0, value=0.0, max_value=1.0, label_visibility='collapsed')
    if false_info_select:
        st.session_state.false_info = False
        draw_graph(df_filter_data, 'false_info', 'false_info_numeric')
        if false_info_weight_slider == 0.00:
            st.session_state.false_info = True
        false_info_slider = st.slider('Select a range of values',0.5, 1.0, (0.5, 1.0), format="%f",
                                      key='false_info_slider', disabled=st.session_state.false_info, label_visibility='collapsed')
    
    st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3.4, 1])
    with col1:
        st.markdown('#### Public interest', help="The tweet has an effect on or will be of interest to the general public.")
    with col2:
        interest_to_public_select = tog.st_toggle_switch(label=None, key='interest_to_public_select', inactive_color='rgba(151, 166, 195, 1)', active_color='rgb(255, 75, 75)', track_color='rgba(151, 166, 195, 0.5)')
    interest_to_public_weight_slider = st.slider('interest_to_public', key='interest_to_public_weight', min_value=0.0, value=0.0, max_value=1.0, label_visibility='collapsed')
    if interest_to_public_select:
        st.session_state.interest_to_public = False
        if interest_to_public_weight_slider == 0.00:
            st.session_state.interest_to_public = True
        draw_graph(df_filter_data, 'interest_to_public', 'interest_to_public_numeric')
        interest_to_public_slider = st.slider('Select a range of values',0.5, 1.0, (0.5, 1.0), format="%f",
                                    key='interest_to_public_slider', disabled=st.session_state.interest_to_public, label_visibility='collapsed')
        
    st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3.4, 1])
    with col1:
        st.markdown('#### General harm', help="The tweet appears to be harmful to society, people, company, or products.")
    with col2:
        general_harm_select = tog.st_toggle_switch(label=None, key='general_harm_select', inactive_color='rgba(151, 166, 195, 1)', active_color='rgb(255, 75, 75)', track_color='rgba(151, 166, 195, 0.5)')
    general_harm_weight_slider = st.slider('general_harm', key='general_harm_weight', min_value=0.0, value=0.0, max_value=1.0, label_visibility='collapsed')
    if general_harm_select:
        st.session_state.general_harm = False
        if general_harm_weight_slider == 0.00:
            st.session_state.general_harm = True
        draw_graph(df_filter_data, 'general_harm', 'general_harm_numeric')
        general_harm_slider = st.slider('Select a range of values',0.5, 1.0, (0.5, 1.0), format="%f",
                                    key='general_harm_slider', disabled=st.session_state.general_harm, label_visibility='collapsed')
        
    st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)

    col1, col2 = st.columns([3.4, 1])
    with col1:
        st.markdown('#### Attention to fact-checkers', help="A professional fact-checker should verify the claim in the tweet.")
    with col2:
        attention_to_fact_check_select = tog.st_toggle_switch(label=None, key='attention_to_fact_check_select', inactive_color='rgba(151, 166, 195, 1)', active_color='rgb(255, 75, 75)', track_color='rgba(151, 166, 195, 0.5)')
    attention_to_fact_check_weight_slider = st.slider('attention_to_fact_check', key='attention_to_fact_check_weight', min_value=0.0, value=0.0, max_value=1.0, label_visibility='collapsed')
    if attention_to_fact_check_select:
        st.session_state.attention_to_fact_check = False
        if attention_to_fact_check_weight_slider == 0.00:
            st.session_state.attention_to_fact_check = True
        draw_graph(df_filter_data, 'attention_to_fact_check', 'attention_to_fact_check_numeric')
        attention_to_fact_check_slider = st.slider('Select a range of values',0.5, 1.0, (0.0, 1.0), format="%f",
                                    key='attention_to_fact_check_slider', disabled=st.session_state.attention_to_fact_check, label_visibility='collapsed')
    
    # slider = st.slider('Select slider:', 
    #                    min_value=0,
    #                    value=5,
    #                    max_value=10,
    #                    label_visibility='collapsed')
    
    # government_interest_select = st.checkbox('Interest to government authorities')
    # if government_interest_select:
    #     st.session_state.government_interest = False
    # government_interest_slider = st.slider('Select a range of values',0.0, 10.0, (0.0, 10.0), format="%d",
    #                               key='government_interest_slider', disabled=st.session_state.government_interest, label_visibility='collapsed')
    weight_slider_list = [verifiable_weight_slider, false_info_weight_slider, interest_to_public_weight_slider, general_harm_weight_slider, attention_to_fact_check_weight_slider]

    st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)

    st.markdown('## Customized')

    if st.session_state['user_defined_facet']:
        for item in st.session_state['user_defined_facet']:
            new_facet = item['facet_name']
            new_facet_slider = new_facet + '_slider'
            col1, col2 = st.columns([3.4, 1])
            with col1:
                  st.markdown(""" #### <span>{new_facet}</span> """.format(new_facet=item['facet_name'].capitalize()),  unsafe_allow_html=True)
            with col2:
                new_facet_select = tog.st_toggle_switch(label=None, key=new_facet + '_select', inactive_color='rgba(151, 166, 195, 1)', active_color='rgb(255, 75, 75)', track_color='rgba(151, 166, 195, 0.5)')
            new_facet_weight_slider = st.slider('new_facet_weight_slider', key=new_facet + '_weight_slider', min_value=0.0, value=0.0, max_value=1.0, label_visibility='collapsed')
            weight_slider_list.append(new_facet_weight_slider)
            if new_facet_select:
                st.session_state[new_facet] = False
                if new_facet_weight_slider == 0.00:
                    st.session_state[new_facet] = True
                draw_graph(df_filter_data, new_facet, new_facet + "_prob")
                new_facet_slider = st.slider('Select a range of values',0.5, 1.0, (0.5, 1.0), format="%f",
                                        key=new_facet_slider, disabled=st.session_state[new_facet], label_visibility='collapsed')
            st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)

    # st.button('create and add new facet', type="primary")
    st.markdown("""<br> """, unsafe_allow_html=True)

    reset = st.button('reset facet', type="secondary")

# body

## search
query = st.text_input("search:", label_visibility="collapsed", placeholder="search claims using keywords")

## filter data
df_filter_data = re_rank(df_filter_data)
if verifiable_select:
    df_filter_data = filter_data(df_filter_data, 'verifiable')
    df_filter_data = df_filter_data[df_filter_data['verifiable_numeric'].between(st.session_state['verifiable_slider'][0], st.session_state['verifiable_slider'][1])]

if false_info_select:
    df_filter_data = filter_data(df_filter_data, 'false_info')
    df_filter_data = df_filter_data[df_filter_data['false_info_numeric'].between(st.session_state['false_info_slider'][0], st.session_state['false_info_slider'][1])]

if interest_to_public_select:
    df_filter_data = filter_data(df_filter_data, 'interest_to_public')
    df_filter_data = df_filter_data[df_filter_data['interest_to_public_numeric'].between(st.session_state['interest_to_public_slider'][0], st.session_state['interest_to_public_slider'][1])]

if general_harm_select:
    df_filter_data = filter_data(df_filter_data, 'general_harm')
    df_filter_data = df_filter_data[df_filter_data['general_harm_numeric'].between(st.session_state['general_harm_slider'][0], st.session_state['general_harm_slider'][1])]

if attention_to_fact_check_select:
    df_filter_data = filter_data(df_filter_data, 'attention_to_fact_check')
    df_filter_data = df_filter_data[df_filter_data['attention_to_fact_check_numeric'].between(st.session_state['attention_to_fact_check_slider'][0], st.session_state['attention_to_fact_check_slider'][1])]

    # if government_interest_select:
    #     df_filter_data = filter_data(df_filter_data, 'general_harm')
    #     df_filter_data = df_filter_data[df_filter_data['general_harm_numeric'].between(st.session_state['general_harm_slider'][0], st.session_state['general_harm_slider'][1])]

if st.session_state['user_defined_facet']:
    for item in st.session_state['user_defined_facet']:
        new_facet = item['facet_name']
        new_slider = new_facet + '_slider' 
        if st.session_state[new_facet + '_select']:
            df_filter_data = filter_data(df_filter_data, new_facet)
            df_filter_data = df_filter_data[df_filter_data[new_facet + '_prob'].between(st.session_state[new_slider][0], st.session_state[new_slider][1])]

## pagination
pagination = st.container()

bottom_menu = st.columns((1.5,2,1,1))
with bottom_menu[3]:
    batch_size = st.selectbox("Page Size", options=[25, 50, 100], label_visibility="collapsed")
with bottom_menu[2]:
    total_pages = (
        int(len(df_filter_data) / batch_size) if int(len(df_filter_data) / batch_size) > 0 else 1
    )
    current_page = st.number_input(
        "Page", min_value=1, max_value=total_pages, step=1, label_visibility="collapsed"
    )
with bottom_menu[1]:
    st.markdown(f"Pages: **:red[{current_page}]** / **{total_pages}** ")
with bottom_menu[0]:
    st.markdown(f"Filtered claims: **:red[{len(df_filter_data)}]** / **{len(original_data)}** ")

## re-rank data based on user interactions
if query:
    df_filter_data = search(query, df_filter_data)
for ele1, ele2 in zip(weight_slider_list, st.session_state.value_watcher):
    if ele1 != ele2:
        df_filter_data = re_rank(df_filter_data)

## render data
df_filter_data = df_filter_data.reset_index()
pages = split_frame(df_filter_data, batch_size)
# st.dataframe(pages[0])
st.markdown("""<br/> <br/>""", unsafe_allow_html=True)

# claim list
selected_claims = []
for index, rows in pages[current_page - 1].iterrows():
    claim = st.checkbox(rows['tweet_text'], key=f'checkbox{rows.tweet_id}')
    if claim:
        selected_claims.append(rows['tweet_text'])
    # st.write(rows['tweet_text'])
    st.markdown("""<hr style="margin:1em 0px 2em 0px" /> """, unsafe_allow_html=True)
# pagination.dataframe(data = pages[current_page - 1], use_container_width=True)

## download claims
if selected_claims:
    st.session_state.claim_selected = False
else:
    st.session_state.claim_selected = True

# def clear_all():
#     st.session_state.selected_claims = []
#     for index, rows in pages[current_page - 1].iterrows():
#         st.session_state[f'checkbox{rows.tweet_id}'] = False
#     return

claim_select_buttons = st.columns([1.6,1,4])

with claim_select_buttons[0]:
    st.download_button(
        label = f"Download selected claims ({len(selected_claims)})",
        data = pd.DataFrame(selected_claims, columns=['claims']).to_csv().encode('utf-8'),
        file_name = f'page_{current_page}_claims.csv',
        mime='text/csv',
        # disabled=st.session_state.claim_selected
    )

# with claim_select_buttons[1]:
#     deselect_claims = st.button(
#         label = 'Deselect all',
#         disabled=st.session_state.claim_selected,
#         on_click=clear_all
#     )

if reset:
    del st.session_state['user_defined_facet']
    del st.session_state['user_defined_prompts']
    del st.session_state['user_defined_facet_number']
    del st.session_state['GPT_filtered_data']
    st.experimental_rerun()

st.session_state.value_watcher = weight_slider_list