import streamlit as st
import streamlit_toggle as tog
import pandas as pd
import plotly.figure_factory as ff
from prompt_template import Template, GPT

st.set_page_config(layout="centered")

@st.cache_data()
def load_data(url):
    data = pd.read_csv(url)
    return data

TEST_URL = './user_test_data_cleaned_low.csv'
data = load_data(TEST_URL)

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
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    .css-1629p8f h4 {
        font-weight: 100;
        padding_top: 10px;
    }
    label.css-1whk732.e10yodom0 {
        padding-top: 10px;
    }
    </style>
    """,unsafe_allow_html=True)

def draw_graph(data, name, prob):
    df_fig = data[data[name] == 1]
    fig = ff.create_distplot([df_fig[prob]*10], group_labels=['x'], bin_size=.1, show_rug=False, show_curve=False, colors=['rgba(255, 75, 75, 0.65)'])
    fig.update_layout(showlegend=False, height=50, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False}, xaxis={'visible': False, 'showticklabels': False})
    if name + '_slider' in st.session_state:
        fig.add_vrect(x0=st.session_state[name + '_slider'][0]*10, x1=st.session_state[name + '_slider'][1]*10 + 0.06,fillcolor="rgba(255, 75, 75, 0.35)", opacity=0.5,layer="below", line_width=1.5, line_color="rgba(255, 75, 75, 0.7)")
    else:
        fig.add_vrect(x0=0.00, x1=10.10,fillcolor="rgba(255, 75, 75, 0.35)", opacity=0.5,layer="below", line_width=1.5, line_color="rgba(255, 75, 75, 0.7)")
    graph = st.plotly_chart(fig, theme='streamlit', config={'staticPlot': True}, use_container_width=True)
    return graph

st.markdown('### 👋 Hello~')
st.markdown('We are the [Artificial Intelligence and Human-Centered Computing Group](https://ai.ischool.utexas.edu/) from the University of Texas at Austin. We build a claim selection prototype to explore **how fact-checkers and journalists triage different claim criteria to prioritize checkworthy claims for fact-checking**. Three AI models are applied to rank and filter hundreds or thousands of claims posted on social media. We describe corresponding tool functions as followed.')

st.markdown('<br>', unsafe_allow_html=True)

st.markdown('##### 1. Weighted ranking')
st.markdown('lets you personalize the importance of different claim criteria to rank all claims. Claims that appear at the top are predicted to more likely match the corresponding criteria that are assigned with higher weights. You can use the slider to adjust the weight of each criterion.')

col1, col2 = st.columns([4, 3])
with col1:
    test_weight_slider = st.slider('Verifiable', key='test_weight', min_value=0.0, value=0.0, max_value=1.0, format="%f", help="The tweet contains a verifiable factual claim.")
    st.caption(f'You currently weight **:red[{test_weight_slider}]** score on the "verifiable" criterion.')

st.markdown('<br>', unsafe_allow_html=True)

st.markdown('##### 2. Faceted searching')
st.markdown('lets you filter irrelevant claims. If you switch on certain criterion, claims only predicted to contain the criterion will be displayed. You can select a range of probabilities the model uses to filter claims.')

col1, col2 = st.columns([4.5, 3])
with col1:
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<p style="font-size: 14px; padding-top:10px">Verifiable</p>', unsafe_allow_html=True, help="The tweet contains a verifiable factual claim.")
    with col2:
        test_select = tog.st_toggle_switch(label=None, inactive_color='rgba(151, 166, 195, 1)', active_color='rgb(255, 75, 75)', track_color='rgba(151, 166, 195, 0.5)')
    if test_select:
        st.session_state.verifiable = False
        draw_graph(data, 'verifiable', 'verifiable_numeric')
        test_slider = st.slider('Select a range of values',0.00, 1.00, (0.00, 1.00), format="%f",
                                    key='verifiable_slider', disabled=st.session_state.verifiable, label_visibility='collapsed')
        st.caption(f'Claims of which the probability predicted to be verifiable between **:red[{test_slider[0]}]** and **:red[{test_slider[1]}]** will be displayed.')
    else:
        st.caption(f'Please turn on the "verifiable" filter.')

st.markdown('<br>', unsafe_allow_html=True)

st.markdown('##### 3. User-customized facet')
st.markdown('lets you create personalized criteria to rank or filter claims beyond default claim criteria. By providing the context of what the new criterion is, an LLM (Large Language Model) helps you preprocess all claims and add the new criterion as another facet for ranking and searching.')

col1, col2 = st.columns([4, 3])
with col1:
    test_claim = 'Starting later this month, thousands more Ohioans — including Cincinnati residents — will be able to get COVID-19 vaccines.'
    st.markdown(f'**Claim**: {test_claim}')
    st.markdown(f'**New criterion**: Propaganda')
    test_prompt = st.text_area(f'How would you describe the criterion?', 
                                        key='free_form_customized', 
                                        placeholder='e.g., claims that are propaganda usually call people to action.')
    test_submitted = st.button('Process it with GPT')
    if test_submitted:
        if test_prompt:
            prompt_testing = Template('propaganda', test_claim, test_prompt, [])
            test_response = GPT(prompt_testing.prompt())
            if test_response.generate_1()[0] == 0:
                st.caption(f'GPT thinks there is a **:green[{test_response.generate_1()[1]}]** chance that this claim is not propaganda.')
            else:
                st.caption(f'GPT thinks there is a **:red[{test_response.generate_1()[1]}]** chance that this claim is propaganda.')
        else:
            st.caption('Please put context in the text box.')