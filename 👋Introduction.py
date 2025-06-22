import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import datetime
from prompt_template import Template, GPT

# Set page configuration
st.set_page_config(layout="wide", page_title="Claim Prioritization Prototype")

# Initialize session state variables if they don't exist
if 'user_defined_facet_number' not in st.session_state:
    st.session_state['logger'] = []
    st.session_state['user_defined_facet'] = []
    st.session_state['user_defined_prompts'] = []
    st.session_state['user_defined_facet_number'] = 0
    st.session_state['GPT_filtered_data'] = pd.DataFrame([])
    st.session_state['search_type'] = ['none']
    st.session_state['search_query'] = [{'type':'none', 'query':'none'}]
    st.session_state['number_search'] = 0
    st.session_state['number_slider_change'] = 0
    st.session_state['number_new_slider_change'] = 0
    st.session_state['number_similiarity_slider_change'] = 0
    st.session_state['start_time'] = datetime.datetime.now().timestamp()
    st.session_state['end_time'] = 0
    st.session_state['claim_candidate'] = []
    st.session_state['time_series'] = [{'start': datetime.datetime.now().timestamp()}]
    st.session_state.selected_claims = []
    st.session_state.value_watcher = [0,0,0,0]
    st.session_state.query_similarity = 0
    st.session_state.similarity_weight_boolean = True

# Add these new variables if not already initialized
if 'gpt_processing' not in st.session_state:
    st.session_state['gpt_processing'] = False
if 'temp_results' not in st.session_state:
    st.session_state['temp_results'] = []
if 'facet_info' not in st.session_state:
    st.session_state['facet_info'] = None
if 'processing_state_json' not in st.session_state:
    st.session_state['processing_state_json'] = None
if 'verifiable' not in st.session_state:
    st.session_state.verifiable = True

# Cache data loading
@st.cache_data()
def load_data(url):
    data = pd.read_csv(url)
    return data

# Load test data
TEST_URL = './final_test_data.csv'
data = load_data(TEST_URL)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Scrollbar styling */
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
    
    /* Hide fullscreen button */
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    
    /* Custom header weight */
    .css-1629p8f h4 {
        font-weight: 100;
        padding-top: 10px;
    }
    
    /* Spacing for form labels */
    label.css-1whk732.e10yodom0 {
        padding-top: 10px;
    }
    
    /* Remove extra spacing around toggles */
    div.stToggleButton {
        margin-top: -5px !important;
        margin-bottom: -5px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    
    /* Make toggle labels minimal */
    div.stToggleButton label {
        font-size: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 0 !important;
    }
    
    /* Make the toggle itself more compact */
    div.stToggleButton > div {
        transform: scale(0.8);
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to draw distribution graphs
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

# Title and introduction
st.title('Claim Selection Prototype')
st.markdown('👋 Hello! Welcome to the Claim Selection Prototype.')

st.markdown('''
We are the [Artificial Intelligence and Human-Centered Computing Group](https://ai.ischool.utexas.edu/) 
from the University of Texas at Austin. This prototype explores **how fact-checkers and journalists 
can use different claim criteria to prioritize checkworthy claims for fact-checking**.

Three AI models help rank and filter hundreds or thousands of claims posted on social media. 
Below, we describe how each feature works.
''')

# Feature 1: Weighted Ranking
st.markdown('### 1. Weighted Ranking')
st.markdown('''
This feature lets you personalize the importance of different claim criteria to rank all claims. 
Claims that appear at the top are predicted to more likely match the criteria that you give higher weights.
''')

col1, col2 = st.columns([4, 3])
with col1:
    test_weight_slider = st.slider(
        'Verifiable', 
        key='test_weight', 
        min_value=0.0, 
        value=0.1, 
        max_value=1.0, 
        format="%f", 
        help="The tweet contains a verifiable factual claim."
    )
    st.caption(f'You currently weight **:red[{test_weight_slider}]** score on the "verifiable" criterion.')

st.markdown('---')

# Feature 2: Faceted Searching
st.markdown('### 2. Faceted Searching')
st.markdown('''
This feature lets you filter claims based on specific criteria. When you switch on a criterion, 
only claims predicted to match that criterion will be displayed. You can also select a probability 
range to fine-tune the filtering.
''')

col1, col2 = st.columns([4.5, 3])
with col1:
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<p style="font-size: 14px; padding-top:10px">Verifiable</p>', unsafe_allow_html=True, help="The tweet contains a verifiable factual claim.")
    with col2:
        test_select = st.toggle('Toggle', key='toggle_demo', label_visibility='hidden')
    
    if test_select:
        st.session_state.verifiable = False
        draw_graph(data, 'verifiable', 'verifiable_numeric')
        test_slider = st.slider(
            'Select a range of values',
            0.00, 1.00, (0.00, 1.00), 
            format="%f",
            key='verifiable_slider', 
            disabled=st.session_state.verifiable, 
            label_visibility='collapsed'
        )
        st.caption(f'Claims with verifiable probability between **:red[{test_slider[0]}]** and **:red[{test_slider[1]}]** will be displayed.')
    else:
        st.caption(f'Please turn on the "verifiable" filter to see the probability distribution.')

st.markdown('---')

# Feature 3: User-customized Facet
st.markdown('### 3. User-customized Criteria')
st.markdown('''
This feature lets you create your own personalized criteria beyond the default options. 
By describing what your new criterion means, an AI model (GPT) will analyze all claims 
and add your custom criterion as a new facet for ranking and filtering.
''')

col1, col2 = st.columns([4, 3])
with col1:
    test_claim = 'Starting later this month, thousands more Ohioans — including Cincinnati residents — will be able to get COVID-19 vaccines.'
    
    st.markdown(f'**Example Claim**: {test_claim}')
    st.markdown(f'**New Criterion**: Propaganda')
    
    test_prompt = st.text_area(
        'How would you describe this criterion?', 
        key='free_form_customized', 
        placeholder='e.g., Claims that are propaganda usually call people to action or use emotional language to persuade.'
    )
    
    test_submitted = st.button('Process with AI')
    
    if test_submitted:
        if test_prompt:
            with st.spinner('Processing with AI...'):
                prompt_testing = Template('propaganda', test_claim, test_prompt, [])
                test_response = GPT(prompt_testing.prompt())
                result = test_response.generate_1()
                
                if result[0] == 0:
                    st.success(f'AI predicts this claim is **not propaganda** with {result[1]:.2f} confidence.')
                else:
                    st.info(f'AI predicts this claim **is propaganda** with {result[1]:.2f} confidence.')
        else:
            st.warning('Please describe the criterion in the text box above.')

# Footer
st.markdown('---')
st.markdown('''
**Ready to start?** Navigate to the "Select claims" page to try these features with real data.

You can also create your own custom criterion in the "Create facet" page and view your 
selected claims in the "Your selection" page.
''')