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
import datetime
import json


# page config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    button.css-nqowgj.e1ewe7hr3{
        display: none;
    }
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
    data['similarity_numeric'] = 0
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

def save_processing_state():
    if 'facet_info' in st.session_state and 'temp_results' in st.session_state:
        processing_state = {
            'facet_info': st.session_state['facet_info'],
            'temp_results': st.session_state['temp_results'],
            'gpt_processing': st.session_state['gpt_processing']
        }
        # Save to session state as JSON string
        st.session_state['processing_state_json'] = json.dumps(processing_state)

def load_processing_state():
    if 'processing_state_json' in st.session_state and st.session_state['processing_state_json']:
        try:
            processing_state = json.loads(st.session_state['processing_state_json'])
            st.session_state['facet_info'] = processing_state['facet_info']
            st.session_state['temp_results'] = processing_state['temp_results']
            st.session_state['gpt_processing'] = processing_state['gpt_processing']
            return True
        except Exception as e:
            st.error(f"Error loading processing state: {str(e)}")
    return False

# initiate session state
if 'select_search' not in st.session_state:
    st.session_state.before_example = []
    st.session_state.middle_example = []
    st.session_state.after_example = []

if 'user_defined_facet_number' not in st.session_state:
    st.session_state['logger'] = []
    st.session_state['user_defined_facet'] = []
    st.session_state['user_defined_prompts'] = []
    st.session_state['user_defined_facet_number'] = 0
    st.session_state['GPT_filtered_data'] = pd.DataFrame([])
    st.session_state['search_type'] = ['none']
    st.session_state['search_query'] = ['none']
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
    st.session_state['gpt_processing'] = False
    st.session_state['temp_results'] = []
    st.session_state['facet_info'] = None
    st.session_state['processing_state_json'] = None
    
# load data
TEST_URL = './final_test_data.csv'
original_data = load_data(TEST_URL)
if st.session_state['GPT_filtered_data'].empty:
    init_data = original_data
else:
    init_data = st.session_state['GPT_filtered_data'].sort_index()

df = init_data

# UI
st.subheader('Create and add new criterion')

# Check if we need to resume processing
has_processing_state = load_processing_state()

# Check if we're in the middle of processing
if st.session_state['gpt_processing']:
    # Show simple processing status
    if 'facet_info' in st.session_state and st.session_state['facet_info']:
        facet_info = st.session_state['facet_info']
        st.info(f"Creating criterion: **{facet_info['facet_name']}**")
    
    # Show progress
    if 'temp_results' in st.session_state:
        progress = len(st.session_state['temp_results'])
        total = len(df)
        percent_complete = int(progress/total*100)
        
        # Simple progress display
        st.progress(min(1.0, progress/total))
        st.text(f"Processed {progress} of {total} claims ({percent_complete}%)")
        
        # Simple time estimate
        claims_remaining = total - progress
        est_seconds = claims_remaining / 2
        if est_seconds > 60:
            st.caption(f"Approximately {int(est_seconds/60)} minutes remaining")
        else:
            st.caption(f"Less than a minute remaining")
    
    # Just a cancel button
    if st.button("Cancel Processing", type="secondary"):
        st.session_state['gpt_processing'] = False
        st.session_state['temp_results'] = []
        st.session_state['facet_info'] = None
        st.session_state['processing_state_json'] = None
        st.rerun()
        
    # Auto-resume processing - no button needed
    
else:
    st.info(f'You are going to use ChatGPT to create new criteria to filter claims. Please provide **:red[detailed descriptions of the new criteria]** to ChatGPT so that it helps to preprocess claims that are more likely to match the new criteria.')

    def start_GPT():
        st.session_state['time_series'].append({'GPT_edit_prompt': datetime.datetime.now().timestamp()})

    col1, col2, col3 = st.columns([5,0.25,5])

    with col1:
        facet_name = st.text_input(f'**Criterion name**: what is your new criterion?',
                                        placeholder="propaganda", 
                                        on_change=start_GPT)
        prompts_1 = st.text_area(f'**Descriptions**: how would you describe the new criterion?',
                                        placeholder='Claims are sensationalized or contain hyperbolic language grabbing public attention, thus easily calling people to action.',
                                        height= 160,
                                        on_change=start_GPT)

        criterion_name = facet_name
        descriptions = prompts_1
                
    with col3:
        prompt_template = st.selectbox(
            f'**View prompt examples:**',
            ('Propaganda', 'Difficult', 'Urgent')
        )

        if prompt_template == 'Propaganda':
            with st.chat_message("user", avatar = "🧑‍💻"):
                st.write(f"I want to identify claims that are propaganda.")
            with st.chat_message("user", avatar = "🧑‍💻"):
                st.write(f"Claims are sensationalized or contain hyperbolic language grabbing public attention, thus easily calling people to action.")

        if prompt_template == 'Difficult':
            with st.chat_message("user", avatar = "🧑‍💻"):
                st.write(f"I want to identify claims that are difficult to check.")
            with st.chat_message("user", avatar = "🧑‍💻"):
                st.write(f"To fact-check this claim is very difficult because there is no evidence, it requires a long time, or it is too subjective.")

        if prompt_template == 'Urgent':
            with st.chat_message("user", avatar = "🧑‍💻"):
                st.write(f"I want to identify claims that are urgent.")
            with st.chat_message("user", avatar = "🧑‍💻"):
                st.write(f"Fact-checker should immediately fact-check this claim otherwise it might cause distrastous harm to the public.")

    prompts_2 = []

# Processing submission - either from initial submission or resuming
final_submission = st.button('Confirm and add new criterion', type='primary', disabled=st.session_state['gpt_processing'])

if final_submission or st.session_state['gpt_processing']:
    # If we're resuming from a saved state
    if st.session_state['gpt_processing'] and 'facet_info' in st.session_state and st.session_state['facet_info']:
        facet_name = st.session_state['facet_info']['facet_name']
        prompts_1 = st.session_state['facet_info']['description']
        prompts_2 = st.session_state['facet_info'].get('examples', [])
        
    # Initial submission checks
    if (facet_name and (prompts_1 or prompts_2)) or st.session_state['gpt_processing']:
        try:
            # Set processing flag and save facet info if starting new
            if not st.session_state['gpt_processing']:
                st.session_state['time_series'].append({'GPT_start': datetime.datetime.now().timestamp()})
                
                # Check if facet name already exists
                if facet_name in [items['facet_name'] for items in st.session_state['user_defined_facet']]:
                    facet_name = facet_name + "_new"
                
                # Initialize processing state
                st.session_state['gpt_processing'] = True
                st.session_state['temp_results'] = []
                st.session_state['facet_info'] = {
                    'facet_name': facet_name,
                    'description': prompts_1,
                    'examples': prompts_2
                }
                
                # Record user input in session state
                st.session_state['user_defined_facet_number'] += 1
                t = st.session_state['user_defined_facet_number']
                t_facet = {'facet_name': facet_name}
                st.session_state['user_defined_facet'].append(t_facet)
                
                if prompts_1:
                    st.session_state['user_defined_prompts'].append({'prompt': prompts_1})
                if prompts_2:
                    st.session_state['user_defined_prompts'].append({'prompt': prompts_2})
                
                # Save state for persistence
                save_processing_state()
                
                # Show initial message
                message = st.warning("The GPT is processing your texts. Don't leave this page otherwise the data will be lost.")
            else:
                # Resuming processing
                message = st.warning("Resuming processing. Please wait...")
            
            # Set up visual elements for processing
            st.markdown("### Processing Claims")
            
            # Simple progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Add stop button
            stop_col1, stop_col2 = st.columns([3, 1])
            with stop_col1:
                stop_button = st.button("Stop Processing", type="secondary")
                if stop_button:
                    raise InterruptedError("Processing stopped by user")
            
            # Get starting position - continue from where we left off
            start_idx = len(st.session_state['temp_results'])
            if start_idx > 0:
                status_text.text(f"Resuming from claim {start_idx+1}...")
                time.sleep(1)  # Give user time to read the message
                GPT_response_list = st.session_state['temp_results'].copy()
            else:
                GPT_response_list = []
            
            # Process claims in batches
            batch_size = 50  # Process in batches of 50
            
            for i in range(start_idx, len(df), batch_size):
                batch_end = min(i + batch_size, len(df))
                status_text.text(f"Processing claims {i+1} to {batch_end} of {len(df)}...")
                
                batch = df['tweet_text'][i:batch_end]
                batch_responses = []
                
                for j, text in enumerate(batch):
                    # Update progress more frequently
                    if j % 5 == 0 or j == len(batch) - 1:
                        new_percent = min(1.0, (i + j) / len(df))
                        progress_bar.progress(new_percent)
                    
                    try:
                        prompt = Template(facet_name, text, prompts_1, prompts_2)
                        GPT_response = GPT(prompt.prompt())
                        response = GPT_response.generate_1()
                        batch_responses.append(response)
                        
                        # Save progress after each item
                        st.session_state['temp_results'].append(response)
                        save_processing_state()
                    except Exception as e:
                        st.error(f"Error on claim {i+j+1}: {str(e)}")
                        # Save even on error so we can retry
                        save_processing_state()
                        time.sleep(1)
                        continue
                
                GPT_response_list.extend(batch_responses)
                
                # Update progress after each batch
                new_percent = min(1.0, batch_end / len(df))
                progress_bar.progress(new_percent)
                
                # Save batch results
                save_processing_state()
                
                # Small delay between batches
                if batch_end < len(df):
                    time.sleep(0.2)
            
            # Processing complete - create a copy of the dataframe with the new columns
            df_copy = df.copy()
            
            # Add new columns for the facet
            new_facet_answer = [item[0] for item in GPT_response_list]
            new_facet_prob = [item[1] for item in GPT_response_list]
            
            # Make sure the lists are the right length
            if len(new_facet_answer) < len(df_copy):
                padding_needed = len(df_copy) - len(new_facet_answer)
                st.warning(f"Adding {padding_needed} default values to complete the dataset")
                new_facet_answer.extend([1] * padding_needed)
                new_facet_prob.extend([0.5] * padding_needed)
            
            # Add the new columns to the dataframe
            df_copy[facet_name] = new_facet_answer
            df_copy[facet_name + '_prob'] = new_facet_prob
            
            # Processing complete - update UI
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success(f"✅ Successfully added the new criterion: {facet_name}")
            st.info("The new criterion has been added to your filtering options. Go back to the Select Claims page to use it.")
            
            # Update the filtered data in session state
            st.session_state['GPT_filtered_data'] = df_copy
            
            # Clear processing state
            st.session_state['gpt_processing'] = False
            st.session_state['temp_results'] = []
            st.session_state['facet_info'] = None
            st.session_state['processing_state_json'] = None
            
            # Record end time
            st.session_state['time_series'].append({'GPT_end': datetime.datetime.now().timestamp()})
            
        except Exception as e:
            # If there's an error, keep temporary results for resuming
            st.error(f"Processing error: {str(e)}")
            st.info("Your progress has been saved. You can resume processing by returning to this page.")
            
            # Make sure we save the state on error
            save_processing_state()
            
    else:
        st.error('Please complete both the name and prompt for your new criterion. The prompt could be either the description or an example.', icon="👻")