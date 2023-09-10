import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime

# get random string of letters and digits
selected_claims = []
if st.session_state['logger']:
    claim_id = (id for id in range(1, 100_00))
    for item in st.session_state['logger']:
        with st.container():
            if item[2]['user_query']:
                st.markdown(f"""
                    ##### {item[2]['user_query'][-1][0]}: {item[2]['user_query'][-1][1]}  
                    """)
            else:
                st.markdown(f"""
                    ##### No query
                    """)
            for i in item[1]['selected_claims']:
                claim = "claim_checkbox_" + str(next(claim_id))
                if st.checkbox(i['tweet_text'], key=claim, value=False):
                    selected_claims.append([i['tweet_text'], i['tweet_id']])
            
            st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
else:
    st.write("You haven't selected any claims~")

# st.write(st.session_state['logger'])

json_string = json.dumps([st.session_state['logger'], selected_claims, st.session_state['time_series']])

st.download_button(
    label = f"Download task data",
    data = json_string,
    file_name = f'interface_A.json',
    mime='application/json',
    disabled=st.session_state.claim_selected
)

# st.write(st.session_state['time_series'])
# st.write(st.session_state['number_slider_change'])
# st.write(st.session_state['number_search'])