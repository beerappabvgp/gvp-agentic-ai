import streamlit as st 
from api_handlers import groq_stream
st.set_page_config(
    page_title="Universal AI toolkit",
    page_icon='🤖',
    layout='wide'
)

st.sidebar.title('Universal AI toolkit')
st.sidebar.caption('Day2 final project')

options = [
    'Text Chat', 'Audio Studio', 'Vision Analyzer'
]
tab = st.sidebar.radio('Select a tool', options)

if tab == "Text Chat":
    st.header('Text Chat - Groq LLM')
    models = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile']
    col1, col2 = st.columns([3,1])
    with col2:
        model = st.selectbox('Model', models)
        temp = st.slider('Temperature', 0.0, 1.0, 0.6)
    with col1:
        system_msg = st.text_area('System Prompt', value = 'You are a human being and answer the questions accordingly ')
        user_msg = st.text_area('Your Message', height = 120, placeholder="e.g Can you explain me human relations in detail and in depth")
        
        if st.button('Generate Response', type='primary') and user_msg:
            with st.spinner("Generating ........"):
                st.write_stream(groq_stream(system_msg, user_msg, model, temp))