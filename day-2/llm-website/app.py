import streamlit as st 
from api_handlers import groq_stream, generate_image
st.set_page_config(
    page_title="Universal AI toolkit",
    page_icon='🤖',
    layout='wide'
)

st.sidebar.title('Universal AI toolkit')
st.sidebar.caption('Day2 final project')

options = [
    'Text Chat', 'Audio Studio', 'Vision Analyzer', 'Image Generator'
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
                
elif tab == "Image Generator":
    st.header('Image Generator : ')
    prompt = st.text_area('Describe the image you wanted to generate', height = 150, placeholder = "e.g Genearte the image of diety called lord ganesha")
    if  prompt and st.button('Generate Image', type='primary'):
        with st.spinner('Generating image.... (takes 15 - 30 seconds)'):
            image = generate_image(prompt)
            st.image(image, caption="Generated image", use_column_width=True)
            st.download_button('Download Image', image, 'generated_image.png', 'image/png')