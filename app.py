import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
import boto3 
# Import the ConfluenceQA class
from confluence_qa import ConfluenceQA
# AWS Session Initialization - Replace OpenAI API key loading
try:
    aws_profile_name = os.getenv('AWS_PROFILE_NAME')
    session = boto3.Session(profile_name=aws_profile_name)
    bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")
except Exception as e:
    print(e)
    load_dotenv()  # Fallback to loading environment variables if needed



st.set_page_config(
    page_title='Q&A Bot for Confluence Page',
    page_icon='⚡',
    layout='wide',
    initial_sidebar_state='auto',
)
if "config" not in st.session_state:
    st.session_state["config"] = {}
if "confluence_qa" not in st.session_state:
    st.session_state["confluence_qa"] = None

@st.cache_resource
def load_confluence(config):
    # st.write("loading the confluence page")
    confluence_qa = ConfluenceQA(config=config)
    confluence_qa.init_embeddings()
    confluence_qa.init_models()
    confluence_qa.vector_db_confluence_docs()
    confluence_qa.retreival_qa_chain()
    return confluence_qa

with st.sidebar.form(key ='Form1'):
    st.markdown('## Add your configs')
    confluence_url = st.text_input("paste the confluence URL", "https://webconnex.atlassian.net/wiki/")
    username = st.text_input(label="confluence username",
                             help="leave blank if confluence page is public",
                             type="password",
                             value="justin.jimenez@webconnex.io")
    space_key = st.text_input(label="confluence space",
                             help="Space of Confluence",
                             value="EN")
    api_key = st.text_input(label="confluence api key",
                            help="leave blank if confluence page is public",
                            type="password")
    submitted1 = st.form_submit_button(label='Submit')

    if submitted1 and confluence_url and space_key:
        st.session_state["config"] = {
            "persist_directory": None,
            "confluence_url": confluence_url,
            "username": username if username != "" else None,
            "api_key": api_key if api_key != "" else None,
            "space_key": space_key,
        }
        with st.spinner(text="Ingesting Confluence..."):
            ### Hardcoding for https://templates.atlassian.net/wiki/ and space RD to avoid multiple OpenAI calls.
            config = st.session_state["config"]
            if  config["confluence_url"] == "https://templates.atlassian.net/wiki/" and config["space_key"] =="RD":
                config["persist_directory"] = "chroma_db"
            st.session_state["config"] = config

            st.session_state["confluence_qa"]  = load_confluence(st.session_state["config"])
        st.write("Confluence Space Ingested")
        

st.title("Confluence Q&A Demo")

question = st.text_input('Ask a question', "How do I make a space public?")

if st.button('Get Answer', key='button2'):
    with st.spinner(text="Asking LLM..."):
        confluence_qa = st.session_state.get("confluence_qa")
        if confluence_qa is not None:
            result = confluence_qa.answer_confluence(question)
            st.write(result)
        else:
            st.write("Please load Confluence page first.")