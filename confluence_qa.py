import os
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from constants import *
import json
import logging
import boto3


from botocore.exceptions import ClientError
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
import boto3
aws_profile_name = os.getenv('AWS_PROFILE_NAME')

session = boto3.Session(profile_name=aws_profile_name)
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")
class ConfluenceQA:
    def __init__(self,config:dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None
    def init_embeddings(self) -> None:
        # AWS Bedrock Titan Embeddings for text

        def embed_text(text):
            """
            Generate a vector of embeddings for a text input using Amazon Titan Embeddings G1 - Text on demand.
            Args:
                model_id (str): The model ID to use.
                body (str) : The request body to use.
            Returns:
                response (JSON): The text that the model generated, token information, and the
                reason the model stopped generating text.
            """
            
            model_id = "amazon.titan-embed-text-v2:0"
            aws_profile_name = os.getenv('AWS_PROFILE_NAME')
            session = boto3.Session(profile_name=aws_profile_name)
            bedrock = session.client("bedrock-runtime", region_name="us-east-1")

            accept = "application/json"
            content_type = "application/json"
            body = json.dumps({
                "inputText": text,
                "dimensions": 512,
                "normalize": True
            })
            response = bedrock.invoke_model(
                body=body, modelId=model_id, accept=accept, contentType=content_type
            )

            response_body = json.loads(response.get('body').read())

            return response_body


        self.embedding = embed_text
    def init_models(self) -> None:
        # AWS Bedrock Claude API
        self.llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-east-1", client=bedrock_client)

        
    def vector_db_confluence_docs(self,force_reload:bool= False) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        persist_directory = self.config.get("persist_directory",None)
        confluence_url = self.config.get("confluence_url",None)
        username = self.config.get("username",None)
        api_key = self.config.get("api_key",None)
        space_key = self.config.get("space_key",None)
        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            ## Load from the persist db
            self.vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding)
        else:
            ## 1. Extract the documents
            loader = ConfluenceLoader(
                url=confluence_url,
                username = username,
                api_key= api_key
            )
            documents = loader.load(
                space_key=space_key, 
                limit=100)
            ## 2. Split the texts
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
            texts = text_splitter.split_documents(texts)

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
            self.vectordb = Chroma.from_documents(documents=texts, embedding=self.embedding, persist_directory=persist_directory)
    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        ##TODO: Use custom prompt
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k":4})
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff",retriever=self.retriever)

    def answer_confluence(self,question:str) ->str:
        """
        Answer the question
        """
        answer = self.qa.run(question)
        return answer
