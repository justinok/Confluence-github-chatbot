# import 30 gb at beggining
import os
import pandas as pd
import re
from langchain.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st


# load the document and split it into chunks
# file_path = "/content/sql-writer-finetune - Sheet1.csv"
LLM_OPENAI_GPT35 = "gpt-3.5-turbo"
# Define the file path for the Chroma database
CHROMA_DB_DIR = "./chroma_db"
# Load and clean documents
file_path = "/content/sql-writer-finetune - Sheet1.csv"


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return (
            f"Document(page_content={repr(self.page_content)}, "
            f"metadata={self.metadata})"
        )


def clean_metadata_string(metadata_str):
    # Remove unwanted characters and keep only key-value pairs
    cleaned_metadata_str = re.sub(r"[\'\"\[\]\{\}]", "", metadata_str)
    # Split the string into key-value pairs
    metadata_items = cleaned_metadata_str.split(",")
    # Create a dictionary from key-value pairs
    metadata_dict = {}
    for item in metadata_items:
        if ":" in item:
            key, value = item.split(":", 1)
            metadata_dict[key.strip()] = value.strip()
    return metadata_dict


def clean_page_content(content):
    # Remove unwanted characters and multiple spaces
    content = re.sub(r"\s+", " ", content)
    content = re.sub(r"[\'\"]", "", content)
    return content.strip()


def load_documents_from_csv(file_path):
    df = pd.read_csv(file_path)

    df["page_content"] = df["page_content"].fillna("No Content")
    df["metadata"] = df["metadata"].fillna("No metadata").astype(str)

    documents = []
    for index, row in df.iterrows():
        page_content = clean_page_content(row["page_content"])
        cleaned_metadata_dict = clean_metadata_string(row["metadata"])
        documents.append(
            Document(page_content=page_content, metadata=cleaned_metadata_dict)
        )

    return documents


documents = load_documents_from_csv(file_path)


# create the open-source embedding function
# (Hugging face model whule we get OpenAI key)
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2")

# load it into Chroma
# db = Chroma.from_documents(documents, embedding_function)
# Load or create the Chroma database
if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
    db = Chroma(
        persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
else:
    db = Chroma.from_documents(
        documents, embedding_function, persist_directory=CHROMA_DB_DIR
    )
    db.persist()

# RAG config
llm = ChatOpenAI(model_name=LLM_OPENAI_GPT35, temperature=0.0)
retriever = db.as_retriever(search_kwargs={"k": 4})


def format_document(document):
    # Format the page content and metadata for better readability
    formatted_content = f"**Content:**\n```\n{document.page_content}\n```"
    formatted_metadata = f"**Metadata:** {document.metadata}"
    return f"{formatted_content}\n\n{formatted_metadata}"


def sql_query_assistant():
    QUERY_PROMPT_TEMPLATE = """\
    Human:
    You are an expert SQL writer.
    Create a SQL query based on the provided context.
    Only use the following tables to create the query:
    - wbx_data_dbt.dim_account(
        id INTEGER,
        name CHARACTER VARYING,
        date_created TIMESTAMP,
        date_updated TIMESTAMP,
        date_deleted TIMESTAMP,
        source CHARACTER VARYING,
        referral_code CHARACTER VARYING,
        contact CHARACTER VARYING,
        email CHARACTER VARYING,
        phone CHARACTER VARYING,
        organization_id INTEGER
    )
    - wbx_data_dbt.dim_account_managers (
        id INTEGER,
        first_name CHARACTER VARYING,
        last_name CHARACTER VARYING,
        date_created DATE,
        date_deleted DATE
    )
    - wbx_data_dbt.dim_account_managers_account (
        id NUMERIC,
        account_manager_id INTEGER
    )
    - wbx_data_dbt.dim_billing (
        id INTEGER,
        account_id INTEGER,
        product CHARACTER VARYING(11),
        date_created TIMESTAMP WITHOUT TIME ZONE,
        waive_invoices SMALLINT,
        package_id INTEGER,
        package_name CHARACTER VARYING(150),
        date_next_billing TIMESTAMP WITHOUT TIME ZONE
    )
    - wbx_data_dbt.dim_billing_item (
        id INTEGER,
        account_id INTEGER,
        billing_id INTEGER,
        page_id INTEGER,
        subject_type CHARACTER VARYING(75),
        subject_id BIGINT,
        exclude_from_invoicing SMALLINT,
        pricing_type INTEGER,
        quantity INTEGER,
        total INTEGER,
        currency CHARACTER VARYING(9),
        date_created TIMESTAMP WITHOUT TIME ZONE,
        date_updated TIMESTAMP WITHOUT TIME ZONE
    )
    - wbx_data_dbt.dim_dates (
        full_date DATE,
        month_day_number INTEGER
    );

    - wbx_data_dbt.dim_event_end_dates (
        form_id INTEGER,
        event_end TIMESTAMP WITHOUT TIME ZONE,
        event_end_source CHARACTER VARYING(27)
    );

    - wbx_data_dbt.dim_form (
        id INTEGER,
        account_id INTEGER,
        currency CHARACTER VARYING(9),
        product CHARACTER VARYING(11),
        name CHARACTER VARYING(225),
        date_created TIMESTAMP WITHOUT TIME ZONE,
        event_start TIMESTAMP WITHOUT TIME ZONE,
        event_end TIMESTAMP WITHOUT TIME ZONE
    );
    - dim_deal
    - wbx_data_dbt.dim_gateway (
        id INTEGER,
        type CHARACTER VARYING(150),
        payment_method_provider_id BIGINT,
        gateway_provider_type CHARACTER VARYING(150),
        date_created TIMESTAMP WITHOUT TIME ZONE
    );

    - wbx_data_dbt.dim_invoice (
        id INTEGER,
        billing_id INTEGER,
        account_id INTEGER,
        package_id BIGINT,
        product CHARACTER VARYING(11),
        package_name CHARACTER VARYING(150),
        amount NUMERIC,
        date_created TIMESTAMP WITHOUT TIME ZONE,
        complete_date TIMESTAMP WITHOUT TIME ZONE,
        start_date TIMESTAMP WITHOUT TIME ZONE,
        end_date TIMESTAMP WITHOUT TIME ZONE,
        billing_date TIMESTAMP WITHOUT TIME ZONE,
        status CHARACTER VARYING(150),
        forwarded_to_invoice_id INTEGER,
        forwarded_from_invoice_id BIGINT
    );
    - dim_product
    - wbx_data_dbt.dim_registrant_data (
        registration_id INTEGER,
        account_id INTEGER,
        form_id INTEGER,
        hash CHARACTER VARYING(168),
        registrant_count INTEGER
    );

    - wbx_data_dbt.dim_registration_data (
        id INTEGER,
        account_id INTEGER,
        form_id INTEGER,
        hash CHARACTER VARYING(168),
        total NUMERIC,
        status SMALLINT,
        date_created TIMESTAMP WITHOUT TIME ZONE
    );

    - wbx_data_dbt.dim_ticket_data (
        registration_id INTEGER,
        account_id INTEGER,
        form_id INTEGER,
        hash CHARACTER VARYING(168),
        registration_date DATE,
        last_ticket_date DATE,
        ticket_count INTEGER
    );
    - wbx_data_dbt.dim_transaction (
        id INTEGER,
        account_id INTEGER,
        form_id INTEGER,
        registration_id INTEGER,
        gateway_id INTEGER,
        transaction_type SMALLINT,
        status INTEGER,
        amount NUMERIC,
        amount_refunded NUMERIC,
        payment_method CHARACTER VARYING(384),
        source_type SMALLINT,
        currency CHARACTER VARYING(9),
        app_fee NUMERIC,
        date_created TIMESTAMP WITHOUT TIME ZONE
    );

    Look for examples in the {context}
    on how to use these tables and structure the query.
    If you do not know how to proceed with a specific part,
    state that you do not know how to formulate it,
    but continue the query as far as possible. For example,
    say that a WHERE clause is missing if
    you do not have the context to formulate it.

    {context}
    Question: {question}
    Assistant:
    """
    # QA chain for RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
        },
    )
    st.title("SQL Query Helper")

    # Get user input
    query = st.text_area("Enter your SQL query needs:")

    # Create a button to submit the query
    if st.button("Submit SQL Query"):
        try:
            response = qa_chain({"query": query})
            st.markdown("**Result:**")
            st.markdown(response["result"])
            st.markdown("**Sources:**")
            for i, doc in enumerate(response["source_documents"]):
                with st.expander(f"Source Document {i + 1}"):
                    st.markdown(format_document(doc))
        except Exception as e:
            st.error(f"An error occurred: {e}")


def python_query_assistant():

    QUERY_PROMPT_TEMPLATE = """\
    Human:
    You are a Webconnex Oracle, you know everything about the business,
    confluence github and more,>

    Look for examples in the {context} to have more information to answer the
    question. If you do now how to answer say that but formulate as far

    {context}
    Question: {question}
    Assistant:
    """
    # QA chain for RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
        },
    )
    st.title("Wbx Oracle Assistant")

    # Get user input
    query = st.text_area("Enter your company question:")

    # Create a button to submit the
    if st.button("Submit question"):
        try:
            response = qa_chain({"query": query})
            st.markdown("**Result:**")
            st.markdown(response["result"])
            st.markdown("**Source Documents:**")
            for i, doc in enumerate(response["source_documents"]):
                with st.expander(f"Source Document {i + 1}"):
                    st.markdown(format_document(doc))
        except Exception as e:
            st.error(f"An error occurred: {e}")


def main():
    st.set_page_config(page_title="AI Assistant", layout="wide")

    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f0f2f6;
        }
        .sidebar .sidebar-content {
            background: #f0f2f6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["SQL Query Helper", "Wbx Oracle Assistant"])

    with tabs[0]:
        sql_query_assistant()
    with tabs[1]:
        python_query_assistant()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; padding: 10px;'>"
        "<small>Information is accurate as of May 21, 2024</small>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
