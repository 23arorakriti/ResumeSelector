import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables.")
    st.stop()

# Initialize the Gemini model
llm = GoogleGenerativeAI(google_api_key=api_key, model='gemini-1.5-flash', temperature=0.9)

# Embeddings for building the vector DB and for retrieval queries
instructor_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_query"
)

# Pre-defined CSV file path
csv_file_path = "project.csv"
vectordb_file_path = "faiss_index"

def create_vector_db_from_csv():
    """Loads CSV data and creates a FAISS vector database."""
    loader = CSVLoader(file_path=csv_file_path, source_column="Job Type")
    data = loader.load()
    
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)
    st.success("Vector database created from CSV.")

def load_csv_context():
    """Loads FAISS vector store and retrieves all CSV job type context."""
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7, top_k=10)
    csv_docs = retriever.get_relevant_documents("Job types and skills required")
    return "\n".join([doc.page_content for doc in csv_docs])

# Streamlit UI
st.title("Resume Analysis")

# Ensure vector DB is created
if not os.path.exists(vectordb_file_path):
    create_vector_db_from_csv()

# PDF Upload
uploaded_pdf = st.file_uploader("Upload a Resume (PDF)", type=["pdf"])
if uploaded_pdf:
    pdf_path = f"temp_{uploaded_pdf.name}"
    
    # Save the uploaded PDF temporarily
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    # Load the PDF content
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    pdf_text = "\n".join([doc.page_content for doc in pdf_docs])

    # Retrieve CSV job data
    csv_context = load_csv_context()

    # Prompt template
    prompt_template = """
    You are an expert recruiter.
    
    Below is context extracted from a job requirement dataset:
    {csv_context}

    Below is additional content extracted from a resume document:
    {pdf_context}

    Your job is to analyze the resume and determine if the candidate is suitable for a Software Engineering job. 
    - If YES, explain why and highlight important skills.  
    - If NO, suggest which other job types they are suitable for.  
    - Use the CSV data to refer to different skills associated with job types.  
    - Highlight key details from the resume for HR review.
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["csv_context", "pdf_context"])
    full_prompt = prompt.format(csv_context=csv_context, pdf_context=pdf_text)

    # Generate response
    answer = llm(full_prompt)

    # Display answer
    st.subheader("Resume Analysis Result:")
    st.write(answer)
