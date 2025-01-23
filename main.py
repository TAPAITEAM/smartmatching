import os

# Langchain and AI libraries
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable

# Import your functions
from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, clean_text
from prompt import PROJECT_SUMMARY_PROMPT, CONSULTANT_MATCH_PROMPT, AI_CHAT_PROMPT

# Streamlit UI setup
import streamlit as st

# Set API keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["general"]["GOOGLE_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]["API_KEY"]
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Project-Consultant-Matcher-TAP"

# Initialize embeddings
@st.cache_resource
@traceable(
    metadata={"embedding_model": "gemini/text-embedding-004"},
)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Load consultant data
def process_uploaded_file(uploaded_file):
    """Process the uploaded file based on its type."""
    file_text = ""
    if uploaded_file.type == "application/pdf":
        file_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_text = extract_text_from_docx(uploaded_file)
    elif uploaded_file.type == "text/plain":
        file_text = extract_text_from_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file type")
    return file_text

# Project summary function using AI
@traceable()
def generate_project_summary(text, prompt=PROJECT_SUMMARY_PROMPT):
    """Generate structured project summary using AI"""
    text = clean_text(text)
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0.2,
        api_key=os.environ["GOOGLE_API_KEY"]
    )
    try:
        response = llm.invoke(prompt.format(text=text))
        return response.content
    except Exception as e:
        st.error(f"❌ Error generating project summary: {e}")
        return "Unable to generate summary."

# Create vector store for consultants
@st.cache_resource
@traceable(
    metadata={"vectordb": "FAISS"}
)
def create_consultant_vector_store(_embeddings, df):
    if df is None or df.empty:
        st.error("❌ Consultant DataFrame is None or empty")
        return None
    try:
        df.columns = ['Fullname', 'Financexpertise', 'Lightfinance', 'Strategyexpertise', 'Entrepreneurshipexpertise', 'Operationsexpertise', 'Marketingexpertise', 'Areasofexpertise', 'Industryexpertise', 'Consultantavailabilitystatus', 'Anticipatedavailabilitydate']
        # available_df = df[df['Availability'].str.lower().isin(['yes', 'y', 'available'])]
        text_data = [
            f"Fullname: {fullname}; Financexpertise: {financexpertise}; Lightfinance: {lightfinance}; Strategyexpertise: {strategyexpertise}; Entrepreneurshipexpertise: {entrepreneurshipexpertise}; Operationsexpertise: {operationsexpertise}; Marketingexpertise:{marketingexpertise}; Areasofexpertise: {areasofexpertise}; Industryexpertise: {industryexpertise}; Consultantavailabilitystatus: {consultantavailabilitystatus}; Anticipatedavailabilitydate: {anticipatedavailabilitydate}"
            for fullname, financexpertise, lightfinance, strategyexpertise, entrepreneurshipexpertise, operationsexpertise, marketingexpertise, areasofexpertise, industryexpertise, consultantavailabilitystatus, anticipatedavailabilitydate in zip(
                df['Fullname'], df['Financexpertise'], df['Lightfinance'], df['Strategyexpertise'], df['Entrepreneurshipexpertise'], df['Operationsexpertise'], df['Marketingexpertise'], df['Areasofexpertise'], df['Industryexpertise'], df['Consultantavailabilitystatus'], df['Anticipatedavailabilitydate']
            )
        ]
        metadatas = df.to_dict('records')
        vector_store = FAISS.from_texts(
            texts=text_data, 
            embedding=_embeddings,
            metadatas=metadatas
        )
        return vector_store
    except Exception as e:
        st.error(f"❌ Error creating consultant vector store: {e}")
        return None

# Analyze consultant match with AI
@traceable()
def analyze_consultant_match(project_summary, consultant_details, prompt=CONSULTANT_MATCH_PROMPT):
    """Generate detailed analysis of consultant match"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0.2,
        api_key=os.environ["GOOGLE_API_KEY"]
    )
    try:
        response = llm.invoke(prompt.format(project_summary=project_summary, consultant_details=consultant_details))
        return response.content
    except Exception as e:
        st.error(f"❌ Error analyzing consultant match: {e}")
        return "Unable to generate detailed match analysis."

# Find best consultant matches
@traceable(
        run_type="retriever"
)
def find_best_consultant_matches(vector_store, project_summary, top_k=5):
    """Find the best consultant matches based on project summary"""
    if not vector_store:
        return []
    try:
        results = vector_store.similarity_search(project_summary, k=top_k)
        matches = []
        for result in results:
            consultant_details = "\n".join([
                f"{key}: {value}" for key, value in result.metadata.items()
            ])
            match_analysis = analyze_consultant_match(project_summary, consultant_details)
            matches.append({
                "Fullname": result.metadata.get('Fullname', 'N/A'),
                "Financexpertise": result.metadata.get('Financexpertise', 'N/A'),
                "Lightfinance": result.metadata.get('Lightfinance', 'N/A'),
                "Strategyexpertise": result.metadata.get('Strategyexpertise', 'N/A'),
                "Entrepreneurshipexpertise": result.metadata.get('Entrepreneurshipexpertise', 'N/A'),
                "Operationsexpertise": result.metadata.get('Operationsexpertise', 'N/A'),
                "Marketingexpertise": result.metadata.get('Marketingexpertise', 'N/A'),
                "Areasofexpertise": result.metadata.get('Areasofexpertise', 'N/A'),
                "Industryexpertise": result.metadata.get('Industryexpertise', 'N/A'),
                "Consultantavailabilitystatus": result.metadata.get('Consultantavailabilitystatus', 'N/A'),
                "Anticipatedavailabilitydate": result.metadata.get('Anticipatedavailabilitydate', 'N/A'),
                "Match Analysis": match_analysis
            })
        return matches
    except Exception as e:
        st.error(f"❌ Error finding consultant matches: {e}")
        return []

# Chat with consultant database
@traceable()
def chat_with_consultant_database(prompt, vector_store):
    """Chat with the consultant database."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0.2,
        api_key=os.environ["GOOGLE_API_KEY"]
    )

    vector_results = vector_store.similarity_search(prompt, k=5)
    context = "\n".join([doc.page_content for doc in vector_results])

    session_messages = ' '.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])

    ai_prompt = AI_CHAT_PROMPT.format(
            context=context,
            prompt=prompt,
            session_messages=session_messages
        )
    try:
        response = llm.invoke(ai_prompt)
        return response.content
    except Exception as e:
         return f"Sorry, I encountered an error: {e}"
