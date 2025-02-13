import os

DEFAULT_VALUE = "N/A"
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_TEXT_EMBEDDING_004 = "models/text-embedding-004"

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
    return GoogleGenerativeAIEmbeddings(model=GEMINI_TEXT_EMBEDDING_004)

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
        model=GEMINI_2_0_FLASH,
        temperature=0.2,
        api_key=os.environ["GOOGLE_API_KEY"]
    )
    try:
        response = llm.invoke(prompt.format(text=text))
        return response.content
    except KeyError as e:
        st.error(f"❌ API key error: Please check if GOOGLE_API_KEY is set correctly")
        return "Unable to generate summary: Missing API key"
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return f"Unable to generate summary: {str(e)}"

# Create vector store for consultants
@st.cache_resource
@traceable(
    metadata={"vectordb": "FAISS"}
)
def create_consultant_vector_store(_embeddings, df):
    """Create a vector store for consultants"""
    # Validate input parameters
    if _embeddings is None:
        st.error("❌ Embeddings model is not initialized")
        return None
        
    if df is None or df.empty:
        st.error("❌ Consultant DataFrame is empty or not provided")
        return None

    try:
        # Define column names
        columns = [
            'Fullname', 'Email', 'Financeexpertise', 'Lightfinance', 'Strategyexpertise',
            'Entrepreneurshipexpertise', 'Operationsexpertise', 'Marketingexpertise',
            'Areasofexpertise', 'Industryexpertise', 'Otherindustryexpertise', 
            'Otherareasofexpertise', 'Otherskills', 'Consultantavailabilitystatus', 
            'Anticipatedavailabilitydate', 'Availabilitynotes', 'Onboardingnotes', 'Staffinginsights'
        ]

        # Add debug logging to check actual DataFrame columns
        # st.write("Debug - DataFrame columns:", df.columns.tolist())

        # Validate DataFrame columns
        missing_columns = set(columns) - set(df.columns)
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None

        # Normalize column names in DataFrame
        df.columns = [col.strip() for col in df.columns]

        # Create combined columns
        df['Areaskills'] = df['Areasofexpertise'].fillna('') + '; ' + \
                          df['Otherareasofexpertise'].fillna('') + '; ' + \
                          df['Otherskills'].fillna('')
        
        df['Industryskills'] = df['Industryexpertise'].fillna('') + '; ' + \
                              df['Otherindustryexpertise'].fillna('')
        
        df['Comments'] = df['Onboardingnotes'].fillna('') + '; ' + \
                        df['Staffinginsights'].fillna('')

        # Create text data by combining all fields
        text_data = [
            f"Fullname: {fullname}; "
            f"Email: {email}; "
            f"Financeexpertise: {financeexpertise}; "
            f"Lightfinance: {lightfinance}; "
            f"Strategyexpertise: {strategyexpertise}; "
            f"Entrepreneurshipexpertise: {entrepreneurshipexpertise}; "
            f"Operationsexpertise: {operationsexpertise}; "
            f"Marketingexpertise: {marketingexpertise}; "
            f"Areaskills: {areaskills}; "
            f"Industryskills: {industryskills}; "
            f"Consultantavailabilitystatus: {consultantavailabilitystatus}; "
            f"Anticipatedavailabilitydate: {anticipatedavailabilitydate}; "
            f"Availabilitynotes: {availabilitynotes}; "
            f"Comments: {comments}; "
            for (
                fullname, email, financeexpertise, lightfinance, strategyexpertise,
                entrepreneurshipexpertise, operationsexpertise, marketingexpertise,
                areaskills, industryskills, consultantavailabilitystatus,
                anticipatedavailabilitydate, availabilitynotes, comments
            ) in zip(*[df[col] for col in [
                'Fullname', 'Email', 'Financeexpertise', 'Lightfinance', 'Strategyexpertise',
                'Entrepreneurshipexpertise', 'Operationsexpertise', 'Marketingexpertise',
                'Areaskills', 'Industryskills', 'Consultantavailabilitystatus',
                'Anticipatedavailabilitydate', 'Availabilitynotes', 'Comments'
            ]])
        ]

        # Convert DataFrame to dictionary for metadata
        metadatas = df.to_dict('records')

        # Create vector store
        vector_store = FAISS.from_texts(
            texts=text_data, 
            embedding=_embeddings,
            metadatas=metadatas
        )
        return vector_store
    
    except (KeyError, ValueError) as e:
        st.error(f"❌ Column access error: Unable to access column {str(e)}")
        return None
    
    except Exception as e:
        st.error("❌ Unexpected error during vector store creation")
        st.error(f"Details: {str(e)}")
        return None


# Analyze consultant match with AI
@traceable()
def analyze_consultant_match(project_summary, consultant_details, prompt=CONSULTANT_MATCH_PROMPT):
    """Generate detailed analysis of consultant match"""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_2_0_FLASH, 
        temperature=0.2,
        api_key=os.environ["GOOGLE_API_KEY"]
    )
    try:
        response = llm.invoke(prompt.format(project_summary=project_summary, consultant_details=consultant_details))
        return response.content
    except (KeyError, ValueError) as e:
        st.error(f"❌ Error analyzing consultant match: {e}")
        return "Unable to generate detailed match analysis."
    except Exception as e:
        st.error(f"❌ Unexpected error analyzing consultant match: {e}")
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
                "Fullname": result.metadata.get('Fullname', DEFAULT_VALUE),
                "Email": result.metadata.get('Email', DEFAULT_VALUE),
                "Financeexpertise": result.metadata.get('Financeexpertise', DEFAULT_VALUE),
                "Lightfinance": result.metadata.get('Lightfinance', DEFAULT_VALUE),
                "Strategyexpertise": result.metadata.get('Strategyexpertise', DEFAULT_VALUE),
                "Entrepreneurshipexpertise": result.metadata.get('Entrepreneurshipexpertise', DEFAULT_VALUE),
                "Operationsexpertise": result.metadata.get('Operationsexpertise', DEFAULT_VALUE),
                "Marketingexpertise": result.metadata.get('Marketingexpertise', DEFAULT_VALUE),
                "Areaskills": result.metadata.get('Areaskills', DEFAULT_VALUE),
                "Industryskills": result.metadata.get('Industryskills', DEFAULT_VALUE),
                "Consultantavailabilitystatus": result.metadata.get('Consultantavailabilitystatus', DEFAULT_VALUE),
                "Anticipatedavailabilitydate": result.metadata.get('Anticipatedavailabilitydate', DEFAULT_VALUE),
                "Availabilitynotes": result.metadata.get('Availabilitynotes', DEFAULT_VALUE),
                "Comments": result.metadata.get('Comments', DEFAULT_VALUE),
                "Match Analysis": match_analysis
            })
        return matches
    except (KeyError, ValueError) as e:
        st.error(f"❌ Error finding consultant matches: {e}")
        return []
    except Exception as e:
        st.error(f"❌ Unexpected error finding consultant matches: {e}")
        return []

# Chat with consultant database
@traceable()
def chat_with_consultant_database(prompt, vector_store):
    """Chat with the consultant database."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_2_0_FLASH, 
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
    except (KeyError, ValueError) as e:
         return f"Sorry, I encountered an error: {e}"
    except Exception as e:
         return f"Sorry, I encountered an unexpected error: {e}"
