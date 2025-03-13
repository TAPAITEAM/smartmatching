import os
import json

DEFAULT_VALUE = "N/A"
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_0_FLASH_LITE  = "gemini-2.0-flash-lite"
GEMINI_2_0_PRO = "gemini-2.0-pro-exp-02-05"
GEMINI_TEXT_EMBEDDING_004 = "models/text-embedding-004"

# Langchain and AI libraries
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable

# Import your functions
from utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, clean_text
from prompt import PROJECT_SUMMARY_PROMPT, CONSULTANT_MATCH_PROMPT, AI_CHAT_PROMPT

# Streamlit UI setup
import streamlit as st

# Set API keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["general"]["GOOGLE_API_KEY"]
# os.environ["OPENAI_API_KEY"] = st.secrets["general"]["OPENAI_API_KEY"]
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
def generate_project_summary(text, model=GEMINI_2_0_FLASH, prompt=PROJECT_SUMMARY_PROMPT):
    """Generate structured project summary using AI"""
    text = clean_text(text)
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
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
            'Full Name', 'Email', 'Finance Expertise', 'Light Finance', 'Strategy Expertise',
            'Entrepreneurship Expertise', 'Operations Expertise', 'Marketing Expertise',
            'Areas of Expertise', 'Industry Expertise', 'Other Industry Expertise', 
            'Other Areas of Expertise', 'Other Skills', 'Consultant Availability Status', 
            'Anticipated Availability Date', 'Availability Notes', 'Onboarding Notes', 'Staffing Insights'
        ]

        # Validate DataFrame columns
        missing_columns = set(columns) - set(df.columns)
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            return None

        # Normalize column names in DataFrame
        df.columns = [col.strip() for col in df.columns]

        # Create combined columns
        df['Area Skills'] = df['Areas of Expertise'].fillna('') + '; ' + \
                          df['Other Areas of Expertise'].fillna('') + '; ' + \
                          df['Other Skills'].fillna('')
        
        df['Industry Skills'] = df['Industry Expertise'].fillna('') + '; ' + \
                              df['Other Industry Expertise'].fillna('')
        
        df['Comments'] = df['Onboarding Notes'].fillna('') + '; ' + \
                        df['Staffing Insights'].fillna('')

        # Create text data by combining all fields
        text_data = [
            f"Full Name: {fullname}; "
            f"Email: {email}; "
            f"Finance Expertise: {financeexpertise}; "
            f"Light Finance: {lightfinance}; "
            f"Strategy Expertise: {strategyexpertise}; "
            f"Entrepreneurship Expertise: {entrepreneurshipexpertise}; "
            f"Operations Expertise: {operationsexpertise}; "
            f"Marketing Expertise: {marketingexpertise}; "
            f"Area Skills: {areaskills}; "
            f"Industry Skills: {industryskills}; "
            f"Consultant Availability Status: {consultantavailabilitystatus}; "
            f"Anticipated Availability Date: {anticipatedavailabilitydate}; "
            f"Availability Notes: {availabilitynotes}; "
            f"Comments: {comments}; "
            for (
                fullname, email, financeexpertise, lightfinance, strategyexpertise,
                entrepreneurshipexpertise, operationsexpertise, marketingexpertise,
                areaskills, industryskills, consultantavailabilitystatus,
                anticipatedavailabilitydate, availabilitynotes, comments
            ) in zip(*[df[col] for col in [
                'Full Name', 'Email', 'Finance Expertise', 'Light Finance', 'Strategy Expertise',
                'Entrepreneurship Expertise', 'Operations Expertise', 'Marketing Expertise',
                'Area Skills', 'Industry Skills', 'Consultant Availability Status',
                'Anticipated Availability Date', 'Availability Notes', 'Comments'
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
def analyze_consultant_match(project_summary, consultant_details, model=GEMINI_2_0_FLASH, prompt=CONSULTANT_MATCH_PROMPT):
    """Generate detailed analysis of consultant match"""
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
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
def find_best_consultant_matches(vector_store, project_summary, model=GEMINI_2_0_FLASH, top_k=5):
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
            match_analysis = analyze_consultant_match(project_summary, consultant_details, model=model)
            matches.append({
                "Full Name": result.metadata.get('Full Name', DEFAULT_VALUE),
                "Email": result.metadata.get('Email', DEFAULT_VALUE),
                "Finance Expertise": result.metadata.get('Finance Expertise', DEFAULT_VALUE),
                "Light Finance": result.metadata.get('Light Finance', DEFAULT_VALUE),
                "Strategy Expertise": result.metadata.get('Strategy Expertise', DEFAULT_VALUE),
                "Entrepreneurship Expertise": result.metadata.get('Entrepreneurship Expertise', DEFAULT_VALUE),
                "Operations Expertise": result.metadata.get('Operations Expertise', DEFAULT_VALUE),
                "Marketing Expertise": result.metadata.get('Marketing Expertise', DEFAULT_VALUE),
                "Area Skills": result.metadata.get('Area Skills', DEFAULT_VALUE),
                "Industry Skills": result.metadata.get('Industry Skills', DEFAULT_VALUE),
                "Consultant Availability Status": result.metadata.get('Consultant Availability Status', DEFAULT_VALUE),
                "Anticipated Availability Date": result.metadata.get('Anticipated Availability Date', DEFAULT_VALUE),
                "Availability Notes": result.metadata.get('Availability Notes', DEFAULT_VALUE),
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

# Chat with consultant database - Enhanced version
@traceable()
def chat_with_consultant_database(prompt, vector_store, df, model=GEMINI_2_0_FLASH):
    """
    Chat with the consultant database with improved handling of column-based queries.
    This enhanced version can better answer questions about consultants with multiple criteria.
    """
    # Initialize messages if not provided
    session_messages = [] 

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Step 1: Analyze the query to detect if it's a criteria-based search
    query_analysis_prompt = f"""
    Analyze this query about consultants and determine if it's asking for consultants matching multiple criteria.
    Query: "{prompt}"
    
    IMPORTANT: Your response must be ONLY valid JSON with no additional text, explanations, or formatting.
    
    If the query is asking for consultants with multiple criteria (like "both X and Y expertise" or "expertise in Z and industry W"),
    extract these criteria and set "is_criteria_search" to true. Otherwise, set it to false.
    
    Return ONLY this JSON structure:
    {{
        "is_criteria_search": true/false,
        "criteria": [
            {{"field": "field_name", "value": "required_value"}},
            ...
        ]
    }}
    
    Focus only on expertise fields (e.g., finance, marketing), industry expertise, and availability.
    For queries about 'expertise in [area]', treat [area] as the field (e.g., "finance", "marketing") and "expertise" as the value.
    
    Examples:
    1. For "Find consultants with finance expertise who know healthcare industry":
    {{"is_criteria_search": true, "criteria": [{{"field": "finance", "value": "expertise"}}, {{"field": "industry", "value": "healthcare"}}]}}
    
    2. For "Get me consultants who have expertise in both finance and marketing":
    {{"is_criteria_search": true, "criteria": [{{"field": "finance", "value": "expertise"}}, {{"field": "marketing", "value": "expertise"}}]}}

    3. For "Who is available next month?":
    {{"is_criteria_search": true, "criteria": [{{"field": "Consultant Availability Status", "value": "available"}}]}}

    4. For "Tell me about the consultant database":
    {{"is_criteria_search": false, "criteria": []}}
    """
    
    try:
        # Analyze the query first
        analysis_response = llm.invoke(query_analysis_prompt)
        
        # More robust JSON parsing - attempt to extract JSON from the response
        response_text = analysis_response.content
        
        # Try to find JSON content within the response (in case the model adds explanatory text)
        import re
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        
        if json_match:
            try:
                query_analysis = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # If the extracted JSON-like text is invalid, use default values
                query_analysis = {"is_criteria_search": False, "criteria": []}
        else:
            # No JSON-like structure found, use default values
            query_analysis = {"is_criteria_search": False, "criteria": []}
        
        # Check if this is a criteria-based search
        if query_analysis.get("is_criteria_search", False) and query_analysis.get("criteria", []):
            # Step 2: For criteria-based searches, retrieve a larger initial result set
            vector_results = vector_store.similarity_search(prompt, k=50)  # Get more initial results
            # print(f"Initial vector results count: {len(vector_results)}")

            # Step 3: Filter the results based on the extracted criteria
            filtered_results = []
            criteria_fields = [(item["field"], item["value"]) for item in query_analysis["criteria"]]
            
            for doc in vector_results:
                matches_all_criteria = True
                
                for field, value in criteria_fields:
                    # print(f"Checking criteria: {field}={value}")
                    # Map the field name to the actual metadata field name if needed
                    field_mapping = {
                        "finance": "Finance Expertise",
                        "strategy": "Strategy Expertise",
                        "operations": "Operations Expertise",
                        "marketing": "Marketing Expertise",
                        "entrepreneurship": "Entrepreneurship Expertise",
                        # Add more mappings as needed
                    }                    
                    actual_field = field_mapping.get(field.lower(), field)
                    # print(f"Mapped to: {actual_field}")
                    # Check if the metadata field contains the required value
                    if actual_field in doc.metadata:
                        field_value = str(doc.metadata[actual_field]).lower()
                        # print(f"Found in metadata: {field_value}")
                        # Determine if the search is looking for a positive match
                        is_looking_for_positive = value.lower() in ["high", "yes", "true", "expertise"]

                        # Determine if the field has a positive value
                        has_positive_value = field_value in ["yes", "high", "medium", "true", "1", "1.0"]

                        # If we're looking for positive but field is not positive, OR
                        # If we're looking for negative but field is positive
                        if (is_looking_for_positive and not has_positive_value) or \
                        (not is_looking_for_positive and has_positive_value):
                            # print(f"Does not match: {value} != {field_value}")
                            matches_all_criteria = False
                            break

                    elif "Industry Skills" in doc.metadata and "industry" in field.lower():
                        # Special case for industry-related queries
                        industry_skills = str(doc.metadata["Industry Skills"]).lower()
                        if value.lower() not in industry_skills:
                            matches_all_criteria = False
                            break
                    elif "Area Skills" in doc.metadata and ("expertise" in field.lower() or "skills" in field.lower()):
                        # Special case for expertise-related queries
                        area_skills = str(doc.metadata["Area Skills"]).lower()
                        if value.lower() not in area_skills:
                            matches_all_criteria = False
                            break
                    else:
                        # If we can't find the field, assume it doesn't match
                        matches_all_criteria = False
                        break
                
                if matches_all_criteria:
                    filtered_results.append(doc)
            
            # Use the filtered results if we have any, otherwise fall back to regular search
            if filtered_results:
                vector_results = filtered_results[:20]  # Limit to top 5
                
                # Create a special context highlighting that these results match ALL criteria
                context = "Consultants matching ALL specified criteria:\n\n" + \
                          "\n\n---\n\n".join([doc.page_content for doc in vector_results])
            else:
                # If no exact matches, fall back to regular search with a note
                vector_results = vector_store.similarity_search(prompt, k=10)
                context = "No consultants match ALL specified criteria exactly. Here are the closest matches:\n\n" + \
                          "\n\n---\n\n".join([doc.page_content for doc in vector_results])
        else:
            # Step 4: For regular queries, proceed with standard similarity search
            vector_results = vector_store.similarity_search(prompt, k=10)
            context = "\n\n---\n\n".join([doc.page_content for doc in vector_results])
        
        # Step 5: Generate the response using the context and conversation history
        session_messages = ' '.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
        
        ai_prompt = AI_CHAT_PROMPT.format(
            context=context,
            prompt=prompt,
            session_messages=session_messages
        )
        
        response = llm.invoke(ai_prompt)
        return response.content
        
    except json.JSONDecodeError:
        # This should rarely happen now with our improved parsing
        st.warning("⚠️ Using standard search instead of criteria-based search")
        vector_results = vector_store.similarity_search(prompt, k=10)
        context = "\n".join([doc.page_content for doc in vector_results])
        
        ai_prompt = AI_CHAT_PROMPT.format(
            context=context,
            prompt=prompt,
            session_messages=session_messages
        )
        
        response = llm.invoke(ai_prompt)
        return response.content
        
    except (KeyError, ValueError) as e:
        return f"Sorry, I encountered an error while processing your query: {e}"
    except Exception as e:
        return f"Sorry, I encountered an unexpected error: {e}"
