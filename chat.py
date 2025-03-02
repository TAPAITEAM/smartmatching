from typing import List, Optional
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
import streamlit as st

# Constants
GEMINI_2_0_FLASH = "gemini-2.0-flash"
GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
GEMINI_2_0_PRO = "gemini-2.0-pro-exp-02-05"

# Pydantic model for query analysis
class Criterion(BaseModel):
    """A single search criterion for consultant search."""
    field: str = Field(description="The field to search on (e.g., 'finance', 'marketing', 'strategy', 'operations', 'entrepreneurship', 'industry')")
    value: str = Field(description="The required value for this field (e.g., 'expertise', 'healthcare')")

class QueryAnalysis(BaseModel):
    """Analysis of a consultant database query to determine if it's criteria-based."""
    is_criteria_search: bool = Field(
        description="Whether the query is asking for consultants matching multiple criteria"
    )
    criteria: List[Criterion] = Field(
        default_factory=list,
        description="List of criteria extracted from the query"
    )

# State model for the LangGraph
class ConsultantQueryState(BaseModel):
    """State for the consultant query processing flow."""
    query: str
    analysis: Optional[QueryAnalysis] = None
    filtered_results: List = Field(default_factory=list)
    context: str = ""
    response: str = ""
    session_messages: str = ""

# Function to initialize the LLM
def get_llm(model=GEMINI_2_0_FLASH, temperature=0):
    """Initialize and return the LLM based on the specified model.
    
    Args:
        model (str): The Gemini model to use, one of:
            - GEMINI_2_0_FLASH ("gemini-2.0-flash")
            - GEMINI_2_0_FLASH_LITE ("gemini-2.0-flash-lite")
            - GEMINI_2_0_PRO ("gemini-2.0-pro-exp-02-05")
        temperature (float): Controls randomness in the output (0.0 to 1.0)
            
    Returns:
        ChatGoogleGenerativeAI: The initialized LLM instance
    """
    # Map from the frontend model options to the actual model IDs
    model_mapping = {
        "Gemini 2.0 Flash": GEMINI_2_0_FLASH,
        "Gemini 2.0 Flash Lite": GEMINI_2_0_FLASH_LITE,
        "Gemini 2.0 Pro Experimental": GEMINI_2_0_PRO,
        # Also include the direct model IDs for backward compatibility
        GEMINI_2_0_FLASH: GEMINI_2_0_FLASH,
        GEMINI_2_0_FLASH_LITE: GEMINI_2_0_FLASH_LITE,
        GEMINI_2_0_PRO: GEMINI_2_0_PRO
    }
    
    # If a friendly name was passed, convert to the actual model ID
    model_id = model_mapping.get(model, model)
    
    # Validate model selection
    valid_models = [GEMINI_2_0_FLASH, GEMINI_2_0_FLASH_LITE, GEMINI_2_0_PRO]
    if model_id not in valid_models:
        print(f"Warning: Model {model} not in supported list. Defaulting to {GEMINI_2_0_FLASH}")
        model_id = GEMINI_2_0_FLASH
    
    return ChatGoogleGenerativeAI(
        model=model_id,
        temperature=temperature,
        api_key=os.environ["GOOGLE_API_KEY"]
    )

# Graph nodes
def analyze_query(state: ConsultantQueryState) -> ConsultantQueryState:
    """Analyze the query to detect if it's a criteria-based search."""
    llm = get_llm(temperature=0)
    
    # Create a structured output model for the LLM
    structured_llm = llm.with_structured_output(QueryAnalysis)
    
    query_analysis_prompt = f"""
    Analyze this query about consultants and determine if it's asking for consultants matching multiple criteria.
    Query: "{state.query}"
    
    Focus only on expertise fields (e.g., finance, marketing), industry expertise, and availability.
    For queries about 'expertise in [area]', treat [area] as the field (e.g., "finance", "marketing") and "expertise" as the value.
    
    Examples:
    1. "Find consultants with finance expertise who know healthcare industry" should extract:
       - field: "finance", value: "expertise"
       - field: "industry", value: "healthcare"
    
    2. "Get me consultants who have expertise in both finance and marketing" should extract:
       - field: "finance", value: "expertise"
       - field: "marketing", value: "expertise"

    3. "Who is available next month?" should extract:
       - field: "Consultant Availability Status", value: "available"

    4. "Tell me about the consultant database" should NOT be a criteria search.
    """
    
    # Get structured output from the LLM
    analysis = structured_llm.invoke(query_analysis_prompt)
    state.analysis = analysis
    
    return state

def search_consultants(state: ConsultantQueryState, vector_store) -> ConsultantQueryState:
    """Search for consultants based on the query analysis."""
    # If this is a criteria-based search with criteria
    if state.analysis and state.analysis.is_criteria_search and state.analysis.criteria:
        # Step the original vector search to get an initial result set
        vector_results = vector_store.similarity_search(state.query, k=30)
        
        # Filter the results based on the extracted criteria
        filtered_results = []
        criteria_fields = [(item.field, item.value) for item in state.analysis.criteria]
        
        for doc in vector_results:
            matches_all_criteria = True
            
            for field, value in criteria_fields:
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
                
                # Check if the metadata field contains the required value
                if actual_field in doc.metadata:
                    field_value = str(doc.metadata[actual_field]).lower()
                    # Determine if the search is looking for a positive match
                    is_looking_for_positive = value.lower() in ["high", "yes", "true", "expertise"]

                    # Determine if the field has a positive value
                    has_positive_value = field_value in ["yes", "high", "medium", "true", "1", "1.0"]

                    # If we're looking for positive but field is not positive, OR
                    # If we're looking for negative but field is positive
                    if (is_looking_for_positive and not has_positive_value) or \
                       (not is_looking_for_positive and has_positive_value):
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
            state.filtered_results = filtered_results[:20]  # Limit to top 20
            
            # Create a special context highlighting that these results match ALL criteria
            state.context = "Consultants matching ALL specified criteria:\n\n" + \
                      "\n\n---\n\n".join([doc.page_content for doc in state.filtered_results])
        else:
            # If no exact matches, fall back to regular search with a note
            fallback_results = vector_store.similarity_search(state.query, k=10)
            state.filtered_results = fallback_results
            state.context = "No consultants match ALL specified criteria exactly. Here are the closest matches:\n\n" + \
                      "\n\n---\n\n".join([doc.page_content for doc in fallback_results])
    else:
        # For regular queries, proceed with standard similarity search
        regular_results = vector_store.similarity_search(state.query, k=10)
        state.filtered_results = regular_results
        state.context = "\n\n---\n\n".join([doc.page_content for doc in regular_results])
    
    return state

def generate_response(state: ConsultantQueryState) -> ConsultantQueryState:
    """Generate a response using the context and conversation history."""
    llm = get_llm(temperature=0)
    
    # Use the AI chat prompt template
    ai_prompt = f"""
    You are an AI assistant helping users find consultants in a database. 
    Answer the following query based on the context provided below.
    
    Context:
    {state.context}
    
    Recent conversation:
    {state.session_messages}
    
    User Query: {state.query}
    
    Provide a clear, concise answer that directly addresses the user's question.
    If multiple consultants match the criteria, summarize their key qualifications.
    If no consultants match exactly, suggest the closest matches and explain why.
    """
    
    response = llm.invoke(ai_prompt)
    state.response = response.content
    
    return state

def chat_with_consultant_database(prompt, vector_store, df, model=GEMINI_2_0_FLASH):
    """
    Chat with the consultant database with improved handling of column-based queries.
    This enhanced version uses LangGraph and structured outputs.
    """
    try:
        # Get session messages if in Streamlit
        session_messages = ""
        if hasattr(st, 'session_state') and 'messages' in st.session_state:
            session_messages = ' '.join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]])
        
        # Initialize the state
        initial_state = ConsultantQueryState(
            query=prompt,
            session_messages=session_messages
        )
        
        # Define the LangGraph
        workflow = StateGraph(ConsultantQueryState)
        
        # Add nodes
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("search_consultants", lambda state: search_consultants(state, vector_store))
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.add_edge("analyze_query", "search_consultants")
        workflow.add_edge("search_consultants", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("analyze_query")
        
        # Compile the graph
        app = workflow.compile()
        
        # Run the graph
        result = app.invoke(initial_state)
        
        # In LangGraph, the result is a dictionary of the final state
        # We need to access the response attribute from the state
        if hasattr(result, 'response'):
            return result.response
        elif isinstance(result, dict) and 'state' in result and hasattr(result['state'], 'response'):
            return result['state'].response
        elif isinstance(result, dict) and 'response' in result:
            return result['response']
        else:
            # Try to access the last node's output directly
            nodes = list(result.keys())
            if nodes and 'generate_response' in nodes:
                return result['generate_response'].response
            
            # Fallback to returning a diagnostic message
            return f"Query processed, but couldn't extract response. Result structure: {type(result)}"
        
    except Exception as e:
        return f"Sorry, I encountered an unexpected error: {str(e)}"

