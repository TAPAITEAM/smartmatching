import streamlit as st
from main import (
    generate_project_summary,
    get_embeddings,
    create_consultant_vector_store,
    find_best_consultant_matches,
    process_uploaded_file,
    # chat_with_consultant_database
)
from chat import chat_with_consultant_database
from utils import check_password, save_feedback, load_consultant_data

# Streamlit UI setup
st.set_page_config(page_title="SmartMatch Staffing Platform", layout="wide", page_icon="🤝")

# Setup sidebar with instructions and feedback form
def setup_sidebar():
    """Setup the sidebar with instructions and feedback form."""
    st.sidebar.header("🤝 SmartMatch Staffing Platform")
    st.sidebar.markdown(
        "This app helps you find suitable consultants for your project based on "
        "project description and consultant expertise."
    )
    
    st.sidebar.write("### Instructions")
    st.sidebar.write(
        "1. :key: Enter password to access the app\n"
        "2. :pencil: Upload project description or use Text Query\n"
        "3. :mag: Review matched consultants and analyses\n"
        "4. :speech_balloon: Chat with our consultant database"
    )

    # Feedback section
    if 'feedback' not in st.session_state:
        st.session_state.feedback = ""

    st.sidebar.markdown("---")
    st.sidebar.subheader("💭 Feedback")
    feedback = st.sidebar.text_area(
        "Share your thoughts",
        value=st.session_state.feedback,
        placeholder="Your feedback helps us improve..."
    )

    if st.sidebar.button("📤 Submit Feedback"):
        if feedback:
            try:
                save_feedback(feedback)
                st.session_state.feedback = ""
                st.sidebar.success("✨ Thank you for your feedback!")
            except Exception as e:
                st.sidebar.error(f"❌ Error saving feedback: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please enter feedback before submitting")

    st.sidebar.image("assets/TAP01.jpg", use_container_width=True)
      

# Main Streamlit app
def main():
    """Main application function."""
    setup_sidebar()
    
    if not check_password():
        st.stop()

    st.title("🤝 Project-Consultant Matcher")

    # Add model selection dropdown
    model_options = {
        "Gemini 2.0 Flash": "gemini-2.0-flash",
        "Gemini 2.0 Flash Lite": "gemini-2.0-flash-lite",
        "Gemini 2.0 Pro Experimental": "gemini-2.0-pro-exp-02-05",
    }
    
    selected_model = st.selectbox(
        "🤖 Select AI Model",
        options=list(model_options.keys()),
        index=0,
        help="Choose the Gemini model to use for analysis"
    )
    model_key = model_options[selected_model]
    
    # Store selected model in session state for persistence
    if 'selected_model' not in st.session_state or st.session_state.selected_model != model_key:
        st.session_state.selected_model = model_key


    # Create two tabs using radio buttons
    input_method = st.radio("Choose Input Method", ["📂 File Upload", "✍️ Text Query"], horizontal=True)

    if input_method == "📂 File Upload":
        # File upload and processing section
        uploaded_file = st.file_uploader("Upload Project Document", type=["pdf", "docx", "txt"])
                
        # Process new file upload if provided
        if uploaded_file is not None:
            file_text = process_uploaded_file(uploaded_file)
            
            if file_text:
                # Add Find Best Consultants button immediately after file upload
                st.markdown("---")
                if st.button("✨ Find Best Consultants", key="find_consultants"):
                    with st.spinner('⚙️ Processing project document...'):
                        project_summary = generate_project_summary(file_text, model=st.session_state.selected_model)
                        st.session_state.project_summary = project_summary
                        st.write("**📋 Project Summary:**")
                        st.write(project_summary)
                        
                        embeddings = get_embeddings()
                        consultant_df = load_consultant_data()
                        if consultant_df is not None:
                            vector_store = create_consultant_vector_store(embeddings, consultant_df)
                            if vector_store:
                                with st.spinner('🔍 Finding best consultant matches...'):
                                    matches = find_best_consultant_matches(vector_store, project_summary, model=st.session_state.selected_model)
                                    st.session_state.current_matches = matches
                                    if matches:
                                        st.write("🎯 **Best Matching Consultants**")
                                        for i, consultant in enumerate(matches, 1):
                                            with st.expander(f"👨‍💼 Consultant {i}: {consultant['Full Name']}"):
                                                cols = st.columns(2)
                                                with cols[0]:
                                                    st.markdown(f"**💸 Finance Expertise:** {consultant['Finance Expertise']}")
                                                    st.markdown(f"**💰 Light Finance:** {consultant['Light Finance']}")
                                                    st.markdown(f"**🎖️ Strategy Expertise:** {consultant['Strategy Expertise']}")
                                                    st.markdown(f"**📌 Entrepreneurship Expertise:** {consultant['Entrepreneurship Expertise']}")
                                                    st.markdown(f"**🚚 Operations Expertise:** {consultant['Operations Expertise']}")
                                                    st.markdown(f"**💼 Marketing Expertise:** {consultant['Marketing Expertise']}")
                                                    st.markdown(f"**🔖 Finished Projects:** {consultant['Finished Projects']}")
                                                    st.markdown(f"**🗣️ Languages for Service:** {consultant['Languages for Service']}")
                                                with cols[1]:
                                                    st.markdown(f"**📚 Areas & Skills:** {consultant['Area Skills']}")
                                                    st.markdown(f"**🏢 Industry Skills:** {consultant['Industry Skills']}")
                                                    st.markdown(f"**📅 Consultant Availability Status:** {consultant['Consultant Availability Status']}")
                                                    st.markdown(f"**📆 Anticipated Availability Date:** {consultant['Anticipated Availability Date']}")
                                                    st.markdown(f"**📝 Comments:** {consultant['Comments']}")
                                                    st.markdown(f"**🏘️ Home Address:** {consultant['Home Address']}")
                                                
                                                st.markdown("---")
                                                st.markdown("**🔍 Match Analysis:**")
                                                st.markdown(consultant['Match Analysis'])
                                    else:
                                        st.error("😔 No matching consultants found.")
                            else:
                                st.error("❌ Could not create consultant vector store")
                        else:
                            st.error("❌ Could not load consultant data")

    else:  # Text Query tab
        # Initialize chat messages if not already done
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("💭 Ask about consultant matching..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Get embeddings and vector store for context
                    embeddings = get_embeddings()
                    consultant_df = load_consultant_data()
                    vector_store = create_consultant_vector_store(embeddings, consultant_df)
                    if vector_store:
                        response = chat_with_consultant_database(prompt, vector_store, consultant_df, model=st.session_state.selected_model)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Could not create consultant vector store")

if __name__ == "__main__":
    main()
