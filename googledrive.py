import streamlit as st
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load credentials from streamlit secrets
credentials_data = st.secrets["gcp"]["service_account_json"]
creds = json.loads(credentials_data, strict=False)

# Authenticate and construct the Drive API service
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = service_account.Credentials.from_service_account_info(creds, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def get_file_icon(mime_type):
    """Return appropriate icon and type label based on MIME type"""
    mime_map = {
        'application/vnd.google-apps.folder': ('📁', 'Folder'),
        'application/vnd.google-apps.document': ('📄', 'Google Doc'),
        'application/vnd.google-apps.spreadsheet': ('📊', 'Google Sheet'),
        'application/vnd.google-apps.presentation': ('📽️', 'Google Slides'),
        'application/pdf': ('📑', 'PDF'),
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ('📝', 'Word'),
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ('📊', 'Excel'),
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': ('📽️', 'PowerPoint'),
        'text/plain': ('📄', 'Text'),
    }
    return mime_map.get(mime_type, ('📄', 'Other'))

def list_drive_files(page_size=100, page_token=None):
    """List all files and folders in Google Drive with detailed error handling"""
    try:
        # First, let's check if we can access the drive at all
        about = drive_service.about().get(fields="user").execute()
        st.sidebar.success(f"Connected as: {about.get('user', {}).get('emailAddress', 'Unknown')}")
        
        # Create a query that explicitly includes all common file types
        query = " or ".join([
            "mimeType = 'application/vnd.google-apps.folder'",
            "mimeType = 'application/vnd.google-apps.document'",
            "mimeType = 'application/vnd.google-apps.spreadsheet'",
            "mimeType = 'application/vnd.google-apps.presentation'",
            "mimeType = 'application/pdf'",
            "mimeType contains 'officedocument'",
            "mimeType = 'text/plain'"
        ])
        
        results = drive_service.files().list(
            pageSize=page_size,
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, owners, shared)",
            orderBy="modifiedTime desc",
            q=query,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        
        files = results.get('files', [])
        
        # Log file access information
        st.sidebar.write("Files found:", len(files))
        
        return files, results.get('nextPageToken')
    
    except Exception as e:
        st.error(f"Error accessing Google Drive: {str(e)}")
        st.info("Please check if your service account has been shared with the necessary files/folders")
        return [], None

st.title("Google Drive Files and Folders")

# Add debug information in sidebar
st.sidebar.title("Debug Information")

# Add pagination control
if 'page_token' not in st.session_state:
    st.session_state.page_token = None

files, next_page_token = list_drive_files(page_token=st.session_state.page_token)

if files:
    # Organize files by type
    organized_files = {
        'Folders': [],
        'Google Docs': [],
        'Google Sheets': [],
        'Google Slides': [],
        'PDFs': [],
        'Microsoft Office': [],
        'Other': []
    }
    
    for file in files:
        mime_type = file['mimeType']
        icon, type_label = get_file_icon(mime_type)
        
        # Add ownership information
        owners = file.get('owners', [])
        owner_email = owners[0].get('emailAddress') if owners else 'Unknown'
        
        file_info = {
            'icon': icon,
            'name': file['name'],
            'id': file['id'],
            'mime_type': mime_type,
            'modified_time': file.get('modifiedTime', 'N/A'),
            'owner': owner_email,
            'shared': file.get('shared', False)
        }
        
        if mime_type == 'application/vnd.google-apps.folder':
            organized_files['Folders'].append(file_info)
        elif mime_type == 'application/vnd.google-apps.document':
            organized_files['Google Docs'].append(file_info)
        elif mime_type == 'application/vnd.google-apps.spreadsheet':
            organized_files['Google Sheets'].append(file_info)
        elif mime_type == 'application/vnd.google-apps.presentation':
            organized_files['Google Slides'].append(file_info)
        elif mime_type == 'application/pdf':
            organized_files['PDFs'].append(file_info)
        elif 'officedocument' in mime_type:
            organized_files['Microsoft Office'].append(file_info)
        else:
            organized_files['Other'].append(file_info)
    
    # Display files by category
    for category, items in organized_files.items():
        if items:
            st.subheader(f"{category}")
            for file_info in items:
                st.write(
                    f"{file_info['icon']} {file_info['name']} "
                    f"(ID: {file_info['id']}) - "
                    f"Owner: {file_info['owner']} - "
                    f"Last modified: {file_info['modified_time']}"
                )

    # Pagination controls
    col1, col2 = st.columns(2)
    
    if next_page_token:
        if col2.button("Next Page ▶"):
            st.session_state.page_token = next_page_token
            st.rerun()
            
    if st.session_state.page_token:
        if col1.button("◀ Previous Page"):
            st.session_state.page_token = None
            st.rerun()

else:
    st.warning("No files found or error accessing Google Drive. Check the sidebar for details.")

# Display MIME type statistics
if files:
    st.subheader("File Type Statistics")
    mime_counts = {}
    for file in files:
        mime_type = file['mimeType']
        mime_counts[mime_type] = mime_counts.get(mime_type, 0) + 1
    
    for mime_type, count in mime_counts.items():
        st.write(f"{mime_type}: {count} files")
        
"""
Example MIME Types
Here are some common file types and their associated MIME types you might want to include:

Google Sheets: application/vnd.google-apps.spreadsheet
Google Docs: application/vnd.google-apps.document
Google Slides: application/vnd.google-apps.presentation
PDF: application/pdf
Microsoft Word: application/vnd.openxmlformats-officedocument.wordprocessingml.document
Microsoft Excel: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
Text files: text/plain
"""