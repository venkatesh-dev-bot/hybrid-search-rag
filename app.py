import json
from pathlib import Path
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
from rag_openaiembeddings import ImprovedRAGSystem  # Your RAG system class

# Page configuration
st.set_page_config(
    page_title="Vector Mathematics Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize RAG system
@st.cache_resource
def initialize_rag():
    rag = ImprovedRAGSystem()
    print("Loading and preparing documents...")
    rag.prepare_documents_from_json("vector_content.json")
    print("Creating vector store...")
    rag.create_vectorstore()
    return rag

# Enhanced StreamHandler for better response display
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.full_response = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.full_response += token
        # Update the container with the new token
        self.container.markdown(self.full_response)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Vector Math Assistant")
    st.markdown("""
    This assistant helps with:
    - Vector concepts
    - 3D geometry
    - Mathematical proofs
    - Problem solving
    """)
    
    if st.button("Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Display some example questions
    st.subheader("Example Questions")
    st.markdown("""
    - What is a zero vector?
    - What are parallel vectors?
    - How do you find a unit vector?
    """)

# Main content
st.title("📐 Vector Mathematics Assistant")
st.markdown("""
---
🎓 **Ask questions about vectors and 3D geometry**
---
""")

# Initialize RAG system
rag = initialize_rag()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Ask your vector mathematics question...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Show assistant response with streaming
    with st.chat_message("assistant"):
        # Create placeholder for the response
        response_placeholder = st.empty()
        
        try:
            # Initialize the stream handler
            stream_handler = StreamHandler(response_placeholder)
            
            # Show thinking message
            with response_placeholder:
                st.write("🤔 Thinking...")
            
            # Update the chat model to enable streaming
            rag.chat_model.streaming = True
            rag.chat_model.callbacks = [stream_handler]
            
            # Get response from RAG system
            result = rag.query(question)
            
            # Format full response with markdown and sources
            full_response = f"""
            {stream_handler.full_response}
            
            Sources:
            """
            for source in result['sources']:
                full_response += f"\n- {source}"
            
            # Update the placeholder with the complete response
            response_placeholder.markdown(full_response)
            
            # Store in chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            response_placeholder.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })
        
        finally:
            # Reset streaming configuration
            rag.chat_model.streaming = False
            rag.chat_model.callbacks = []

# Add custom CSS
st.markdown("""
<style>
    /* Chat container */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Mathematical notation */
    .math-block {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Source citations */
    .source-citation {
        font-size: 0.9em;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        padding-top: 0.5rem;
        margin-top: 1rem;
    }
    
    /* Diagrams */
    .diagram-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Add MathJax support
components.html("""
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            },
            svg: {
                fontCache: 'global'
            }
        };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
""", height=0) 
