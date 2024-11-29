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

# Streaming handler for smooth response display
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

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
    
    # Show assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        try:
            # Get response from RAG system
            result = rag.query(question)
            
            # Format response with markdown
            response = f"""
            {result['answer']}
            
            ---
            **Sources:**
            """
            for source in result['sources']:
                response += f"\n- {source}"
            
            # Display response
            response_container.markdown(response)
            
            # Store in chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            response_container.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

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