import streamlit as st
from chat_module import start_chat_module
from image_module import start_image_module
from pdf_module import start_pdf_module
from modules import Functions  # Assuming macros or utility functions are in 'Functions'

# Initialize session state to track the selected module
if 'module' not in st.session_state:
    st.session_state.module = "Home"

# Function to go back to home page (landing page)
def go_home():
    st.session_state.module = "Home"

# Define the landing page
def landing_page():
    st.title("Auralith: The Intelligent AI Assistant")
    st.write("Welcome! Please choose one of the following options:")
    
    if st.button("Chat Mode"):
        st.session_state.module = "Chat"
    
    if st.button("Image Query"):
        st.session_state.module = "Image Query"
    
    if st.button("PDF Query"):
        st.session_state.module = "PDF Query"

# Main function to render the selected module
def main():
    if st.session_state.module == "Home":
        landing_page()
    elif st.session_state.module == "Chat":
        start_chat_module()
    elif st.session_state.module == "Image Query":
        start_image_module()
    elif st.session_state.module == "PDF Query":
        start_pdf_module()

    # Add a Back to Home button to allow navigation back to the main landing page
    if st.session_state.module != "Home" and st.button("Back to Home"):
        go_home()

if __name__ == "__main__":
    main()
