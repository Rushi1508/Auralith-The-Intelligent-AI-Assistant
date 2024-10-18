import streamlit as st
import ollama

def start_image_module():
    st.title("Auralith: The Intelligent AI Assistant")
    st.title("Image Query Mode")

    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    # Ensure the image is uploaded
    if uploaded_image is not None:
        # Call Ollama API for image description
        res = ollama.chat(
            model="llava",
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe this image:',
                    'images': [uploaded_image]
                }
            ]
        )
        
        # Get the image description from the response
        description = res['message']['content']
        
        # Display the image description
        st.write("Description:", description)
        
