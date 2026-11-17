import streamlit as st

capture_voice = st.button("Capture Voice")
if capture_voice:
    user_prompt = listen_to_microphone()
    st.text_area("Enter voice prompt", value=user_prompt, key="voice_prompt")
else:
    user_prompt = st.text_area("Enter your prompt")

# Submit button
submit_button = st.button("Submit")

if submit_button:
    if not user_prompt:
        st.write("Please provide a prompt.")
    else:
        process_user_input(user_prompt)