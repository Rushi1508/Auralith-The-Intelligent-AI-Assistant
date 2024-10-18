import streamlit as st
import speech_recognition as sr
from langchain_openai import OpenAI
import os
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from macros import *
from modules.Path import loadPath
from modules import Functions

def start_chat_module():
    st.title("Auralith: The Intelligent AI Assistant")
    st.title("Chat Mode")
    os.environ["OPENAI_API_KEY"] = "" #Insert your OpenAI API Key here
    FileSystemManager = Functions.FileSystemManager()
    ProcessManager = Functions.ProcessManager()
    
    loadPath(ProcessManager)
    
    llm = OpenAI()
    
    response_schemas = [
        ResponseSchema(name="action", description="The most suitable action to achieve this task for the user"),
        ResponseSchema(name="parameters", description="The parameters accompanying the action to achieve the desired goal")
    ]
    
    code_schemas = [
        ResponseSchema(name="code", description="The code for given program"),
        ResponseSchema(name="language", description="The programming language used to write the code")
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    code_parser = StructuredOutputParser.from_response_schemas(code_schemas)
    
    format_instructions = output_parser.get_format_instructions()
    code_instructions = code_parser.get_format_instructions()
    
    mainTemplate = """
    You will take user query and based on that select most suitable action.
    1) SearchWeb or 2) WriteCode 3) Converse
    and give output in the following template
    {format_instructions}
    
    Query: {query}
    """
    codeTemplate = """
    You are Luna, a virtual assistant. Write code for the user query. Don't write anything else except the code in mentioned programming language.
    
    Query: {query}
    """
    chatTemplate = """
    You are Luna, a virtual assistant. You were made by Kevin Santhosh. Your task is to analyze user query and provide step by step instructions to solve their query.
    Query: {query}
    """
    
    prompt = PromptTemplate(template=mainTemplate, input_variables=["format_instructions", "query"])
    codePrompt = PromptTemplate(template=codeTemplate, input_variables=["query"])
    chatPrompt = PromptTemplate(template=chatTemplate, input_variables=["query"])
    
    
    user_prompt = ""
    
    def listen_to_microphone():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            st.write("Transcribing...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand what you said.")
            return ""
        except sr.RequestError as e:
            st.write("Sorry, there was an error in processing your request. Please try again later.")
            return ""
    
    def process_user_input(user_prompt):
        finalPrompt = prompt.format(query=user_prompt, format_instructions=format_instructions)
        output = llm.invoke(finalPrompt)
        data = output_parser.parse(output)
    
        if data['action'] == 'SearchWeb':
            st.write("Search Web for:", data['parameters'])
            searchWeb(data['parameters'])
        elif data['action'] == 'WriteCode':
            st.write("Sure, I will write the code for you!")
            codeOutput = llm.invoke(codePrompt.format(query=user_prompt))
            st.write(codeOutput)  # Directly print the code output without any extra text
            lines = codeOutput.split('\n')
            
            # Clean up unnecessary lines that may start with backticks or extra formatting
            if lines and lines[0].startswith('`'): 
                lines.pop(0)
            if lines and lines[-1].startswith('`'): 
                lines.pop()
    
            # Join the cleaned lines back together
            codeOutput = '\n'.join(lines)
            
            # Write the clean code output to a file
            FileSystemManager.writeToFile("code.txt", codeOutput)
        else:
            st.write("[Chat Mode]")
            st.write(llm.invoke(chatPrompt.format(query=user_prompt)))
    
        # Comment out or remove the unnecessary API data
        # st.write("API Data (Currently in development)")
        # st.write(output)
    
    
    capture_voice = st.button("Capture Voice")
    if capture_voice:
        user_prompt = listen_to_microphone()
        st.text_area("Enter voice prompt", value=user_prompt, key="voice_prompt")
        process_user_input(user_prompt)
    else:
        user_prompt = st.text_area("Enter your prompt")
    
    button = st.button("Submit")
    
    if button:
        if not user_prompt:
            st.write("Please provide a prompt.")
        else:
            process_user_input(user_prompt)