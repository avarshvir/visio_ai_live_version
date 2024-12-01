#viz_ai.py

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image


import google.generativeai as genai


os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load OpenAI model and get respones
def viz_ai_img():
    def get_gemini_response(input,image):
        model = genai.GenerativeModel('gemini-1.5-flash')
        if input!="":
            response = model.generate_content([input,image])
        else:
            response = model.generate_content(image)
        return response.text

##initialize our streamlit app

    #st.set_page_config(page_title="Gemini Image Demo")

    st.header("🤖Viz AI")
    st.markdown("### Your Personal Imager AI Insighter")
    input=st.text_input("Input Prompt: ",key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)


    submit=st.button("Go and Find Pattern")

## If ask button is clicked

    if submit:
    
        response=get_gemini_response(input,image)
        st.subheader("I find these hidden pattern for you 😇")
        st.write(response)

if __name__ == "__main__":
    viz_ai_img()