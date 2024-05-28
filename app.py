import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers #it helps in calling hugging face model or call it from local

## Function to get response from my Lama-2 model

def getLLamaresponse(input_text, no_words,blog_style):

    #Lama 2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens' : 256,
                                 'temperature' : 0.01})

    #Prompt Template
    template="""  
            Write a blog for {blog_style} for job profile for a topic {input_text} 
            within {no_words} words"""

    prompt = PromptTemplate(input_variables=["blog_style","input_text","no_words"],
                            template=template)                            #this are the info will be given while giving the prompt template

    ##Generate the response from the llama 2 model which is from ggml
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)) #llm will give reponse to this
    print(response)
    return response





st.set_page_config(page_title="Generate Blogs",
                   page_icon=" :) ",
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header("Generate Blogs")
input_text=st.text_input("Enter the Blog Topic")

#creating two more columns for additional 2 fields
col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox("Writing the blog for",
                            ('Researchers', 'DataScientist', 'CommonPeople'),
                            index=0)
submit = st.button("Generate")

# Final Response
if submit:
    st.write(getLLamaresponse(input_text, no_words,blog_style))

