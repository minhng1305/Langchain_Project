import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ["OPENAI_API_KEY"] = 'API Key'

# App framework
st.title("YouTube Script")
prompt = st.text_input("Plug in your prompt here")

template = "Write me a title about {topic}"

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = template
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = "Write me a script about title: {title} by using wikipedia research: {wikipedia_research}"
)

# Memory
title_memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')
script_memory = ConversationBufferMemory(input_key = 'title', memory_key = 'chat_history')

# llms
llm = OpenAI(temperature = .9)
title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'title', memory = title_memory)
script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key = "script", memory = script_memory)
#chain = SequentialChain(chains = [title_chain, script_chain], input_variables = ["topic"], output_variables = ['title', 'script'], verbose = True)

wiki = WikipediaAPIWrapper()

# Present on the screen
if prompt:
    title = title_chain.run(topic=prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title, wikipedia_research = wiki_research)

    #reply = chain({'topic':prompt})
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Scipt History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia History'):
        st.info(wiki_research)
