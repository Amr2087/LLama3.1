
import os
import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

model="llama-3.1-8b-instant"

def get_corrected_text(api_key, text, model=model):
    groq_chat = ChatGroq(
        groq_api_key=api_key,
        model_name=model
    )

    system_prompt = """You are an arabic language checker that takes incorrect arabic sentences and correct it without changing the context or the meaning just correct the spelling for each incorrect word only if the word is right dont change anything.
            Example :
            Input : وابنهشام هالأنصاري الذي يعد إحدى قمم عملاء العربية على مدى العصور
            Your Output should be : وابن هشام الأنصاري الذي يعد إحدى قمم علماء العربية على مدى العصور
            Meaning dont add anything to the output except the correction of the sentence in arabic and no other languages"""

    conversational_memory_length = 5
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history",
                                            return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    corrected_text = conversation.predict(human_input=text)
    return corrected_text




def main():
    st.title('Text Correction and Word Prediction with Groq')

    api_key = st.text_input('Enter your Groq API key:', type='password')

    if api_key:
        tab1, tab2 = st.tabs(["Text Correction", "Word Prediction"])

        with tab1:
            st.header('Text Correction')
            input_text = st.text_area('Enter text to correct:')
            if st.button('Correct Text'):
                if input_text:
                    corrected_text = get_corrected_text(api_key, input_text)
                    st.subheader('Corrected Text:')
                    st.write(corrected_text)
                else:
                    st.error('Please provide text to correct.')


    else:
        st.warning('Please enter your Groq API key.')


if __name__ == "__main__":
    main()
