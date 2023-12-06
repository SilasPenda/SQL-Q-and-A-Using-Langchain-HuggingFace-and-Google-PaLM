import streamlit as st

from utils.langchain_utils import LangchainHelper

langchain_helper = LangchainHelper()

st.title("Sapen Clothing Store")
st.sidebar.subheader('Parameters')
st.markdown(
"""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 400px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    width: 400px;
    margin-left: -400px;
}
</style>
""",
unsafe_allow_html=True,
)

use_few_shot = st.sidebar.checkbox('Few Shot')

question = st.text_input("Question: ")

if question:
    if use_few_shot:
        print("Few shot learning enabled!")
        chain = langchain_helper.get_db_chain(use_few_shot=True)
    else:
        chain = langchain_helper.get_db_chain()

    answer = chain.run(question)

    st.header("Answer: ")
    st.write(answer)