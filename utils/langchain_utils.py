from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import SemanticSimilarityExampleSelector

from utils.few_shots import few_shots

import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class LangchainHelper:
    def __init__(self):
        pass

    def get_db_chain(self, use_few_shot=False):
        llm = GooglePalm(google_api_key=api_key, temperature=0.2)

        db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)

        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

        if use_few_shot:
            example_selector = self.few_shot_learning(few_shots)
            
            example_prompt = PromptTemplate(
                input_variables=["Question", "SQLQuery", "SQLResult", "Answer",],
                template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
            )

            few_shot_prompts = FewShotPromptTemplate(
                example_selector=example_selector,
                example_prompt=example_prompt,
                prefix=_mysql_prompt,
                suffix=PROMPT_SUFFIX,
                input_variables=["input", "table_info", "top_k"],
            )

            db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompts)

        return db_chain

    def few_shot_learning(self, few_shots):
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        to_vectorize = [" ".join(shots.values()) for shots in few_shots]

        # Replace '\n' with ' ' in each element of the list
        to_vectorize = [text.replace("\n", " ") for text in to_vectorize]

        vector_store = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=few_shots)

        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            k=2
        )

        return example_selector