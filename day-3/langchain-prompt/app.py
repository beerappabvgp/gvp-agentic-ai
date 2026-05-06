from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate
)
load_dotenv()

llm_client = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0,
    api_key = os.getenv("GROQ_API_KEY")
)

parser = StrOutputParser()

prompt = PromptTemplate.from_template(
    "What is Deep learning?"
)

chain = prompt | llm_client | parser

result = chain.invoke({})

print(result)