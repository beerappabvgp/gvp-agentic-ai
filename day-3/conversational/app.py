from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()

parser = StrOutputParser()

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    api_key = os.getenv("GROQ_API_KEY")
)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key='chat_history',
)

prompt = ChatPromptTemplate.from_messages([
  ('system', "You are a friendly assistant. Talk like a helpful human"),
  MessagesPlaceholder(variable_name="chat_history"),
  ("human", "{input}")  
])

chain = prompt | llm | parser

def chat(user_input):
    # step1 Load the previous messages
    history = memory.load_memory_variables({})["chat_history"]
    
    # run the chain with input + history
    response = chain.invoke({"input": user_input, "chat_history": history})
    
    memory.save_context(
        {"input": user_input},
        {"output": response},    
    )
    
    return response
    
# test 

print("Bot: ", chat("Hi, My name is Bharath, How are you?"))
print("Bot: ", chat("What is my name?"))