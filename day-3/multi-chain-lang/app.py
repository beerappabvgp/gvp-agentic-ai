from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ORDER PROMPT
order_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a pizza shop assistant. Help customers place or track orders.'),
    ('human', '{prompt}')
])

# MENU PROMPT
menu_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a pizza shop menu expert. Answer questions about prices, toppings, and sizes.'),
    ('human', '{prompt}')
])

# GENERAL PROMPT
general_prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a pizza shop assistant. Handle general questions about the shop.'),
    ('human', '{prompt}')
])

# CREATE CHAINS
order_chain = order_prompt | llm | parser
menu_chain = menu_prompt | llm | parser
general_chain = general_prompt | llm | parser

# ROUTER
router = RunnableBranch(
    (
        lambda x: any(
            kw in x['prompt'].lower()
            for kw in ['order', 'pizza', 'delivery']
        ),
        order_chain
    ),

    (
        lambda x: any(
            kw in x['prompt'].lower()
            for kw in ['menu', 'price', 'topping']
        ),
        menu_chain
    ),

    general_chain
)

# TESTS
tests = [
    'I want to order a large pizza',
    'What toppings are available',
    'What are the opening hours of the restaurant'
]

for q in tests:
    print(f"Query: {q}")
    print(f"Answer: {router.invoke({'prompt': q})}")
    print()