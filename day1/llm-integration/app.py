from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

REAL_RATE_DATA = '''
    BankEase Fixed Deposit Rates (updated 1 May 2025)
    - 1 year FD: 7.50% per annum
    - 2 year FD: 7.75% per annum
    - 3 year FD: 8.00% per annum
    Source: BankEase Rate Card v12
'''

# system_msg = f'''
#     You are a customer assistant for BankEase, Mumbai.
#     Answer questions using ONLY the data below.
#     If the answer is not in the data, say: I do not have that information.
#     {REAL_RATE_DATA}
# '''

# user_msg = "What is the interest rate on your 1-year fixed deposit?"

system_msg = "Imagine you are a human being and answer the below question"
user_msg = "When two of your friends are in difficulty whom do you save and why"

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages = [
        {
            'role': 'system',
            'content': system_msg
        },
        {
            'role': 'user',
            'content': user_msg
        }
    ],
    temperature=0
)

print(response.choices[0].message.content)