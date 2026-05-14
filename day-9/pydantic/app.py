from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class StudentInfo(BaseModel):
    """ THis is how the student data looks like"""
    name: str = Field(description ="Full name of the student")
    age: int = Field(description = "Age of the student in years")
    course: str = Field(description = "The course or subject the student is studying")
    grade: str = Field(description = "Current grade or score")
    email: str = Field(description = "Student's email address")
    
    
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0
)

structured_llm = llm.with_structured_output(StudentInfo)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Extract student information from the given text"),
    ("human", "{text}")
])

chain = prompt | structured_llm

sample_text = """
    Hi, I am Ganesh, I am 21 years old and currently studying AI/ML at GVP college. I scored an A+ in my last semester.
    My email address is ganesh@gmail.com
"""
print("=" * 50)
print(" INPUT TEXT: ")
print("=" * 50)

result = chain.invoke({
    "text": sample_text
})

print("result of llm call: ", result)
print("\n Pydantic validated studentInfo Object:\n")
print(f"Name: {result.name}")
print(f"Age: {result.age}")
print(f"Course: {result.course}")
print(f"Grade: {result.grade}")
print(f"Email: {result.email}")