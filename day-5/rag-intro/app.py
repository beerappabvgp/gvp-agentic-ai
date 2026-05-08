from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Step1: setup embedding
embeddings = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
)

# check: Only index if not already done
persist_dir = "./bank_db"
collection_name = "bank_policy"

# trying to connect to existing chromaDB first 
vectorstore = Chroma(
    persist_directory=persist_dir,
    collection_name=collection_name,
    embedding_function = embeddings
)

existing_count = vectorstore._collection.count()
print(f"Existing chunks in ChromaDB: {existing_count}")

if existing_count == 0:
    print("No data found. Indexing pdf now")
    # load the pdf document 
    loader = PyPDFLoader('bank.pdf')
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # chunking 
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        separators = ['\n\n', '\n', ', ', ' ', '']
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    
    # embed and store 
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = persist_dir,
        collection_name= collection_name,
    )
    
else:
    print(f"Found {existing_count} chunks already indexed. Skipping re-indeing.")
    
# Testing the similarity search

query = "What documents does a self-employed doctor need to apply for a home loan?"
results = vectorstore.similarity_search_with_score(query, k = 3)
print('Results: ===== ', results)
print(f"\nTop 3 results for: {query}")
# results = [
    
# ]
for i, (doc, score) in enumerate(results):
    print(f'Result {i+1} (similarity: {score:.3f}):')
    print(f'Page: {doc.metadata.get("page", "?")} | {doc.page_content[:120]}....')
    
# set up the llm

llm = ChatGroq(
    model = 'llama-3.1-8b-instant',
    temperature = 0,
    api_key = os.getenv("GROQ_API_KEY")
)

# create a RAG prompt 

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '''You are bank agent specialist.Answer the questions using ONLY the context provided below/If the answer is not in the context, say: "I could not find in this policy.Always cite the page number from the context metadata
        CONTEXT: {context} 
        '''),
        ('human', '{question}')
    ]
)

# Finally answer the question 

def answer_question(question): 
    docs = vectorstore.similarity_search(question, k = 3)
    context = '\n\n'.join([
        f'[Page {d.metadata.get("page", "?")}: {d.page_content}]'
        for d in docs
    ])
    # print("context: =======", context)
    chain = rag_prompt | llm | StrOutputParser()
    return chain.invoke({'question': question, 'context': context})

# run test questions 

questions = [
    "What documents does a self-employed doctor need to apply for a home loan?",
    "What income proof do I need to submit if I am a salaried employee?",
    "How many years of ITR are required for self-employed applicants?",
    "Do I need to submit bank statements? If yes, for how many months?",
    "What is the LTV ratio for a property worth Rs 80 lakhs?",
    "Can I get a fixed rate home loan at BankEase? What is the premium?",
    "What happens if my CIBIL score is 670 — can I still get a loan?",
    "Can I club my spouse's income with mine for the loan eligibility?",
]

for q in questions:
    print(f'\nQ: {q}')
    print(f'A: {answer_question(q)}')
    print("=================================================")