"""
Simple BankEase Tool Calling Agent
"""

# =========================
# IMPORTS
# =========================

import os
import json
import sqlite3

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage
)

from langchain_groq import ChatGroq


load_dotenv()

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)



@tool
def search_web(query: str) -> str:
    """
    Use ONLY for:
    - latest news
    - current rates
    - live information
    - internet searches

    Do NOT use for account details.
    """

    try:

        response = tavily.search(
            query=query,
            max_results=2
        )

        results = []

        for item in response.get("results", []):

            results.append({
                "title": item["title"],
                "content": item["content"][:150]
            })

        return json.dumps(results, indent=2)

    except Exception as e:
        return f"Web search error: {e}"




def build_agent():

    llm = ChatGroq(

        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    tools = [
        search_web,
    ]

    agent = llm.bind_tools(tools)

    return agent, tools



def run_agent(agent, tools, question):

    tool_map = {
        tool.name: tool
        for tool in tools
    }

    messages = [

        SystemMessage(
            content="""
            You are a banking assistant.

            Rules:
            1. Use search_web only for latest internet information.
            2. Use query_account only for customer account details.
            3. After getting the correct tool result,
               give the final answer immediately.
            4. Do not call unnecessary tools.
            """
        ),

        HumanMessage(content=question)
    ]

    print("\n" + "=" * 60)
    print("Customer:", question)
    print("=" * 60)

    while True:

        response = agent.invoke(messages)

        # Final answer
        if not response.tool_calls:

            print("\nAgent:", response.content)
            break

        messages.append(response)

        # Execute tools
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\nCalling Tool: {tool_name}")
            print("Arguments:", tool_args)

            selected_tool = tool_map[tool_name]

            result = selected_tool.invoke(tool_args)

            print("Tool Result:", result)

            messages.append(

                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )




def main():

    agent, tools = build_agent()

    questions = [

        "What is the current USD-INR exchange rate?",

        "What is the latest inflation rate?",
    ]

    for q in questions:

        run_agent(agent, tools, q)


if __name__ == "__main__":
    main()