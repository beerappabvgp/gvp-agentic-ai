

import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage
)

from langchain_groq import ChatGroq



load_dotenv()


def execute_file_create(filename: str, content: str) -> str:
    """Create a new file with the given content."""
    try:
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            f.write(content)
        return f"✓ File created successfully: {filename}\n   Size: {len(content)} bytes"
    except Exception as e:
        return f"✗ Error creating file: {str(e)}"


def execute_file_read(filename: str) -> str:
    """Read and return the contents of a file."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return f"File: {filename}\n{'='*50}\n{content}\n{'='*50}"
    except FileNotFoundError:
        return f"✗ File not found: {filename}"
    except Exception as e:
        return f"✗ Error reading file: {str(e)}"


def execute_file_update(filename: str, old_text: str, new_text: str) -> str:
    """Update a file by replacing old text with new text."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        if old_text not in content:
            return f"✗ Text not found in file: {filename}"
        updated_content = content.replace(old_text, new_text)
        with open(filename, 'w') as f:
            f.write(updated_content)
        return f"✓ File updated successfully: {filename}"
    except FileNotFoundError:
        return f"✗ File not found: {filename}"
    except Exception as e:
        return f"✗ Error updating file: {str(e)}"


# Tools with decorators for LangChain
@tool
def create_file(filename: str, content: str) -> str:
    """Create a new Python or text file with the given code or content."""
    return execute_file_create(filename, content)


@tool
def read_file(filename: str) -> str:
    """Read the contents of a file."""
    return execute_file_read(filename)


@tool
def update_file(filename: str, old_text: str, new_text: str) -> str:
    """Update a file by replacing text."""
    return execute_file_update(filename, old_text, new_text)


def init_conversation():
    """Initialize empty conversation history"""
    return []


def add_user_message(history, message):
    """Add user message to history"""
    history.append({"role": "user", "content": message})
    return history


def add_assistant_message(history, message):
    """Add assistant message to history"""
    history.append({"role": "assistant", "content": message})
    return history


def add_tool_message(history, content, tool_name):
    """Add tool result to history"""
    history.append({"role": "tool", "content": content, "tool": tool_name})
    return history


def display_history(history):
    """Display formatted conversation history"""
    if not history:
        return "(No history yet)"
    
    formatted = []
    for msg in history:
        role = msg["role"].upper()
        if role == "TOOL":
            tool_name = msg.get("tool", "")
            formatted.append(f"\n🔧 Tool [{tool_name}]:\n{msg['content']}")
        elif role == "USER":
            formatted.append(f"\n👤 You:\n{msg['content']}")
        elif role == "ASSISTANT":
            formatted.append(f"\n🤖 Copilot:\n{msg['content']}")
    
    return "\n".join(formatted)




def build_copilot_agent():
    """Build a Copilot-like agent with file tools"""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    tools = [
        create_file,
        read_file,
        update_file
    ]
    
    # Bind tools to LLM
    agent = llm.bind_tools(tools)
    
    return agent, tools



def run_copilot():
    """
    Run the Copilot agent in an interactive loop with persistent conversation history.
    """
    
    print("\n" + "="*70)
    print(" COPILOT-LIKE AGENT - File Management & Code Assistant")
    print("="*70)
    print("\nCommands:")
    print("  - Ask anything about files or coding")
    print("  - Type 'history' to see conversation history")
    print("  - Type 'clear' to reset conversation")
    print("  - Type 'exit' to quit")
    print("="*70 + "\n")
    
    agent, tools = build_copilot_agent()
    conversation = init_conversation()
    
    # Tool map for execution
    tool_map = {tool.name: tool for tool in tools}
    
    # System message
    system_prompt = """You are GitHub Copilot - an advanced AI programming assistant.

Your capabilities:
1. Create files with code (support all the programming languages), 
2. Read and understand file contents
3. Update files with improvements
4. Answer questions about code
5. Provide suggestions and explanations

Guidelines:
- Be helpful and concise
- Use tools appropriately for file operations
- Remember the conversation history and context
- Provide explanations for your changes"""
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("(Please enter a message)")
                continue
            
            if user_input.lower() == "exit":
                print("\n🤖 Copilot: Goodbye! Thanks for using Copilot.")
                break
            
            if user_input.lower() == "history":
                print("\n" + "="*70)
                print("CONVERSATION HISTORY")
                print("="*70)
                print(display_history(conversation))
                print("\n" + "="*70)
                continue
            
            if user_input.lower() == "clear":
                conversation = init_conversation()
                print("\n🤖 Copilot: Conversation history cleared.")
                continue
            
            # Add user message to history
            conversation = add_user_message(conversation, user_input)
            
            # Build messages with full conversation history
            messages = [SystemMessage(content=system_prompt)]
            
            # Add all previous messages from conversation history
            for msg in conversation[:-1]:  # Exclude the current message we just added
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "tool":
                    messages.append(ToolMessage(
                        content=msg["content"],
                        tool_call_id=msg.get("tool", "tool_call")
                    ))
            
            # Add current user message
            messages.append(HumanMessage(content=user_input))
            
            print("\n🤖 Copilot: ", end="", flush=True)
            
            # Agent loop
            iteration = 0
            max_iterations = 5
            
            while iteration < max_iterations:
                iteration += 1
                
                try:
                    response = agent.invoke(messages)
                except Exception as e:
                    print(f"\n✗ Error: {str(e)[:100]}")
                    break
                
                # Check if there are tool calls
                if not response.tool_calls:
                    # Final answer - no more tool calls
                    print(response.content)
                    conversation = add_assistant_message(conversation, response.content)
                    break
                
                # Add assistant response to messages
                messages.append(response)
                
                # Execute tools
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    print(f"\n   [Executing {tool_name}...]", end="", flush=True)
                    
                    try:
                        selected_tool = tool_map[tool_name]
                        result = selected_tool.invoke(tool_args)
                    except Exception as e:
                        result = f"Error: {str(e)}"
                    
                    # Add tool result to messages
                    messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                    
                    # Store in conversation history
                    conversation = add_tool_message(conversation, str(result), tool_name)
        
        except KeyboardInterrupt:
            print("\n\n🤖 Copilot: Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {str(e)[:150]}")
            continue


# =========================
# MAIN
# =========================

def main():
    """Entry point"""
    run_copilot()


if __name__ == "__main__":
    main()

