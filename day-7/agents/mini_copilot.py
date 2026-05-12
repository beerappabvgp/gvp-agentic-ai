import os
import langchain
from groq import *

class MiniCopilot:
    def __init__(self):
        self.files = {}
        self.llm = langchain.llms.HuggingFaceHub()
        self.tools = {
            "create_file": self.create_file,
            "read_file": self.read_file,
            "update_file": self.update_file
        }

    def create_file(self, filename, content):
        with open(filename, 'w') as f:
            f.write(content)
        self.files[filename] = content

    def read_file(self, filename):
        if filename in self.files:
            return self.files[filename]
        else:
            with open(filename, 'r') as f:
                return f.read()

    def update_file(self, filename, new_content):
        if filename in self.files:
            self.files[filename] = new_content
        else:
            self.create_file(filename, new_content)

    def agent_loop(self):
        while True:
            user_input = input("User: ")
            if user_input.startswith("create_file"):
                filename, content = user_input.split(" ")[1:]
                self.create_file(filename, content)
            elif user_input.startswith("read_file"):
                filename = user_input.split(" ")[1]
                print(self.read_file(filename))
            elif user_input.startswith("update_file"):
                filename, new_content = user_input.split(" ")[1:]
                self.update_file(filename, new_content)
            else:
                print("Invalid command. Please use create_file, read_file, or update_file.")

if __name__ == '__main__':
    agent = MiniCopilot()
    agent.agent_loop()
