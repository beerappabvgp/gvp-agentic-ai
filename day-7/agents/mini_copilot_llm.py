import langchain
import groq

class MiniCopilot:
    def __init__(self):
        self.llm = groq.Groq()
        self.tools = {
            "create_file": self.create_file,
            "read_file": self.read_file,
            "update_file": self.update_file
        }

    def create_file(self, filename, content):
        with open(filename, "w") as f:
            f.write(content)

    def read_file(self, filename):
        try:
            with open(filename, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "File not found"

    def update_file(self, filename, new_content):
        try:
            with open(filename, "w") as f:
                f.write(new_content)
        except FileNotFoundError:
            return "File not found"

    def run(self):
        while True:
            user_input = input("What would you like to do? ")
            if user_input.startswith("create_file"):
                filename = input("Enter filename: ")
                content = input("Enter content: ")
                self.create_file(filename, content)
            elif user_input.startswith("read_file"):
                filename = input("Enter filename: ")
                print(self.read_file(filename))
            elif user_input.startswith("update_file"):
                filename = input("Enter filename: ")
                new_content = input("Enter new content: ")
                self.update_file(filename, new_content)
            else:
                print("Invalid command")

if __name__ == "__main__":
    mini_copilot = MiniCopilot()
    mini_copilot.run()
