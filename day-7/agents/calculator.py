# Calculator Program

class Calculator:
    def __init__(self):
        pass

    # Method to add two numbers
    def add(self, num1, num2):
        return num1 + num2

    # Method to subtract two numbers
    def subtract(self, num1, num2):
        return num1 - num2

    # Method to multiply two numbers
    def multiply(self, num1, num2):
        return num1 * num2

    # Method to divide two numbers
    def divide(self, num1, num2):
        if num2 == 0:
            raise ZeroDivisionError("Cannot divide by zero!")
        return num1 / num2

    # Method to get user input and perform calculations
    def calculate(self):
        while True:
            try:
                num1 = float(input("Enter the first number: "))
                operator = input("Enter the operator (+, -, *, /): ")
                num2 = float(input("Enter the second number: "))

                if operator == "+":
                    print(f"{num1} + {num2} = {self.add(num1, num2)}")
                elif operator == "-":
                    print(f"{num1} - {num2} = {self.subtract(num1, num2)}")
                elif operator == "*":
                    print(f"{num1} * {num2} = {self.multiply(num1, num2)}")
                elif operator == "/":
                    print(f"{num1} / {num2} = {self.divide(num1, num2)}")
                else:
                    print("Invalid operator! Please enter +, -, *, or /")
            except ValueError:
                print("Invalid input! Please enter a number.")
            except ZeroDivisionError as e:
                print(e)

            cont = input("Do you want to continue? (y/n): ")
            if cont.lower() != "y":
                break

if __name__ == "__main__":
    calculator = Calculator()
    calculator.calculate()