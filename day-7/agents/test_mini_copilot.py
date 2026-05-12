from mini_copilot import MiniCopilot

copilot = MiniCopilot()

copilot.create_file('example.txt', 'This is an example file.')
print(copilot.read_file('example.txt'))
copilot.update_file('example.txt', 'This is an updated example file.')
print(copilot.read_file('example.txt'))