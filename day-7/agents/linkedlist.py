class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        if not self.head:
            self.head = Node(data)
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = Node(data)

    def print_list(self):
        current = self.head
        while current:
            print(current.data)
            current = current.next

# Create a linked list and insert some data
linked_list = LinkedList()
linked_list.insert('A')
linked_list.insert('B')
linked_list.insert('C')

# Print the linked list
linked_list.print_list()