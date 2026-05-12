class Train:
    def __init__(self, name, number, seats, fare):
        self.name = name
        self.number = number
        self.seats = seats
        self.fare = fare
        self.available_seats = seats
        self.passengers = []

    def book_ticket(self, passenger_name):
        if self.available_seats > 0:
            self.available_seats -= 1
            self.passengers.append(passenger_name)
            print("Ticket booked successfully!")
        else:
            print("Sorry, no seats available.")

    def cancel_ticket(self, passenger_name):
        if passenger_name in self.passengers:
            self.available_seats += 1
            self.passengers.remove(passenger_name)
            print("Ticket canceled successfully!")
        else:
            print("Passenger not found.")

    def get_details(self):
        return f"Name: {self.name}, Number: {self.number}, Total Seats: {self.seats}, Fare: {self.fare}, Available Seats: {self.available_seats}"

    def get_passengers(self):
        return self.passengers

    def get_fare(self):
        return self.fare

    def get_available_seats(self):
        return self.available_seats

    def delete_train(self, train_number, trains):
        if train_number in trains:
            del trains[train_number]
            print("Train deleted successfully!")
        else:
            print("Train not found.")


class RailwayReservationSystem:
    def __init__(self):
        self.trains = {}

    def add_train(self, name, number, seats, fare):
        self.trains[number] = Train(name, number, seats, fare)
        print("Train added successfully!")

    def book_ticket(self, train_number, passenger_name):
        if train_number in self.trains:
            self.trains[train_number].book_ticket(passenger_name)
        else:
            print("Train not found.")

    def cancel_ticket(self, train_number, passenger_name):
        if train_number in self.trains:
            self.trains[train_number].cancel_ticket(passenger_name)
        else:
            print("Train not found.")

    def display_trains(self):
        print("Available Trains:")
        for train in self.trains.values():
            print(train.get_details())

    def display_passengers(self, train_number):
        if train_number in self.trains:
            print(self.trains[train_number].get_passengers())
        else:
            print("Train not found.")

    def get_train_details(self, train_number):
        if train_number in self.trains:
            print(self.trains[train_number].get_details())
        else:
            print("Train not found.")

    def get_fare(self, train_number):
        if train_number in self.trains:
            print(self.trains[train_number].get_fare())
        else:
            print("Train not found.")

    def get_available_seats(self, train_number):
        if train_number in self.trains:
            print(self.trains[train_number].get_available_seats())
        else:
            print("Train not found.")

    def delete_train(self, train_number):
        if train_number in self.trains:
            del self.trains[train_number]
            print("Train deleted successfully!")
        else:
            print("Train not found.")


def main():
    railway_system = RailwayReservationSystem()
    while True:
        print("\nRailway Reservation System\n")
        print("1. Add Train")
        print("2. Book Ticket")
        print("3. Cancel Ticket")
        print("4. Display Trains")
        print("5. Display Passengers")
        print("6. Get Train Details")
        print("7. Get Fare")
        print("8. Get Available Seats")
        print("9. Delete Train")
        print("10. Exit")
        choice = input("Enter your choice: ")
        if choice == "1":
            name = input("Enter train name: ")
            number = input("Enter train number: ")
            seats = int(input("Enter total seats: "))
            fare = float(input("Enter fare: "))
            railway_system.add_train(name, number, seats, fare)
        elif choice == "2":
            train_number = input("Enter train number: ")
            passenger_name = input("Enter passenger name: ")
            railway_system.book_ticket(train_number, passenger_name)
        elif choice == "3":
            train_number = input("Enter train number: ")
            passenger_name = input("Enter passenger name: ")
            railway_system.cancel_ticket(train_number, passenger_name)
        elif choice == "4":
            railway_system.display_trains()
        elif choice == "5":
            train_number = input("Enter train number: ")
            railway_system.display_passengers(train_number)
        elif choice == "6":
            train_number = input("Enter train number: ")
            railway_system.get_train_details(train_number)
        elif choice == "7":
            train_number = input("Enter train number: ")
            railway_system.get_fare(train_number)
        elif choice == "8":
            train_number = input("Enter train number: ")
            railway_system.get_available_seats(train_number)
        elif choice == "9":
            train_number = input("Enter train number: ")
            railway_system.delete_train(train_number)
        elif choice == "10":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()