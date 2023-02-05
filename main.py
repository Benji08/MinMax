import random
import time
import matplotlib.pyplot as plt
import numpy as np

random.seed(11)


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if len(vector) % 2 == 0:
            left = sum(vector[::2])
            right = sum(vector) - left
            if left >= right:
                self.numbers.append(vector[0])
                return vector[1:]
            self.numbers.append(vector[-1])
            return vector[:-1]
        else:
            left = max(sum(vector[1::2]), sum(vector[2::2]))
            right = max(sum(vector[:-1:2]), sum(vector[:-2:2]))
            if left >= right:
                self.numbers.append(vector[-1])
                return vector[:-1]
            self.numbers.append(vector[0])
            return vector[1:]


class MinMaxAgent:
    def __init__(self, max_depth=50):
        self.numbers = []
        self.max_depth = max_depth - 1

    def minmax(self, vector, actual_depth=0, whose_move=1):
        if len(vector) == 1:
            return [vector[0]*whose_move, False]
        elif actual_depth == self.max_depth:
            if whose_move == 1:
                return [max(vector[0], vector[-1]), max(vector[0], vector[-1]) == vector[0]]
            elif whose_move == -1:
                return [-max(vector[0], vector[-1]), min(vector[0], vector[-1]) == vector[0]]
        left_value = self.minmax(vector[1:], actual_depth+1, whose_move * -1)[0]
        right_value = self.minmax(vector[:-1], actual_depth + 1, whose_move * -1)[0]
        left_value += whose_move * vector[0]
        right_value += whose_move * vector[-1]
        if whose_move == 1:
            return [max(right_value, left_value), max(right_value, left_value) == left_value]
        else:
            return [min(right_value, left_value), min(right_value, left_value) == left_value]

    def act(self, vector: list):
        turn = self.minmax(vector)
        if turn[1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def histogram(list_of_points):
    plt.hist(list_of_points, bins=100)
    plt.show()


def main():
    sum_of_time = 0
    list_of_points_player1 = np.array([])
    list_of_points_player2 = np.array([])
    depth = 50
    for i in range(500):
        t = time.time()
        vector = [random.randint(-10, 10) for _ in range(15)]
        first_agent, second_agent = MinMaxAgent(depth), MinMaxAgent()
        run_game(vector, first_agent, second_agent)
        sum_of_time += (time.time() - t)
        list_of_points_player1 = np.insert(list_of_points_player1,
                                           len(list_of_points_player1), sum(first_agent.numbers))
        list_of_points_player2 = np.insert(list_of_points_player2,
                                           len(list_of_points_player2), sum(second_agent.numbers))

        t = time.time()
        vector = [random.randint(-10, 10) for _ in range(15)]
        first_agent, second_agent = MinMaxAgent(depth), MinMaxAgent()
        run_game(vector, second_agent, first_agent)
        sum_of_time += (time.time() - t)
        list_of_points_player1 = np.insert(list_of_points_player1,
                                           len(list_of_points_player1), sum(first_agent.numbers))
        list_of_points_player2 = np.insert(list_of_points_player2,
                                           len(list_of_points_player2), sum(second_agent.numbers))

    #histogram(list_of_points_player1)
    avg_time = sum_of_time / 1000
    avg_player1 = sum(list_of_points_player1) / 1000
    avg_player2 = sum(list_of_points_player2) / 1000
    deviation_of_player1 = list_of_points_player1.std()
    deviation_of_player2 = list_of_points_player2.std()

    print(f"średni czas wykonania programu: {avg_time}\n"
          f"średnia suma punktów gracza1: {round(avg_player1, 2)}"
          f", średnia suma punktów gracza2: {round(avg_player2, 2)} \n"
          f"średnie odchylenie gracza1: {round(deviation_of_player1, 2)}"
          f", średnie odchylenie gracza2: {round(deviation_of_player2, 2)}")


if __name__ == "__main__":
    main()
