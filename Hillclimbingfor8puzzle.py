import numpy as np

class Puzzle:
    def __init__(self, board):
        self.board = np.array(board)
        self.size = self.board.shape[0]
        self.goal = np.arange(1, self.size ** 2 + 1).reshape((self.size, self.size))
        self.goal[-1, -1] = 0  # Goal state ends with 0 (empty tile)

    def find_empty(self):
        return tuple(np.argwhere(self.board == 0)[0])

    def is_goal(self):
        return np.array_equal(self.board, self.goal)

    def neighbors(self):
        empty_x, empty_y = self.find_empty()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
        neighbors = []

        for dx, dy in directions:
            new_x, new_y = empty_x + dx, empty_y + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                new_board = self.board.copy()
                new_board[empty_x, empty_y], new_board[new_x, new_y] = new_board[new_x, new_y], new_board[empty_x, empty_y]
                neighbors.append(Puzzle(new_board))

        return neighbors

    def heuristic(self):
        """
        Heuristic function: Manhattan distance.
        """
        distance = 0
        for x in range(self.size):
            for y in range(self.size):
                value = self.board[x, y]
                if value != 0:
                    target_x, target_y = divmod(value - 1, self.size)
                    distance += abs(x - target_x) + abs(y - target_y)
        return distance

    def __repr__(self):
        return str(self.board)

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        return hash(self.board.tostring())

def steepest_ascent_solver(start_board):
    current = Puzzle(start_board)
    while not current.is_goal():
        neighbors = current.neighbors()
        next_state = min(neighbors, key=lambda p: p.heuristic())

        # If no better state is found, terminate.
        if next_state.heuristic() >= current.heuristic():
            print("Stuck at a local optimum.")
            return None

        current = next_state
        print(f"Current state:\n{current}\nHeuristic: {current.heuristic()}\n")

    return current

# Example usage
start_board = [
    [1, 8, 3],
    [5, 0, 6],
    [7, 4, 2]
]

solution = steepest_ascent_solver(start_board)
if solution:
    print("Solution found:")
    print(solution)
else:
    print("No solution found.")
