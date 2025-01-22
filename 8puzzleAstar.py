import heapq
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

    def manhattan_distance(self):
        distance = 0
        for x in range(self.size):
            for y in range(self.size):
                value = self.board[x, y]
                if value != 0:
                    target_x, target_y = divmod(value - 1, self.size)
                    distance += abs(x - target_x) + abs(y - target_y)
        return distance

    def __lt__(self, other):
        return True  # Required for heapq to compare puzzles

    def __repr__(self):
        return str(self.board)

def a_star_solver(start_board):
    start_puzzle = Puzzle(start_board)
    open_set = []
    heapq.heappush(open_set, (0, start_puzzle))

    came_from = {}
    g_score = {start_puzzle: 0}
    f_score = {start_puzzle: start_puzzle.manhattan_distance()}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current.is_goal():
            return reconstruct_path(came_from, current)

        for neighbor in current.neighbors():
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + neighbor.manhattan_distance()

                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    path.reverse()
    return path

# Example usage
start_board = [
    [1, 8, 3],
    [5, 7, 6],
    [0, 4, 2]
]

solution = a_star_solver(start_board)
if solution:
    print("Solution:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
