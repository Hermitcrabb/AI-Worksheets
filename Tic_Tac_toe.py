def calculate_tic_tac_toe_heuristic(board, player, opponent):
    def is_open_line(line, player, opponent):
        """
        Checks if a line (row, column, or diagonal) is open for a player.
        A line is open if it contains only empty spaces or the player's mark.
        """
        return all(cell == player or cell == '' for cell in line)

    # Generate all rows, columns, and diagonals
    rows = board
    cols = [[board[r][c] for r in range(3)] for c in range(3)]
    diagonals = [[board[i][i] for i in range(3)], [board[i][2 - i] for i in range(3)]]

    lines = rows + cols + diagonals

    # Count open lines for player and opponent
    player_open_lines = sum(is_open_line(line, player, opponent) for line in lines)
    opponent_open_lines = sum(is_open_line(line, opponent, player) for line in lines)

    # Calculate heuristic value
    heuristic_value = player_open_lines - opponent_open_lines
    return heuristic_value


# Example Board States
board_1 = [['', '', ''], 
           ['', '', ''], 
           ['', '', '']]  # Empty board

board_2 = [['X', '', ''], 
           ['', '', ''], 
           ['', '', '']]  # Board with one "X"

board_3 = [['X', 'X', ''], 
           ['', '', ''], 
           ['', '', '']]  # Board with two "X" in the same row

# Calculate and Print Heuristic Values
e1 = calculate_tic_tac_toe_heuristic(board_1, 'X', 'O')
print("Heuristic Value for Empty Board (e1):", e1)

e2 = calculate_tic_tac_toe_heuristic(board_2, 'X', 'O')
print("Heuristic Value for Board with one 'X' (e2):", e2)

e3 = calculate_tic_tac_toe_heuristic(board_3, 'X', 'O')
print("Heuristic Value for Board with two 'X's in the same row (e3):", e3)
