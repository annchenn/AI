import numpy as np
import random
import game


def print_INFO():
    """
    Prints your homework submission details.
    Please replace the placeholders (date, name, student ID) with valid information
    before submitting.
    """
    print(
        """========================================
        DATE: 2025/03/30
        STUDENT NAME: 陳璽安
        STUDENT ID: 112550184
        ========================================
        """)


#
# Basic search functions: Minimax and Alpha‑Beta
#

def minimax(grid, depth, maximizingPlayer, dep=4):
    """
    TODO (Part 1): Implement recursive Minimax search for Connect Four.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
    """
    # Placeholder return to keep function structure intact
    if depth==0 or grid.terminate():
      return get_heuristic(grid) , set()
    
    move = grid.valid
    
    if not move:
      return 0, set()
    
    candidate=set()
    value=0
    
    if maximizingPlayer:
      value=float('-inf')
      for col in move:
        next=game.drop_piece(grid, col)
        next_value, _ = minimax(next, depth-1, False, dep)
        if next_value>value:
          value=next_value
          candidate={col}
        elif next_value==value:
          candidate.add(col)
          
    else:
      value=float('inf')
      for col in move:
        next=game.drop_piece(grid, col)
        next_value, _= minimax(next, depth-1, True, dep)
        if next_value<value:
          candidate={col}
          value=next_value
        elif next_value==value:
          candidate.add(col)
    
    return value, candidate
    

def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    TODO (Part 2): Implement Alpha-Beta pruning as an optimization to Minimax.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
      - Prune branches when alpha >= beta
    """
    # Placeholder return to keep function structure intact
    if depth==0 or grid.terminate():
      return get_heuristic(grid) , set()
    
    move = grid.valid
    
    if not move:
      return 0, set()
    
    candidate=set()
    value=0
    
    if maximizingPlayer:
      value=float('-inf')
      for col in move:
        next=game.drop_piece(grid, col)
        next_value, _ = alphabeta(next, depth-1, False, alpha, beta, dep)
        if next_value>value:
          value=next_value
          candidate={col}
        elif next_value==value:
          candidate.add(col)
        if value>alpha:
          alpha=value
        if beta<=alpha:
          break
          
    else:
      value=float('inf')
      for col in move:
        next=game.drop_piece(grid, col)
        next_value, _= alphabeta(next, depth-1, True, alpha, beta, dep)
        if next_value<value:
          candidate={col}
          value=next_value
        elif next_value==value:
          candidate.add(col)
        if value<beta:
          beta=value
        if beta<=alpha:
          break
          
    return value, candidate

#
# Basic agents
#

def agent_minimax(grid):
    """
    Agent that uses the minimax() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(minimax(grid, 4, True)[1]))


def agent_alphabeta(grid):
    """
    Agent that uses the alphabeta() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))


def agent_reflex(grid):
    """
    A simple reflex agent provided as a baseline:
      - Checks if there's an immediate winning move.
      - Otherwise picks a random valid column.
    """
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)


def agent_strong(grid):
    """
    TODO (Part 3): Design your own agent (depth = 4) to consistently beat the Alpha-Beta agent (depth = 4).
    This agent will typically act as Player 2.
    """
    # Placeholder logic that calls your_function().
    
    return random.choice(list(your_function(grid, 4, False, -np.inf, np.inf)[1]))


#
# Heuristic functions
#



def get_heuristic(board):
    """
    Evaluates the board from Player 1's perspective using a basic heuristic.

    Returns:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
      - Intermediate scores based on partial connect patterns
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score


def get_heuristic_strong(board):
    """
    TODO (Part 3): Implement a more advanced board evaluation for agent_strong.
    Currently a placeholder that returns 0.
    """
    now=2
    opponent=1

    #we win
    if board.win(now):
      return -1e10
    #opponent win
    if board.win(opponent):
      return 1e10
    
    score=0

    immediate_win = False
    immediate_loss = False
    
    for col in board.valid:
        # Check if we can win in the next move
        if game.check_winning_move(board, col, now):
            immediate_win = True
            score -= 1000000
        
        # Check if opponent can win in the next move
        if game.check_winning_move(board, col, opponent):
            immediate_loss = True
            score += 1200000  # Higher penalty to prioritize blocking
    
    # If we have both an immediate win and loss, prioritize winning
    if immediate_win and immediate_loss:
        score -= 500000
  

    #center control strategy
    center=board.column//2

    center_score = 0
    for row in range(board.row):
        if board.table[row][center] == now:
            center_score -= 7 * (row + 1)  # Higher weight for lower positions
        elif board.table[row][center] == opponent:
            center_score += 8 * (row + 1)  # Even higher penalty for opponent center control
    
    score -= center_score * 10

    #weight the cols
    for col in range(board.column):
      col_weight = 6 - abs(col - center)
      for row in range (board.row):
        row_weight= row +1
        if board.table[row][col] == now:
            score -= col_weight *row_weight* 3
        elif board.table[row][col] == opponent:
            score += col_weight *row_weight* 3.5
              
    for piece_count in [2, 3]:
      # Our pieces
      window_count = game.count_windows(board, piece_count, now)
      score -= window_count * (80 if piece_count == 2 else 800)
      
      # Opponent pieces
      opp_window_count = game.count_windows(board, piece_count, opponent)
      score += opp_window_count * (100 if piece_count == 2 else 1000)
      
      
    three_in_row_cols = 0
    for col in board.valid:
      next_board = game.drop_piece(board, col)
      if game.count_windows(next_board, 3, now) > game.count_windows(board, 3, now):
        three_in_row_cols += 1
    
    if three_in_row_cols > 1:
      score -= 10000  # Bonus for creating multiple threats
    
    
    return score
    


def your_function(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    A stronger search function that uses get_heuristic_strong() instead of get_heuristic().
    You can employ advanced features (e.g., improved move ordering, deeper lookahead).

    Return:
      (boardValue, {setOfCandidateMoves})

    Currently a placeholder returning (0, {0}).
    """
    if depth==0 or grid.terminate():
      return get_heuristic_strong(grid) , set()
    
    move = grid.valid
    
    if not move:
      return 0, set()
    
    center = grid.column // 2
    # Sort columns by distance from center (center first)
    move = sorted(move, key=lambda col: abs(col - center))
    
    candidate=set()
    value=0
    
    if maximizingPlayer:
      value=float('-inf')
      for col in move:
        next=game.drop_piece(grid, col)
        next_value, _ = your_function(next, depth-1, False, alpha, beta, dep)
        if next_value>value:
          value=next_value
          candidate={col}
        elif next_value==value:
          candidate.add(col)
        if value>alpha:
          alpha=value
        if beta<=alpha:
          break
          
    else:
      value=float('inf')
      for col in move:
        next=game.drop_piece(grid, col)
        next_value, _= your_function(next, depth-1, True, alpha, beta, dep)
        if next_value<value:
          candidate={col}
          value=next_value
        elif next_value==value:
          candidate.add(col)
        if value<beta:
          beta=value
        if beta<=alpha:
          break
          
    
    return value, candidate

