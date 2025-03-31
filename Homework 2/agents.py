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
    
    return random.choice(list(your_function(grid, 4, True, -np.inf, np.inf)[1]))


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
    now=board.mark
    opponent=3-board.mark

    #we win
    if board.win(now):
      return 1e10
    #opponent win
    if board.win(opponent):
      return -1e10
    
    score=0

    #check for forced win
    for col in board.valid:
      if game.check_winning_move(board, col, now):
        score+=1e9
      
      if game.check_winning_move(board, col, opponent):
        score-=1e9

    #center control strategy
    center=board.column//2
    cen_weight=3.0

    #weight the cols
    for col in range(board.column):
      array=[board.table[row][col] for row in range(board.row)]
      weight=cen_weight-abs(col-center)*0.5
      score+=weight*array.count(now)
      #weight * 我們的棋子數
      score-=weight*array.count(opponent)
      

    patterns={
      #[當前玩家的棋子數量, empty num, value]
      'two':[2,2,5],
      'three':[3,1,50],
      'opp_two':[0,2,-4],
      'opp_three':[0,1,-40]
    }

    #evaluate patterns
    score += evaluate_pattern(board, now, opponent, patterns)
    #detect multiple winning threats
    score +=1000* double_threats(board, now)
    score -= 1200* double_threats(board, opponent)

    return score
    
def evaluate_pattern(board, player, opponent, patterns):
    score=0
    directions=[
      (1,0),
      (0,1),
      (1,1),
      (1,-1)
    ]

    for x,y in directions:
      for r in range(board.row):
        for c in range(board.column):
          if(r+3*x>=board.row or r+3*x<0 or c+3*y>=board.column or c+3*y<0):
            continue

          window=[]
          for i in range (4):
            window.append(board.table[r+i*x][c+i*y])

          p_count=window.count(player)
          o_count=window.count(opponent)
          empty=window.count(0)

          if o_count==0:
            if p_count==2 and empty==2:
              score+=patterns['two'][2]
            elif p_count==3 and empty==1:
              score+=patterns['three'][2]
          
          if p_count==0:
            if o_count==2 and empty==2:
              score+= patterns['opp_two'][2]
            elif o_count==3 and empty==1:
              score+= patterns['opp_three'][2]

    return score

def double_threats(board, player):
    # Only count immediate threats
    winning_moves = 0
    for col in board.valid:
        if game.check_winning_move(board, col, player):
            winning_moves += 1
    
    # If there's more than one way to win, it's a double threat
    return 1 if winning_moves > 1 else 0

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

