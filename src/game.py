import numpy as np

corner_heuristic = np.array([[[4,2,1,0],[2,1,0,-1],[1,0,-1,-2],[0,-1,-2,-4]]])
corner_heuristic = np.concatenate((corner_heuristic, np.rot90(corner_heuristic, 1, axes=(1,2)), np.rot90(corner_heuristic, 2, axes=(1,2)), np.rot90(corner_heuristic, 3, axes=(1,2))), axis=0)

snake_heuristic = np.array([[[1,2,3,4],[8,7,6,5],[9,10,11,12],[16,15,14,13]]])
snake_heuristic = np.concatenate((snake_heuristic, np.rot90(snake_heuristic, 1, axes=(1,2)), np.rot90(snake_heuristic, 2, axes=(1,2)), np.rot90(snake_heuristic, 3, axes=(1,2)),
                                  np.transpose(snake_heuristic, axes=(0,2,1)), np.rot90(np.transpose(snake_heuristic, axes=(0,2,1)), 1, axes=(1,2)),
                                  np.rot90(np.transpose(snake_heuristic, axes=(0,2,1)), 2, axes=(1,2)), np.rot90(np.transpose(snake_heuristic, axes=(0,2,1)), 3, axes=(1,2))), axis=0)

class Game():
  def __init__(self, target=2048, size=4, initial_values=2, eval=0, transposition=0):
    ## inital_values : number of value to be put in the board before starting the game
    self.board = np.zeros((size,size))
    self.target = target # value to reach
    self.size = size # length of a board side
    self.eval = eval #indicates the type of evaluation (0:score, 1:#moves, 2:corner heuristic, 3:snake heuristic)
    self.transposition = transposition #<=0: no transposition table, >0:threshold
    self.score = 0 # current score of the game
    self.moves = 0 # current number of moves played
    self.won = False # True if the target is reached
    self.table = {} # transposition table : hash -> [scores sum, #playouts]
    self.hash_table = np.zeros((self.size**2, int(np.log2(self.target))), dtype="int64") # Zobrist hashing table (works only if the game is terminated when the target is reached)
    
    if self.transposition > 0:
      for i in range(self.hash_table.shape[0]):
        for j in range(self.hash_table.shape[1]):
          self.hash_table[i,j] = np.random.randint(1,2**63)
 
    for v in range(initial_values):
      self.add()

 
  def __str__(self):
    ## return : a representation of the game
    return str(self.board)
 
  def deepcopy(self):
    ## return a copy of the game (mix of shallow and deep)
    game = Game(self.target, self.size, 0, self.eval, 0)
    game.board = self.board.copy()
    game.score = self.score
    game.moves = self.moves
    game.won = self.won
    game.transposition = self.transposition
    game.hash_table = self.hash_table.copy()
    game.table = self.table # no copy because initial game and derived playouts share the same transposition table
 
    return game

  def hashcode(self):
    ## return : minimum value of symmetric boards hash
    hashes = []
    board = self.board.copy()
    for t in range(2): # transposition
      for r in range(4): # rotation
        hash = 0
        for i in range(self.size):
          for j in range(self.size):
            if board[i,j] > 0:
              hash = hash ^ self.hash_table[i*self.size+j, int(np.log2(board[i,j]))-1]

        hashes.append(hash)
        board = np.rot90(board)

      board = board.T

    return np.min(hashes)
 
  def evaluate(self):
    ## return : evaluation of the game
    if self.eval==0:
      return self.score
    elif self.eval==1:
      return self.moves
    elif self.eval == 2:
      return np.max(np.sum(corner_heuristic * self.board, axis=(1,2)))
    elif self.eval == 3: 
      return np.max(np.sum(snake_heuristic * self.board, axis=(1,2)))
    else:
      print('Warning: value {} for eval is not defined, using score by default'.format(self.eval))
      return self.score
 
  def possible_moves(self):
    ## return : list of possible move among (0,1,2,3)
    moves = set()
    up = down = left = right = False
    board = self.board
    for i in range(self.size):
      for j in range(self.size):
        if i > 0:
          if (not up) and ((board[i-1,j] == 0 and board[i,j] != 0) or ((board[i-1,j] == board[i,j]) and (board[i,j] != 0))):
            moves.add(0)
            up = True

          if (not down) and ((board[i,j] == 0 and board[i-1,j] != 0) or ((board[i,j] == board[i-1,j]) and (board[i,j] != 0))):
            moves.add(2)
            down = True
 
        if j > 0:
          if (not right) and ((board[i,j] == 0 and board[i,j-1] != 0) or ((board[i,j] == board[i,j-1]) and (board[i,j] != 0))):
            moves.add(1)
            right = True

          if (not left) and ((board[i,j-1] == 0 and board[i,j] != 0) or ((board[i,j-1] == board[i,j]) and (board[i,j] != 0))):
            moves.add(3)
            left = True
            
 
        if up and right and down and left:
          return list(moves)
 
    return list(moves)
 
  def add(self):
    ## add a 2 or a 4 tile in a random empty cell
    ## return : the added tile value
    if np.random.binomial(1, 0.9):
      newTile = 2
    else:
      newTile = 4
      
    I = np.random.choice(self.size, self.size, False)
    J = np.random.choice(self.size, self.size, False)
 
    for i in I:
      for j in J:
        if self.board[i,j] == 0:
          self.board[i,j] = newTile
          return newTile
        
    return -1
 
  def slide(self, move):
    ## move : direction of sliding # 0:up, 1:right, 2:down, 3:left
    modif = [0,1,2,3]
    if move == 0:
      while len(modif) > 0:
        temp = set()
        for j in modif:
          for i in range(1,self.size):
            if (self.board[i-1,j] == 0) and (self.board[i,j] != 0):
              self.board[i-1,j] = self.board[i,j]
              self.board[i,j] = 0
              temp.add(j)
        modif = list(temp)
 
    elif move == 1:
      while len(modif) > 0:
        temp = set()
        for i in modif:
          for j in range(1,self.size):
            if (self.board[i,j] == 0) and (self.board[i,j-1] != 0):
              self.board[i,j] = self.board[i,j-1]
              self.board[i,j-1] = 0
              temp.add(i)
        modif = list(temp)
 
    elif move == 2:
      while len(modif) > 0:
        temp = set()
        for j in modif:
          for i in range(1,self.size):
            if (self.board[i,j] == 0) and (self.board[i-1,j] != 0):
              self.board[i,j] = self.board[i-1,j]
              self.board[i-1,j] = 0
              temp.add(j)
        modif = list(temp)
 
    elif move == 3:
      while len(modif) > 0:
        temp = set()
        for i in modif:
          for j in range(1,self.size):
            if (self.board[i,j-1] == 0) and (self.board[i,j] != 0):
              self.board[i,j-1] = self.board[i,j]
              self.board[i,j] = 0
              temp.add(i)
        modif = list(temp)

 
  def merge(self, move):
    ## move : direction of merging # 0:up, 1:right, 2:down, 3:left
    ## return : True if a merge occured, False otherwise
    modif = False
    if move == 0: 
      for j in range(self.size):
        i = 0
        while i < self.size-1:
          if self.board[i,j] == self.board[i+1,j]:
            if self.board[i,j] != 0:
              modif = True
              self.board[i,j] *= 2 
              self.board[i+1,j] = 0
              self.score += self.board[i,j]
            i += 1
          
          i += 1
 
    elif move == 1: 
      for i in range(self.size):
        j = self.size - 1
        while j > 0:
          if self.board[i,j] == self.board[i,j-1]:
            if self.board[i,j] != 0:
              modif = True
              self.board[i,j] *= 2 
              self.board[i,j-1] = 0
              self.score += self.board[i,j]
            j -= 1
            
          
          j -= 1
 
    elif move == 2: 
      for j in range(self.size):
        i = self.size - 1
        while i > 0:
          if self.board[i,j] == self.board[i-1,j]:
            if self.board[i,j] != 0:
              modif = True
              self.board[i,j] *= 2 
              self.board[i-1,j] = 0
              self.score += self.board[i,j]
            i -= 1
            
          
          i -= 1
 
    elif move == 3: 
      for i in range(self.size):
        j = 0
        while j < self.size-1:
          if self.board[i,j] == self.board[i,j+1]:
            if self.board[i,j] != 0:
              modif = True
              self.board[i,j] *= 2 
              self.board[i,j+1] = 0
              self.score += self.board[i,j]
            j += 1

          
          j += 1
 
    return modif
 
  def play(self, move):
    ## move : move to play 0:up, 1:right, 2:down, 3:left
    ## return : the value of the random tile added after the move
    self.slide(move)
    if self.merge(move):
      self.slide(move)

    newTile = self.add()
    self.moves += 1

    if self.target in self.board:
      self.won = True

    return newTile
    
      
  def playout(self, depth=np.Inf):
    ## depth : maximum number of moves
    ## return : the score achieved
    moves = self.possible_moves()
    if (len(moves) == 0) or (depth == 0) or self.won:
      return self.evaluate()
    else:
      hash = self.hashcode()
      move = np.random.choice(moves)
      self.play(move)
      score = self.playout(depth-1)
      if self.transposition > 0:
        if hash not in self.table.keys():
          self.table[hash] = [score, 1]
        else:
          self.table[hash][0] += score
          self.table[hash][1] += 1

      return score

 
 
  def bestMoveMCS(self, playouts=1): 
    ## playouts : number of playouts per possible move
    ## return : best mean score and associated move
    moves = self.possible_moves()
    bestMove = -1
    bestScore = -1
    for move in moves:
      summedScore = 0.0
      for i in range(playouts):
        game = self.deepcopy()
        tile = game.play(move)
        
        if self.transposition > 0:
          hash = game.hashcode()
          if (hash in self.table) and (self.table[hash][1] >= self.transposition):
            summedScore += self.table[hash][0] / self.table[hash][1]
          else:
            summedScore += game.playout()
        else:
          summedScore += game.playout()
       

      if summedScore / playouts > bestScore:
        bestScore = summedScore / playouts
        bestMove = move
    
    return bestScore, bestMove
 
  def MCS(self, playouts=4):
    ## playouts : number of playouts per possible move
    ## return : final evaluation of the game
    while True:
      game = self.deepcopy()
      _, bestMove = game.bestMoveMCS(playouts)
      if (bestMove == -1) or self.won:
        return self.evaluate()
      self.play(bestMove)
 
  def bestMoveNMCS(self, level=1):
    ## level : level of nesting (>=1)
    ## return : best mean score and associated move
    moves = self.possible_moves()
    if (len(moves) == 0) or self.won:
      return self.evaluate(), -1
    if level == 0:
      return self.playout(), -1
 
    bestMove = -1
    bestScore = -1
    summedScore = 0.0 # pour moyenne
    for move in moves:
      game = self.deepcopy()
      game.play(move)
      if self.transposition > 0:
        hash = game.hashcode()
        if (hash in self.table) and (self.table[hash][1] >= self.transposition*4**(level-1)):
          score = self.table[hash][0] / self.table[hash][1]
        else:
          score,_ = game.bestMoveNMCS(level-1)
      else:
        score,_ = game.bestMoveNMCS(level-1)
        
      summedScore += score
      if score > bestScore:
        bestMove = move
        bestScore = score
    
    return summedScore / len(moves), bestMove
 
  def NMCS(self, level):
    ## level : level of nesting (>=1)
    ## return : final score of the game
    while True:
      game = self.deepcopy()
      _, bestMove = game.bestMoveNMCS(level)
      if bestMove == -1 or self.won:
        return self.evaluate()
      self.play(bestMove)
