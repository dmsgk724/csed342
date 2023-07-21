from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

SEED=3

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        succ_v = min([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states])
        return succ_v

    legalActions = gameState.getLegalActions(0)
    succ_states = [gameState.generateSuccessor(0, action) for action in legalActions]
    succ_values = [recurse(succ_state, 1, self.depth) for succ_state in succ_states]
    maxIndex = succ_values.index(max(succ_values))
    return legalActions[maxIndex]    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        succ_v = min([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states])
        return succ_v
    
    succ_state = gameState.generateSuccessor(0, action)
    succ_value = recurse(succ_state, 1, self.depth)
    return succ_value
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        succ_v = sum([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states]) / len(succ_states) # == / len(legalActions) 일 듯
        return succ_v

    legalActions = gameState.getLegalActions(0)
    succ_states = [gameState.generateSuccessor(0, action) for action in legalActions]
    succ_values = [recurse(succ_state, 1, self.depth) for succ_state in succ_states]
    maxIndex = succ_values.index(max(succ_values))
    return legalActions[maxIndex]    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        succ_v = sum([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states]) / len(succ_states) # == / len(legalActions) 일 듯
        return succ_v

    succ_state = gameState.generateSuccessor(0, action)
    succ_value = recurse(succ_state, 1, self.depth)
    return succ_value   
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        stop_v = 0.5 * recurse(state.generateSuccessor(agentIndex, Directions.STOP), nextAgentIndex, nextDepth) if Directions.STOP in legalActions else 0
        succ_v = 0.5 * sum([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states]) / len(succ_states) # == / len(legalActions) 일 듯
        return succ_v + stop_v

    legalActions = gameState.getLegalActions(0)
    succ_states = [gameState.generateSuccessor(0, action) for action in legalActions]
    succ_values = [recurse(succ_state, 1, self.depth) for succ_state in succ_states]
    maxIndex = succ_values.index(max(succ_values))
    return legalActions[maxIndex]    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        stop_v = 0.5 * recurse(state.generateSuccessor(agentIndex, Directions.STOP), nextAgentIndex, nextDepth) if Directions.STOP in legalActions else 0
        succ_v = 0.5 * sum([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states]) / len(succ_states) # == / len(legalActions) 일 듯
        return succ_v + stop_v

    succ_state = gameState.generateSuccessor(0, action)
    succ_value = recurse(succ_state, 1, self.depth)
    return succ_value   
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        if(agentIndex%2 == 0): # even -> random policy
          succ_v = sum([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states]) / len(succ_states) # == / len(legalActions) 일 듯
        else: # odd -> min policy
          succ_v = min([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states])
        
        return succ_v

    legalActions = gameState.getLegalActions(0)
    succ_states = [gameState.generateSuccessor(0, action) for action in legalActions]
    succ_values = [recurse(succ_state, 1, self.depth) for succ_state in succ_states]
    maxIndex = succ_values.index(max(succ_values))
    return legalActions[maxIndex] 
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        succ_v = max([recurse(succ_state, nextAgentIndex, depth) for succ_state in succ_states])
        return succ_v
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        if(agentIndex%2 == 0): # even -> random policy
          succ_v = sum([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states]) / len(succ_states) # == / len(legalActions) 일 듯
        else: # odd -> min policy
          succ_v = min([recurse(succ_state, nextAgentIndex, nextDepth) for succ_state in succ_states])
        
        return succ_v
    
    succ_state = gameState.generateSuccessor(0, action)
    succ_value = recurse(succ_state, 1, self.depth)
    return succ_value  
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth, alpha, beta):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        newV = float("-inf")
        for succ_state in succ_states:
          newV = max(newV, recurse(succ_state, nextAgentIndex, depth, alpha, beta))
          alpha = max(alpha, newV)
          if alpha > beta:
            break
        return newV
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        if(agentIndex%2 == 0): # even -> random policy
          acc = 0
          for succ_state in succ_states:
            newV = recurse(succ_state, nextAgentIndex, nextDepth, alpha, beta)
            acc = acc + newV
          average = acc / len(succ_states)
          beta = min(beta, average)
          return average
        else: # odd -> min policy
          newV = float("inf")
          for succ_state in succ_states:
            newV = min(newV, recurse(succ_state, nextAgentIndex, nextDepth, alpha, beta))
            beta = min(beta, newV)
            if alpha > beta:
              break
          return newV

    maxV = float("-inf")
    maxIndex = 0
    i = 0
    legalActions = gameState.getLegalActions(0)
    succ_states = [gameState.generateSuccessor(0, action) for action in legalActions]
    for succ_state in succ_states:
      newV = recurse(succ_state, 1, self.depth, float("-inf"), float("inf"))
      maxV, maxIndex = (newV, i) if maxV < newV else (maxV, maxIndex)
      i = i + 1

    return legalActions[maxIndex]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def recurse(state, agentIndex, depth, alpha, beta):
      if(depth==0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      succ_states = [state.generateSuccessor(agentIndex, action) for action in legalActions]
      if(agentIndex==0):
        nextAgentIndex = 1
        newV = float("-inf")
        for succ_state in succ_states:
          newV = max(newV, recurse(succ_state, nextAgentIndex, depth, alpha, beta))
          alpha = max(alpha, newV)
          if alpha > beta:
            break
        return newV
      else:
        nextAgentIndex, nextDepth = (agentIndex + 1, depth) if agentIndex + 1 < state.getNumAgents() else (0, depth-1)
        if(agentIndex%2 == 0): # even -> random policy
          acc = 0
          for succ_state in succ_states:
            newV = recurse(succ_state, nextAgentIndex, nextDepth, alpha, beta)
            acc = acc + newV
          average = acc / len(succ_states)
          beta = min(beta, average)
          return average
        else: # odd -> min policy
          newV = float("inf")
          for succ_state in succ_states:
            newV = min(newV, recurse(succ_state, nextAgentIndex, nextDepth, alpha, beta))
            beta = min(beta, newV)
            if alpha > beta:
              break
          return newV

    succ_state = gameState.generateSuccessor(0, action)
    sV = recurse(succ_state, 1, self.depth, float("-inf"), float("inf"))
    return sV
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """
  

  # BEGIN_YOUR_ANSWER
  features = []
  pacmanPos = currentGameState.getPacmanPosition()
  numFood = currentGameState.getNumFood()
  isEveryGhostWeak = True
  
  # current score
  features.append(currentGameState.getScore())

  # L1 distance with the closest food
  foodPosList = currentGameState.getFood().asList()
  foodDistList = []
  for foodPos in foodPosList:
    L1dist = manhattanDistance(pacmanPos, foodPos)
    foodDistList.append(L1dist)
  if(len(foodDistList)!=0):
    features.append(1/min(foodDistList))
  else:
    features.append(0)
  
  # L1 distance with the closest capsule
  capsulePosList = currentGameState.getCapsules()
  capsuleDistList = []
  for capsulePos in capsulePosList:
    L1dist = manhattanDistance(pacmanPos, capsulePos)
    capsuleDistList.append(L1dist)
  if(len(capsuleDistList)!=0):
    features.append(1/max(capsuleDistList))
  else:
    features.append(0)
  
  # L1 distance with ghosts
  ghostStateList = currentGameState.getGhostStates()
  weakenGhostDistList = []
  normalGhostDistList = []
  for ghostState in ghostStateList:
    ghostPos = ghostState.getPosition()
    L1dist = manhattanDistance(pacmanPos, ghostPos)
    if (ghostState.scaredTimer != 0):
      weakenGhostDistList.append(ghostState.scaredTimer/L1dist)
    else:
      normalGhostDistList.append(L1dist)
  if(len(weakenGhostDistList)!=0):
    features.append(min(weakenGhostDistList))
  else:
    isEveryGhostWeak = False
    features.append(0)
  if(len(normalGhostDistList)!=0):
    features.append(min(normalGhostDistList))
  else:
    features.append(0)

  # score, closest food, closest capsule, weakenghost, normalghost
  # weights = [1, 1/(numFood ** 2), 5, 16, 0.8]
  weights=[]
  weights.append(1)
  weights.append(1/(numFood ** 2)) if numFood!=0 else weights.append(0)
  weights.append(0) if isEveryGhostWeak else weights.append(5)
  weights.append(16)
  weights.append(0.8)
  value = sum([w*f for w, f in zip(weights, features)])
  return value  
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  # return 'MinimaxAgent'
  # return 'AlphaBetaAgent'
  # return 'ExpectiminimaxAgent'
  return 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
