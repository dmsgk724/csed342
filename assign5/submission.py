from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



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
    가장 좋은 Action을 return한다.
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
    특정 agent가 action을 취했을 때 successor state를 return한다.
        
    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector
    #pacman Agent state를 return한다.
    

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts
      # ghost들의 state를 return한다.

    gameState.getNumAgents():
        Returns the total number of agents in the game
    Agent들의 수를 returng한다.

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    score를 return한다.

    
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
  # The code below extract some useful information from the state, like eremaining food and Pacman position
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
    def maximizer(state,depth):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      #agent index에 대한 legal action return
      if legalActions==None :
            return self.evaluationFunction(state)
      #만약 없으면 evaluation한다.
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      #action에 대해서 succesor를 generation한다.
      value = max([minimizer(succ_state,depth,1) for succ_state in succ_states])
      #그중에 가장 큰 값을 take한다.
      return value

    
    def minimizer(state,depth,agentIndex):
          if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
          legalActions = state.getLegalActions(agentIndex)
          if legalActions==None :
                return self.evaluationFunction(state)
          succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
          value = min([maximizer(succ_state,depth-1) if agentIndex+1==state.getNumAgents() else minimizer(succ_state,depth,agentIndex+1) for succ_state in succ_states])
          return value
    

    ## 여러개 opponent가 있다고 생각해보자,, 그럼 player는 가장 큰 값을 뽑고, opponent_index 는 next state 에서 opponent_index_1이 생성하는 것중 가장,, minimum한 것 고름
    legalActions = gameState.getLegalActions(0)
    value = float("-inf")
    ret_action = Directions.STOP
    for action in legalActions:
          temp_value = minimizer(gameState.generateSuccessor(0,action),self.depth,1)
          if temp_value>value:
                ret_action = action
                value = temp_value

    return ret_action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def maximizer(state,depth):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
            
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      value = max([minimizer(succ_state,depth,1) for succ_state in succ_states])
      return value

    
    def minimizer(state,depth,agentIndex):
          if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
          legalActions = state.getLegalActions(agentIndex)
          if legalActions==None :
                return self.evaluationFunction(state)
          succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
          value = min([maximizer(succ_state,depth-1) if agentIndex+1==state.getNumAgents() else minimizer(succ_state,depth,agentIndex+1) for succ_state in succ_states])
          return value
    
    return minimizer(gameState.generateSuccessor(0,action),self.depth,1) #이건 agent가 action을 취했을 때 value
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
    agent는 next state 최대화 하는 action찾고,
    opponent는 확률을 따른다.
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def maximizer(state,depth):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      value = max([expectation(succ_state,depth,1) for succ_state in succ_states])
      return value
  
    
    def expectation(state,depth,agentIndex):
          if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
          legalActions = state.getLegalActions(agentIndex)
          if legalActions==None :
                return self.evaluationFunction(state)
          succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
          value = sum([maximizer(succ_state,depth-1) if agentIndex== state.getNumAgents()-1 else expectation(succ_state,depth,agentIndex+1) for succ_state in succ_states ])
          #각 agent마다 next state들의 평균값을 return한다.
          return value/len(succ_states)
    
    legalActions = gameState.getLegalActions(0)
    value = float("-inf")
    ret_action = Directions.STOP
    for action in legalActions:
          temp_value = expectation(gameState.generateSuccessor(0,action),self.depth,1)
          if temp_value>value:
                ret_action = action
                value = temp_value

    return ret_action
    

    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def maximizer(state,depth):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      value = max([expectation(succ_state,depth,1) for succ_state in succ_states])
      return value

    
    def expectation(state,depth,agentIndex):
          if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
          legalActions = state.getLegalActions(agentIndex)
          if legalActions==None :
                return self.evaluationFunction(state)
          
          succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
          value = sum([maximizer(succ_state,depth-1) if agentIndex== state.getNumAgents()-1 else expectation(succ_state,depth,agentIndex+1) for succ_state in succ_states ])
          return value/len(succ_states)

    return expectation(gameState.generateSuccessor(0,action),self.depth,1)
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
    def maximizer (state,depth):
          if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
          legalActions = state.getLegalActions(0)
          succStates = [gameState.generateSuccessor(0,action) for action in legalActions]
          value = max([BiasedExpectation(succState,depth,1) for succState in succStates])
          return value
    
    def BiasedExpectation(state,depth,agentIndex):
          if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
          legalActions = state.getLegalActions(agentIndex)
          succStates = [gameState.generateSuccessor(agentIndex,action) for action in legalActions]
          prob = []
          for idx, succ_state in enumerate(succStates):
                if(legalActions[idx] == Directions.STOP):
                      prob.append(0.5 + 0.5 / len(legalActions))
                else : prob.append(0.5/len(legalActions))

          value = 0 
          for idx, one_prob in enumerate(prob):
                if agentIndex+1 == state.getNumAgents():
                      value += one_prob * maximizer(succStates[idx],depth-1)
                else :
                      value += one_prob * BiasedExpectation(succStates[idx],depth,agentIndex+1)

          return value
    
    legalActions = gameState.getLegalActions(0)
    value = float("-inf")
    ret_action = Directions.STOP
    for action in legalActions:
          temp_value = BiasedExpectation(gameState.generateSuccessor(0,action),self.depth,1)
          if(temp_value>value):
                ret_action = action
                value = temp_value
    return ret_action
    

    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def maximizer(state,depth):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [gameState.generateSuccessor(0,action) for action in legalActions]
      value = max([BiasedExpectation(succ_state,depth,1) for succ_state in succ_states])
      return value

    
    def BiasedExpectation(state,depth,agentIndex):
          if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
          legalActions = state.getLegalActions(agentIndex)
          if legalActions==None :
                return self.evaluationFunction(state)
          
          succ_states = [gameState.generateSuccessor(agentIndex,action) for action in legalActions]
          prob = []
          action_length = len(legalActions)
          for idx,succ_state in enumerate(succ_states):
                if(legalActions[idx] == Directions.STOP):
                      prob.append(0.5+0.5/action_length)
                else:
                      prob.append(0.5/action_length)
              
          value = 0
          for idx,one_prob in enumerate(prob):
                if agentIndex+1 == state.getNumAgents():
                      value += one_prob* maximizer(succ_states[idx],depth-1)
                else :
                      value += one_prob *BiasedExpectation(succ_states[idx],depth,agentIndex+1)
          return value
    
    return BiasedExpectation(gameState.generateSuccessor(0,action),self.depth,1)
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
    def maximizer(state,depth):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      value = max([RandomGhosts(succ_state,depth,1) for succ_state in succ_states])
      return value
    

    def RandomGhosts(state,depth,agentIndex):
      if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
      legalActions = state.getLegalActions(agentIndex)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
      if agentIndex%2==0:
        if agentIndex+1== state.getNumAgents():
          value =sum([maximizer(succ_state,depth -1) for succ_state in succ_states])/len(succ_states)
        else :
          value=sum([RandomGhosts(succ_state,depth, agentIndex+1) for succ_state in succ_states])/len(succ_states)
      else:
        if agentIndex+1== state.getNumAgents():
          value =min([maximizer(succ_state,depth -1) for succ_state in succ_states])
        else :
          value=min([RandomGhosts(succ_state,depth, agentIndex+1) for succ_state in succ_states])
      return value

    legalActions = gameState.getLegalActions(0)
    value = float("-inf")
    ret_action= Directions.STOP

    for action in legalActions:
      temp_value = RandomGhosts(gameState.generateSuccessor(0,action),self.depth,1)
      if temp_value>value :
        ret_action = action
        value = temp_value
    
    return ret_action

          
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def maximizer(state,depth):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      value = max([RandomGhosts(succ_state,depth,1) for succ_state in succ_states])
      return value
    
    def RandomGhosts(state,depth,agentIndex):
      if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
      legalActions = state.getLegalActions(agentIndex)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
      if agentIndex%2==0:
        if agentIndex+1== state.getNumAgents():
          value =sum([maximizer(succ_state,depth -1) for succ_state in succ_states])/len(succ_states)
        else :
          value=sum([RandomGhosts(succ_state,depth, agentIndex+1) for succ_state in succ_states])/len(succ_states)
      else:
        if agentIndex+1== state.getNumAgents():
          value =min([maximizer(succ_state,depth -1) for succ_state in succ_states])
        else :
          value=min([RandomGhosts(succ_state,depth, agentIndex+1) for succ_state in succ_states])
      return value
    
    return RandomGhosts(gameState.generateSuccessor(0,action),self.depth,1)
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
    def maximizer(state,depth,alpha,beta):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      
      next = float("-inf")
      for succ_state in succ_states:
            next = max(next, RandomGhosts(succ_state,depth,1,alpha,beta))
            if next>=beta : return next
            alpha = max(alpha,next)
      return next
    
    def RandomGhosts(state,depth,agentIndex,alpha,beta):
      if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
      legalActions = state.getLegalActions(agentIndex)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
      
      if agentIndex%2==0:
        if agentIndex+1== state.getNumAgents():
          value =sum([maximizer(succ_state,depth -1,alpha,beta) for succ_state in succ_states])/len(succ_states)
        else :
          value=sum([RandomGhosts(succ_state,depth, agentIndex+1,alpha,beta) for succ_state in succ_states])/len(succ_states)
        beta = min(beta, value)
        return value
      else:
        next = float("inf")
        for succ_state in succ_states:   
            if agentIndex+1== state.getNumAgents():
              next =min(next, maximizer(succ_state,depth -1,alpha,beta))
            else :
              next=min(next,RandomGhosts(succ_state,depth, agentIndex+1 ,alpha,beta))
            if (next<=alpha): return next
            beta = min( beta, next)
        return next
      
    legalActions = gameState.getLegalActions(0)
    value = float("-inf")
    ret_action= Directions.STOP

    for action in legalActions:
      temp_value = RandomGhosts(gameState.generateSuccessor(0,action),self.depth,1,float("-inf"),float("inf"))
      if temp_value>value :
        ret_action = action
        value = temp_value
    
    return ret_action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def maximizer(state,depth,alpha,beta):
      if(depth == 0 or state.isWin() or state.isLose()):
        return self.evaluationFunction(state)
      legalActions = state.getLegalActions(0)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(0,action) for action in legalActions]
      
      next = float("-inf")
      for succ_state in succ_states:
            next = max(next, RandomGhosts(succ_state,depth,1,alpha,beta))
            if next>=beta : return next
            alpha = max(alpha,next)
      return next
    
    def RandomGhosts(state,depth,agentIndex,alpha,beta):
      if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
      legalActions = state.getLegalActions(agentIndex)
      if legalActions==None :
            return self.evaluationFunction(state)
      succ_states = [state.generateSuccessor(agentIndex,action) for action in legalActions]
      
      if agentIndex%2==0:
        if agentIndex+1== state.getNumAgents():
          value =sum([maximizer(succ_state,depth -1,alpha,beta) for succ_state in succ_states])/len(succ_states)
        else :
          value=sum([RandomGhosts(succ_state,depth, agentIndex+1,alpha,beta) for succ_state in succ_states])/len(succ_states)
        beta = min(beta, value)
        return value
      else:
        next = float("inf")
        for succ_state in succ_states:   
            if agentIndex+1== state.getNumAgents():
              next =min(next, maximizer(succ_state,depth -1,alpha,beta))
            else :
              next=min(next,RandomGhosts(succ_state,depth, agentIndex+1 ,alpha,beta))
            if (next<=alpha): return next
            beta = min( beta, next)
        return next
      
    return RandomGhosts(gameState.generateSuccessor(0,action),self.depth,1, float("-inf"),float("inf"))
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  curPacmanPos = currentGameState.getPacmanPosition()
  features = []
  features.append(currentGameState.getScore())
  weights = [2]
  closeCapsuleList = [manhattanDistance(curPacmanPos,capsulePosition) for capsulePosition in currentGameState.getCapsules()]
  mostFarCapsuleInverse = 1/max(closeCapsuleList) if len(closeCapsuleList)!= 0 else 0
  features.append(mostFarCapsuleInverse)
  #capsule 가까이 있을수록 점수 높 -> 가장 멀리 있는 캡슐이 가장 가깝게 
  closeFoodList = [manhattanDistance(curPacmanPos,foodPosition) for foodPosition in currentGameState.getFood().asList()]
  closestFoodDistInverse = 1/min(closeFoodList) if len(closeFoodList)!= 0 else 0
  features.append(closestFoodDistInverse)
  #음식 가까이에 있을수록 점수 높 -> 가장 가까이 있는 음식이 가장 가깝게
  ghostList = currentGameState.getGhostStates()
  noEffectGhostList = [ghost.scaredTimer/manhattanDistance(curPacmanPos,ghost.getPosition()) for ghost in ghostList if ghost.scaredTimer != 0]
  #이건 가까워야함
  usualGhostList = [manhattanDistance(curPacmanPos,ghost.getPosition()) for ghost in ghostList if ghost.scaredTimer == 0]
  #이건 멀어야함
  noEffectGhostDist = sum(noEffectGhostList)/len(noEffectGhostList) if len(noEffectGhostList)!=0 else 0
  usualGhostDist = min(usualGhostList) if len(usualGhostList)!=0 else 0
  features.append(noEffectGhostDist)
  features.append(usualGhostDist)
  weights.append(0) if len(usualGhostList)==0 else 10# 다 freeze 일시 capsule위치 의미 x
  weights.append(0) if currentGameState.getNumFood()==0 else weights.append(10/currentGameState.getNumFood())
  weights.append(16)
  weights.append(1)
  return sum([w * f for w, f in zip(weights, features)])






  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectimaxAgent' # remove this line before writing code
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
