# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


INF = 999999999


def mean(x: List) -> float:
    return sum(x) / len(x)


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if oldPos == newPos:
            return -INF

        currentFood = currentGameState.getFood()
        newGhostPos = [g.configuration.pos for g in newGhostStates]
        newScaredGhostPos = [g.configuration.pos for g in newGhostStates if g.scaredTimer > 0]
        newRegularGhostPos = [g.configuration.pos for g in newGhostStates if g.scaredTimer == 0]

        food = [(i, j) for i in range(len(currentFood.data)) for j in range(len(currentFood.data[i])) if currentFood.data[i][j]]
        successor_food = [(i, j) for i in range(len(newFood.data)) for j in range(len(newFood.data[i])) if newFood.data[i][j]]

        if len(successor_food) == 0:
            return INF

        foods_eaten = len(successor_food) - len(food)

        aghost2food_dist = [manhattanDistance(g, f) if newScaredTimes[i] == 0 else 9999999 for f in successor_food for i, g in enumerate(newGhostPos)]

        ghost2food_dist = [manhattanDistance(g, f) for f in successor_food for g in newRegularGhostPos]
        scaredghost2food_dist = [manhattanDistance(g, f) for f in successor_food for g in newScaredGhostPos]

        ghost2pac_dist = [manhattanDistance(g, newPos) for g in newRegularGhostPos]
        scaredghost2pac_dist = [manhattanDistance(g, newPos) for g in newScaredGhostPos]

        pac2food_dist = [manhattanDistance(f, newPos) for f in successor_food]

        reachable_foods = [pac2food_dist[i] for i, f in enumerate(successor_food) if aghost2food_dist[i] > pac2food_dist[i]]

        if newPos in food and (len(ghost2pac_dist) == 0 or min(ghost2pac_dist) > 4):
            return INF

        if len(scaredghost2pac_dist) > 0 and min(scaredghost2pac_dist) < 1:
            return INF

        score = currentGameState.getScore()
        score += 0 if len(ghost2pac_dist) == 0 else max(ghost2pac_dist)
        score += 0 if len(scaredghost2pac_dist) == 0 else min(scaredghost2pac_dist)
        score += 0 if len(ghost2food_dist) == 0 else max(ghost2food_dist)
        score += 0 if len(scaredghost2food_dist) == 0 else min(scaredghost2food_dist)
        score += foods_eaten
        score += 0 if len(reachable_foods) == 0 else 25 / min(reachable_foods)

        return score

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState.
        """
        n_ghosts = gameState.getNumAgents() - 1
        return self.pacman_move(gameState, n_ghosts=n_ghosts, return_action=True)

    def pacman_move(self, state, n_ghosts, depth=0, return_action=False):
        if depth == self.depth:
            return self.evaluationFunction(state)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalMoves = state.getLegalActions(0)
        values = {move: self.ghost_move(state.generateSuccessor(0, move), n_ghosts=n_ghosts, depth=depth) for move in legalMoves}
        max_idx = max(values, key=values.get)
        if return_action:
            return max_idx
        return values[max_idx]

    def ghost_move(self, state, agent=0, n_ghosts=1, depth=0):
        ghost = agent + 1
        if ghost > n_ghosts:
            return self.pacman_move(state, depth=depth + 1, n_ghosts=n_ghosts)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalMoves = state.getLegalActions(ghost)
        values = {move: self.ghost_move(state.generateSuccessor(ghost, move), agent=agent + 1, depth=depth, n_ghosts=n_ghosts) for move in legalMoves}
        min_idx = min(values, key=values.get)
        return values[min_idx]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        n_ghosts = gameState.getNumAgents() - 1
        return self.pacman_move(gameState, n_ghosts=n_ghosts, return_action=True)

    def pacman_move(self, state, n_ghosts, alpha=-INF, beta=INF, depth=0, return_action=False):
        if depth == self.depth:
            return self.evaluationFunction(state)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        mx, mxm = -INF, None
        legalMoves = state.getLegalActions(0)
        for move in legalMoves:
            v = self.ghost_move(state.generateSuccessor(0, move), alpha=alpha, beta=beta, n_ghosts=n_ghosts, depth=depth)
            alpha = max(alpha, v)
            mx, mxm = (v, move) if v > mx else (mx, mxm)
            if v > beta:
                return v
        if return_action:
            return mxm
        return mx

    def ghost_move(self, state, alpha, beta, agent=0, n_ghosts=1, depth=0):
        ghost = agent + 1
        if ghost > n_ghosts:
            return self.pacman_move(state, alpha=alpha, beta=beta, depth=depth + 1, n_ghosts=n_ghosts)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalMoves = state.getLegalActions(ghost)
        mn = INF
        for move in legalMoves:
            v = self.ghost_move(state.generateSuccessor(ghost, move), alpha=alpha, beta=beta, agent=agent + 1, depth=depth, n_ghosts=n_ghosts)
            beta = min(beta, v)
            mn = min(mn, v)
            if v < alpha:
                return v
        return mn

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        n_ghosts = gameState.getNumAgents() - 1
        return self.pacman_move(gameState, n_ghosts=n_ghosts, return_action=True)

    def pacman_move(self, state, n_ghosts, depth=0, return_action=False):
        if depth == self.depth:
            return self.evaluationFunction(state)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalMoves = state.getLegalActions(0)
        values = {move: self.ghost_move(state.generateSuccessor(0, move), n_ghosts=n_ghosts, depth=depth) for move in
                  legalMoves}
        max_idx = max(values, key=values.get)
        if return_action:
            return max_idx
        return values[max_idx]

    def ghost_move(self, state, agent=0, n_ghosts=1, depth=0):
        ghost = agent + 1
        if ghost > n_ghosts:
            return self.pacman_move(state, depth=depth + 1, n_ghosts=n_ghosts)
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        legalMoves = state.getLegalActions(ghost)
        values = [self.ghost_move(state.generateSuccessor(ghost, move), agent=agent + 1, depth=depth, n_ghosts=n_ghosts) for move in legalMoves]
        return mean(values)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Basically a combination of the food left on the map, with respect to map size
    while encouraging following scared ghosts and avoiding ones that are not scared.
    """
    from numpy import array as arr
    pac = currentGameState.getPacmanPosition()
    food_mat = currentGameState.getFood().data
    wall_mat = currentGameState.getWalls().data
    num_cells = arr(not wall_mat).sum()
    capsule_mat = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()

    regular_ghosts = [g.configuration.pos for g in ghost_states if g.scaredTimer == 0]
    scared_ghosts = [g.configuration.pos for g in ghost_states if g.scaredTimer > 0]
    food = [(i, j) for i in range(len(food_mat)) for j in range(len(food_mat[i])) if food_mat[i][j]]
    capsules = [(i, j) for i in range(len(capsule_mat)) for j in range(len(capsule_mat[i])) if capsule_mat[i][j]]

    if len(food) == 0:
        # Win state!
        return INF

    pac2food = arr([manhattanDistance(f, pac) for f in food])

    pac2ghost = arr([manhattanDistance(g, pac) for g in regular_ghosts])
    nearest_ghost = -1 if len(pac2ghost) <= 0 else pac2ghost.argmin()

    pac2scaredghost = arr([manhattanDistance(g, pac) for g in scared_ghosts])
    nearest_scaredghost = -1 if len(pac2scaredghost) <= 0 else pac2scaredghost.argmin()

    scared_timers = arr([g.scaredTimer for g in ghost_states if g.scaredTimer > 0])

    # Start with the actual score
    score = currentGameState.getScore()

    # Deduct points for sum of distances between pacman and food points
    # If the food on one side of the map is still not taken, pacman better make a run for it.
    score -= sum(pac2food)
    # Add points for the number of cells that are empty (food taken) -- maximum when there's no food left
    score += num_cells / len(food)

    # If there is a scared ghost, add points for their timer
    # (max steps that can be taken to hit the scared ghost)
    # and deduct points for the sum of their distances.
    if nearest_scaredghost >= 0:
        score += sum(scared_timers) - pac2scaredghost.sum()

    # If there is a normal ghost, add points for avoid the nearest one
    # and add points for capsules (eat a capsule and kill the ghost)
    if nearest_ghost >= 0:
        score += pac2ghost[nearest_ghost]
        score += arr(not capsule_mat).sum() - len(capsules)

    return score


# Abbreviation
better = betterEvaluationFunction

# ghosts = [g.configuration.pos for g in ghost_states]
# nearest_food = pac2food.argmin()
# food_dist_avg = 1 / pac2food.sum()

# pac2capsule = arr([manhattanDistance(c, pac) for c in capsules])
# nearest_capsule = -1 if len(pac2capsule) <= 0 else pac2capsule.argmin()

# ghost2food = arr([[manhattanDistance(f, g) for f in food] for g in ghosts])
# ghost2capsule = arr([[manhattanDistance(c, g) for c in capsules] for g in ghosts])

# p2f = arr([[manhattanDistance(f1, f2) for f1 in food] for f2 in food])
# score += len(food) - np.min(p2f, axis=0).sum() - min(pac2food)
