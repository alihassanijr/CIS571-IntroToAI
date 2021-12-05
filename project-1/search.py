# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueue
from game import Directions


class State:
    def __init__(self, position, move, parent=None):
        self.position = position
        self.move = move
        self.parent = parent

    def getParent(self):
        return None if self.parent is None else self.parent

    def isStartingPoint(self):
        return self.getParent is None


def getMoves(target_state):
    moves = []
    state = target_state
    while state is not None and not state.isStartingPoint():
        if state.move is None:
            break
        moves.append(state.move)
        state = state.getParent()
    return moves[::-1]


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



def mediumMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    n = Directions.NORTH
    s = Directions.SOUTH
    e = Directions.EAST
    w = Directions.WEST
    return [s, s, w, w, w, w, s, s, e, e, e, e, s, s, w, w, w, w, s,
    s, e, e, e, e, s, s, w, w, w, w, s, s, e, e, e, e, s, s, s, w, w,
    w, w, w, w, w, n, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w,
    w, s, w, w, w, w, w, w, w, w, w
    ]


def generalSearch(problem, queue, weighted=False, heuristic=None):
    visited_states = []
    state = State(problem.getStartState(), None)
    args = (state, 0) if weighted else (state,)
    queue.push(*args)
    while not queue.isEmpty() and not problem.isGoalState(state.position):
        state = queue.pop()
        if problem.isGoalState(state.position):
            break
        if state.position not in visited_states:
            visited_states.append(state.position)
            for position, move, cost in problem.getSuccessors(state.position):
                if position not in visited_states:
                    new_state = State(position, move, state)
                    if weighted:
                        cost = problem.getCostOfActions(getMoves(new_state))
                        cost = cost if heuristic is None else cost + heuristic(position, problem)
                        queue.push(new_state, cost)
                    else:
                        queue.push(new_state)
    return getMoves(state)


def depthFirstSearch(problem):
    """
    Searches the deepest nodes in the search tree first.
    """
    return generalSearch(problem, queue=Stack(), weighted=False)


def breadthFirstSearch(problem):
    """Searches the shallowest nodes in the search tree first."""
    return generalSearch(problem, queue=Queue(), weighted=False)

def uniformCostSearch(problem):
    """Searches the node of least total cost first."""
    return generalSearch(problem, queue=PriorityQueue(), weighted=True)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Searches the node that has the lowest combined cost and heuristic first."""
    return generalSearch(problem, queue=PriorityQueue(), weighted=True, heuristic=heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
