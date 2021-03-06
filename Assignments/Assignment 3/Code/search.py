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
import math

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
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s]

def mySearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    startState=problem.getStartState()
    childStates=problem.getSuccessors(startState)
    leftChild=childStates[0]

  
    print(startState)
    print(childStates)
    print(leftChild)
    return [s]
def isExistInPQ(self, item):
    for p in self.heap:
        if(item == p):
            return True
    return False

def isExistInQ(self, item):
    for p in self.list:
        if(item == p):
            return True
    return False
    
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    currentState = problem.getStartState()
    F = util.Stack()
    E = []
    a = []
    paths= {}
    F.push(currentState)
    paths[currentState] = []
    while not F.isEmpty():
        currentState = F.pop()
        path = paths[currentState]
        E.append(currentState)
        if problem.isGoalState(currentState):
            return path
        else:
            successors = problem.getSuccessors(currentState)
            #successors.reverse()
            for child in  successors:
                if (child[0] not in E) and (not isExistInQ(F, child[0])):
                    F.push(child[0])
                    a.append(child[1])
                    paths[child[0]] = path + a
                    a = []
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    currentState = problem.getStartState()
    F = util.Queue()
    E = []
    a = []
    paths= {}
    F.push(currentState)
    paths[currentState] = []
    while not F.isEmpty():
        currentState = F.pop()
        path = paths[currentState]
        E.append(currentState)
        if problem.isGoalState(currentState):
            return path
        else:
            for child in  problem.getSuccessors(currentState):
                if (child[0] not in E) and (not isExistInQ(F, child[0])):
                    F.push(child[0])
                    a.append(child[1])
                    paths[child[0]] = path + a
                    a = []
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    currentState = problem.getStartState()
    F = util.PriorityQueue()
    E = []
    a = []
    paths= {}
    F.push(currentState, 0)
    paths[currentState] = []
    while not F.isEmpty():
        currentState = F.pop()
        path = paths[currentState]
        E.append(currentState)
        if problem.isGoalState(currentState):
            return path
        else:
            for child in  problem.getSuccessors(currentState):
                if (child[0] not in E) and (not isExistInPQ(F, child[0])):
                    F.push(child[0], child[2])
                    a.append(child[1])
                    paths[child[0]] = path + a
                    a = []
                elif isExistInPQ(F, child[0]):
                    F.update(child[0], child[2])
                    
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def manHattanHeuristic(state, problem=None):
    goalState = problem.goal
    i = abs(state[0][0] - goalState[0]) + abs(state[0][1] - goalState[1])
    p = state[2] + i
    return p;
def EuclideanHeuristic(state, problem=None):
    goalState = problem.goal
    i = math.sqrt((state[0][0] - goalState[0])**2 + (state[0][1] - goalState[1])**2)
    p = state[2] + i
    return p;
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    currentState = problem.getStartState()
    F = util.PriorityQueue()
    E = []
    a = []
    paths= {}
    goalState = problem.goal
    #i = abs(currentState[0] - goalState[0]) + abs(currentState[1] - goalState[1])
    i = math.sqrt((currentState[0] - goalState[0])**2 + (currentState[1] - goalState[1])**2)
    F.push(currentState,  i)
    paths[currentState] = []
    while not F.isEmpty():
        currentState = F.pop()
        path = paths[currentState]
        E.append(currentState)
        if problem.isGoalState(currentState):
            return path
        else:
            for child in  problem.getSuccessors(currentState):
                if (child[0] not in E) and (not isExistInPQ(F, child[0])):
                    
                    # using absolute distance heuristic
                    #p = manHattanHeuristic(child, problem);
                    
                    # using Euclidean Heuristic
                    p = EuclideanHeuristic(child, problem)
                    F.push(child[0], p)
                    a.append(child[1])
                    paths[child[0]] = path + a
                    a = []
                elif isExistInPQ(F, child[0]):
                    F.update(child[0], child[2])
                    
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
