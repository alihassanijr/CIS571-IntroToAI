# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            # Temporarily store values at t
            # Avoiding mixup between t and t + 1 when computing
            values = self.values.copy()
            for s in states:
                if self.mdp.isTerminal(s):
                    continue
                actions = self.mdp.getPossibleActions(s)
                if len(actions) == 0:
                    continue
                values[s] = max([self.getQValue(s, action) for action in actions])

            for s in states:
                self.values[s] = values[s]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        sp = self.mdp.getTransitionStatesAndProbs(state, action)
        sop = 0
        for x in sp:
            s_prime, prob = x
            sop += prob * (self.mdp.getReward(state, action, s_prime) + self.discount * self.values[s_prime])
        return sop

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        a = util.Counter()
        for action in actions:
            a[action] = self.getQValue(state, action)
        return a.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            j = i % len(states)
            s = states[j]
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            if len(actions) == 0:
                continue
            self.values[s] = max([self.getQValue(s, action) for action in actions])


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computePredecessors(self, states):
        predecessors = {s: [] for s in states}
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                predictions = self.mdp.getTransitionStatesAndProbs(state, action)
                for (pred_state, prob) in predictions:
                    None if prob <= 0 else predecessors[pred_state].append(state)
        return predecessors

    def runValueIteration(self):
        states = self.mdp.getStates()
        predecessors = self.computePredecessors(states)
        queue = util.PriorityQueue()
        for s in states:
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            if len(actions) == 0:
                continue
            diff = abs(self.values[s] - max([self.getQValue(s, action) for action in actions]))
            queue.push(s, -diff)
        for i in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            actions = self.mdp.getPossibleActions(state)
            if len(actions) == 0:
                continue
            self.values[state] = max([self.getQValue(state, action) for action in actions])
            for pred in predecessors[state]:
                pred_actions = self.mdp.getPossibleActions(pred)
                diff = abs(self.values[pred] - max([self.getQValue(pred, action) for action in pred_actions]))
                if diff > self.theta:
                    queue.update(pred, -diff)
