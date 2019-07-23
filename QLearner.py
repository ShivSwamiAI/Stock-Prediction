"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.T = np.zeros((self.num_states, self.num_actions, self.num_states)) + 0.001
        self.R = np.zeros((self.num_states, self.num_actions))
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.s = 0
        self.a = 0

    def author(self):
        return 'rsadiq3' # My Georgia Tech username.

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if np.random.random_sample() < self.rar:
            action = np.random.choice(range(self.num_actions), size=1)[0]
            self.rar = self.rar * self.radr
        else:
            action = np.argmax(self.Q[s])
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # set previous state and action as local variable
        state = self.s
        action = self.a
        print "next_action", len(self.Q)
        next_action = np.argmax(self.Q[s_prime])

        # Update Q-Value
        self.update_Q(state, action, s_prime, next_action, r)

        if np.random.random_sample() < self.rar:
            action = np.random.choice(range(self.num_actions), size=1)[0]
            self.rar = self.rar * self.radr
        else:
            action = np.argmax(self.Q[s_prime])

        self.a = action
        self.s = s_prime
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

    def update_Q(self, current_state, current_action, next_state, next_action, reward):
        self.Q[current_state][current_action] = ((1 - self.alpha)* self.Q[current_state][current_action]) + (self.alpha * (reward + self.gamma * (self.Q[next_state][next_action])))

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
