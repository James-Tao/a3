import sys
import time
from constants import *
from environment import *
from state import State
import math

"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""


class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment):
        self.environment = environment
        self.gamma = 0.99999
        self.alpha = 0.1
        self.epsilon = 0.4
        self.EXP_BIAS = 1.4
        self.MAX_STEPS = 50
        self.MAX_EPISODES = 3500
        self.q_values = {}
        self.actions = [FORWARD, REVERSE, SPIN_LEFT, SPIN_RIGHT]
        self.num_of_states = {}
        self.num_of_actions = {}
        self.reach_state = []
        self.persistent_state = environment.get_init_state()
        self.visited_states = set()
        self.q_rewards = []

        pass

    def reachable_stat(self,state):
        reachable = []
        for action in ROBOT_ACTIONS:
            reward, next_state = self.environment.perform_action(state, action)
            reachable.append(next_state)
        return reachable
    # === Q-learning ===================================================================================================

    def q_learn_train(self):

        """
        Train this RL agent via Q-Learning.
        """
        #
        # TODO: Implement your Q-learning training loop here.
        #
        t0 = time.time()
        self.reach_state = self.reachable_stat(self.environment.get_init_state())
        epis_state = random.choice(self.reach_state)
        flag = self.environment.is_solved(epis_state)
        while self.environment.get_total_reward() > self.environment.training_reward_tgt and \
                time.time() - t0 < self.environment.training_time_tgt - 1:
            t = 0
            while t < self.MAX_STEPS:
                t += 1
                epis_action = self.q_learn_select_action(epis_state)
                if not flag:
                    reward, next_state = self.environment.perform_action(epis_state, epis_action)
                    best_q = -math.inf
                    best_a = None
                    for a in self.actions:
                        q = self.q_values.get((next_state, a))
                        if q is not None and q > best_q:
                            best_q = q
                            best_a = a
                    if best_a is None:
                        best_q = 0
                    target = reward + (self.gamma * best_q)

                    if (epis_state, epis_action) in self.q_values:
                        old_q = self.q_values[(epis_state, epis_action)]
                    else:
                        old_q = 0
                    self.q_values[(epis_state, epis_action)] = old_q + (self.alpha * (target - old_q))
                    if epis_state in self.num_of_states:
                        self.num_of_states[epis_state] += 1
                    else:
                        self.num_of_states[epis_state] = 1
                    if (epis_state, epis_action) in self.num_of_actions:
                        self.num_of_actions[(epis_state, epis_action)] += 1
                    else:
                        self.num_of_actions[(epis_state, epis_action)] = 1

                    if flag:
                        self.environment.is_solved = True
                        continue
                    self.visited_states.add(epis_state)
                    epis_state = next_state
                else:
                    flag = False

                    epis_state = random.choice(self.reach_state)

                    break
        pass

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  Q-learning Q-values) here.
        #
        unvisited = []
        unvisited_exists = False
        best_u = -math.inf
        best_a = None
        for a in self.actions:
            if (state, a) not in self.q_values:
                unvisited.append(a)
                unvisited_exists = True
            elif not unvisited_exists:
                u = self.q_values[(state, a)] + (self.EXP_BIAS * math.sqrt(math.log(self.num_of_states[state])
                                                                         / self.num_of_actions[(state, a)]))
                if u > best_u:
                    best_u = u
                    best_a = a
        if unvisited_exists:
            return random.choice(unvisited)
        else:
            return best_a
        pass

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # TODO: Implement your SARSA training loop here.
        #
        t0 = time.time()
        self.reach_state = self.reachable_stat(self.environment.get_init_state())
        epi_state = random.choice(self.reach_state)
        flag = self.environment.is_solved(epi_state)
        while self.environment.get_total_reward() > self.environment.training_reward_tgt and \
                time.time() - t0 < self.environment.training_time_tgt - 1:
            t = 0
            epi_action = self.sarsa_select_action(epi_state)
            while t < self.MAX_STEPS:
                t += 1
                if not flag:
                    reward, next_state = self.environment.perform_action(epi_state, epi_action)
                    next_action = self.sarsa_select_action(next_state)
                    q_next = self.q_values.get((next_state, next_action))
                    if q_next is None:
                        q_next = 0
                    target = reward + (self.gamma * q_next)
                    if (epi_state, epi_action) in self.q_values:
                        old_q = self.q_values[(epi_state, epi_action)]
                    else:
                        old_q = 0
                    self.q_values[(epi_state, epi_action)] = old_q + (self.alpha * (target - old_q))

                    if epi_state in self.num_of_states:
                        self.num_of_states[epi_state] += 1
                    else:
                        self.num_of_states[epi_state] = 1
                    if (epi_state, epi_action) in self.num_of_actions:
                        self.num_of_actions[(epi_state, epi_action)] += 1
                    else:
                        self.num_of_actions[(epi_state, epi_action)] = 1

                    if self.environment.is_solved(next_state):
                        self.q_values[(next_state, next_action)] = 0
                        flag = True
                    self.visited_states.add(epi_state)
                    epi_state = next_state
                    epi_action = next_action
                else:
                    flag = False
                    if epi_state in self.num_of_states:
                        self.num_of_states[epi_state] += 1
                    else:
                        self.num_of_states[epi_state] = 1
                    if (epi_state, epi_action) in self.num_of_actions:
                        self.num_of_actions[(epi_state, epi_action)] += 1
                    else:
                        self.num_of_actions[(epi_state, epi_action)] = 1
                    epi_state = random.choice(self.reach_state)
                    epi_action = random.choice(self.actions)
        pass

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.
        #
        unvisited = []
        unvisited_exists = False
        best_u = -math.inf
        best_a = None
        for a in self.actions:
            if (state, a) not in self.q_values:
                unvisited.append(a)
                unvisited_exists = True
            elif not unvisited_exists:
                u = self.q_values[(state, a)] + (self.EXP_BIAS * math.sqrt(math.log(self.num_of_states[state])
                                                                         / self.num_of_actions[(state, a)]))
                if u > best_u:
                    best_u = u
                    best_a = a
        if unvisited_exists:
            return random.choice(unvisited)
        else:
            return best_a
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    #
    #

