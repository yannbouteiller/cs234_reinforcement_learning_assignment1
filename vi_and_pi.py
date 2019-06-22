# Assignment 1 Stanford CS234 - Reinforcement Learning - Winter 2019
# Team members: Yann BOUTEILLER, Amine BELLAHSEN

# MDP Policy Iteration and Value Iteration
# Setting: Finite discrete state-action space, known model

# ------------------------------------------------------------------------------

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""
	Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value(s) - prev_value(s)| < tol
	Returns
	-------
	value: np.ndarray[nS]
		The value function of the given policy, where value[s] is
		the value of state s
	"""

	value = np.zeros(nS) # value function initialized at 0 for all states
	while True: # until convergence
		new_value = np.zeros(nS)
		for state in range(nS): # for all states
			p = P[state][policy[state]] # {(probability, nextstate, reward, terminal),...}[s,pi(s)]
			reward = sum([p_i[0]*p_i[2] for p_i in p]) # reward weighted by probabilities of pi(s) outcomes
			new_value[state] = reward + gamma*(sum([p_i[0]*value[p_i[1]] for p_i in p])) # r + gamma sum (p(s')V(s'))
		if np.max(np.abs(new_value - value)) < tol: # if value has converged
			value = new_value
			break
		value = new_value
	return value


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""
	Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

	new_policy = np.zeros(nS, dtype='int')
	for state in range(nS): # for each state
		best_Q_value = -float("inf") # we seek the action that gives the best Q value from this state
		for action in range(nA): # for each action
			p = P[state][action] # {(probability, nextstate, reward, terminal),...}[state,action]
			reward = sum([p_i[0]*p_i[2] for p_i in p]) # expected reward from state,action
			Q_value = reward + gamma*(sum([p_i[0]*value_from_policy[p_i[1]] for p_i in p])) # expected reward + gamma * expected next value
			if Q_value > best_Q_value: # if this is the best action from this state so far
				best_Q_value = Q_value
				new_policy[state] = action # update policy
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""
	Runs policy iteration, using policy_evaluation() and policy_improvement().

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	policy = np.zeros(nS, dtype=int)
	while True: # while policy changes for at least one state
		value = policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3)
		new_policy = policy_improvement(P, nS, nA, value, policy, gamma=0.9)
		if np.linalg.norm((policy - new_policy), ord=1) == 0:
			break
		policy = new_policy
	return value, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value(s) - prev_value(s)| < tol
	Returns:
	----------
	value: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value = np.zeros(nS) # value function initialized
	policy = np.zeros(nS, dtype=int) # policy initialized
	while True: # until convergence or finite horizon overflow
		new_value = np.zeros(nS)
		for state in range(nS): # for each state
			best_Q_value = -float("inf") # we are looking for the best action in term of Q value
			for action in range(nA): # for each action
				p = P[state][action] # {(probability, nextstate, reward, terminal),...}[state,action]
				reward = sum([i[0]*i[2] for i in p]) # expected reward for this state-action
				Q_value = reward + gamma*(sum([i[0]*value[i[1]] for i in p])) # expected reward + gamma * expected value for this state-action
				if Q_value > best_Q_value:
					new_value[state] = Q_value # max_a Q for this state
					policy[state] = action # argmax_a Q for this state
					best_Q_value = Q_value
		if np.max(np.abs(new_value - value)) < tol: # convergence
			value = new_value
			break
		value = new_value
	return value, policy


def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	print("DETERMINISTIC MODEL:\n")
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)
	print('  Optimal Value Function: %r' % V_pi)
	print('  Optimal Policy:         %r' % p_pi)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)
	print('  Optimal Value Function: %r' % V_vi)
	print('  Optimal Policy:         %r' % p_vi)

	print("\n---------------------------------------\n")
	print("STOCHASTIC MODEL:\n")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)
	print('  Optimal Value Function: %r' % V_pi)
	print('  Optimal Policy:         %r' % p_pi)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)
	print('  Optimal Value Function: %r' % V_vi)
	print('  Optimal Policy:         %r' % p_vi)


# ------------------------------------------------------------------------------

# Found policies:

# Deterministic model:

#[1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0]
#[1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0]

# Stochastic model:

#[0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
#[0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
