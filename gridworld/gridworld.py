"""
gridworld

4x4 grid, top left and bottom right are terminal states;
all else have a reward of -1. 4 actions: up, down, left, right.
transitions are deterministic, running into a wall keeps you in the same state.
"""
from enum import Enum
import numpy as np

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Agent:

    def __init__(self, i = 3, j = 1):
        self.i = i
        self.j = j
        self.episode = 0
        self.rewards = []
        self.policy = {(i,j): {Action.UP: .25, 
                               Action.DOWN: .25, 
                               Action.LEFT: .25, 
                               Action.RIGHT: .25} 
                               for i in range(4) 
                               for j in range(4) 
                               if not is_terminal(i, j)
                               }
       
        self.v = {}

    def simulate(self, N, x = 3, y = 2):
        for i in range(N):
            self.run_episode(x, y)
            if i % 100 == 0:
                print(f"Episode {i} mean reward so far: {np.mean(self.rewards)}")
        return np.mean(self.rewards)

    def run_episode(self, x, y):
        self.i = x
        self.j = y
        self.episode += 1
        this_reward = 0
        while not is_terminal(self.i, self.j):
            a = self.act()
            (self.i, self.j) = transition(self.i, self.j, a)
            this_reward += reward(a)
        self.rewards.append(this_reward)
            
    def act(self):
        p = np.random.rand()
        if p < self.policy[(self.i, self.j)][Action.UP]:
            return Action.UP
        else:
            p -= self.policy[(self.i, self.j)][Action.UP]
        if p < self.policy[(self.i, self.j)][Action.DOWN]:
            return Action.DOWN
        else:
            p -= self.policy[(self.i, self.j)][Action.DOWN]
        if p < self.policy[(self.i, self.j)][Action.LEFT]:
            return Action.LEFT
        else:
            return Action.RIGHT
        
    def iterative_policy_evaluation(self, gamma, theta):
        # arbitrary initial v values except for terminal states: 0
        for i in range(4):
            for j in range(4):
                if is_terminal(i, j):
                    self.v[(i,j)] = 0
                else:
                    self.v[(i,j)] = 0.5
        
        while True:
            delta = 0
            for i in range(4):
                for j in range(4):
                    if is_terminal(i, j):
                        continue
                    v = self.policy_evaluation(i, j, gamma)
                    delta = max(delta, abs(v - self.v[(i,j)]))
                    self.v[(i,j)] = v
            if delta < theta:
                break

    def policy_evaluation(self, i, j, gamma):
        v = 0
        for a in Action:
            (i_prime, j_prime) = transition(i, j, a)
            # deterministic transitions, so no need to sum over the s', r pairs
            v += self.policy[(i, j)][a] * (reward(a) + gamma * self.v[(i_prime, j_prime)])
        return v
    
    def policy_argmax(self, s):
        """
        get set of actions that have the highest value
        """
        return {a for a in Action if self.policy[s][a] == max(self.policy[s].values())}
    
    def policy_iteration(self, gamma, theta):
        while True:
            self.iterative_policy_evaluation(gamma, theta)
            stable = True
            for i in range(4):
                for j in range(4):
                    if is_terminal(i, j):
                        continue
                    
                    old_action = self.policy_argmax((i, j))
                    new_action = self.policy_improvement(i, j, gamma)
                    if old_action != new_action:
                        stable = False
            if stable:
                break
    
    def policy_improvement(self, i, j, gamma):
        best_action_set = set()
        best_value = float('-inf')
        for a in Action:
            (i_prime, j_prime) = transition(i, j, a)
            v = reward(a) + gamma * self.v[(i_prime, j_prime)]
            if v > best_value:
                best_action_set = {a}
                best_value = v
            elif v == best_value:
                best_action_set.add(a)
        for a in Action:
            if a in best_action_set:
                self.policy[(i, j)][a] = 1 / len(best_action_set)
            else:
                self.policy[(i, j)][a] = 0
        return best_action_set
    
    def value_iteration(self, gamma, theta, deterministic = True):
        while True:
            delta = 0
            for i in range(4):
                for j in range(4):
                    if is_terminal(i, j):
                        continue
                    v = self.v[(i,j)]
                    self.v[(i,j)] = self.policy_evaluation(i, j, gamma)
                    delta = max(delta, abs(v - self.v[(i,j)]))
            if delta < theta:
                break
        
        # create a stochastic/deterministic policy:
        for i in range(4):
            for j in range(4):
                if is_terminal(i, j):
                    continue
                if deterministic:
                    best_action = max(self.policy[(i, j)], key = self.policy[(i, j)].get)
                    for a in Action:
                        if a == best_action:
                            self.policy[(i, j)][a] = 1
                        else:
                            self.policy[(i, j)][a] = 0
                else:
                    self.policy_improvement(i, j, gamma)
        


def transition(i, j, a):
    if a == Action.UP:
        return move_up(i, j)
    elif a == Action.DOWN:
        return move_down(i, j)
    elif a == Action.LEFT:
        return move_left(i, j)
    elif a == Action.RIGHT:
        return move_right(i, j)    

def move_left(i, j):
    return (i, max(j - 1, 0))

def move_right(i, j):
    return (i, min(j + 1, 3))

def move_up(i, j):
    return (max(i - 1, 0), j)

def move_down(i, j):
    return (min(i + 1, 3), j)

def reward(a):
    return -1

def is_terminal(i, j):
    return (i, j) == (0, 0) or (i, j) == (3, 3)


agent = Agent()
# simulation
# mean = agent.simulate(1_000_000)
# print(mean)

# policy evaluation
print('\t(equiprobable policy) policy evaluation')
agent.iterative_policy_evaluation(1 - 10**-12, 10**-12)
for i in range(4):
    for j in range(4):
        if not is_terminal(i, j):
            print(f'({i},{j}): {round(agent.v[(i,j)], 6)}')

print('\n\n\t(stochastic policy if ties) policy iteration')

# policy iteration
agent.policy_iteration(1 - 10**-12, 10**-12)
for i in range(4):
    print('\n')
    for j in range(4):
        if not is_terminal(i, j):
            actions = [k.name for k, v in agent.policy[(i, j)].items() if v > 0.0]
            actions_str = ', '.join(actions) if actions else '-'
            print(f'({i},{j}): {actions_str:<20}', end=' ')
        else:
            print(f'({i},{j}): {"n/a":<20}', end=' ')

print('\n\n\t(deterministic policy) value iteration')

# value iteration
agent.value_iteration(1 - 10**-12, 10**-12, deterministic = True)
for i in range(4):
    print('\n')
    for j in range(4):
        if not is_terminal(i, j):
            actions = [k.name for k, v in agent.policy[(i, j)].items() if v > 0.0]
            actions_str = ', '.join(actions) if actions else '-'
            print(f'({i},{j}): {actions_str:<20}', end=' ')
        else:
            print(f'({i},{j}): {"n/a":<20}', end=' ')