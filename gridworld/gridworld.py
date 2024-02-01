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
    
    def policy_iteration(self, gamma, theta):
        while True:
            self.iterative_policy_evaluation(gamma, theta)
            stable = True
            for i in range(4):
                for j in range(4):
                    if is_terminal(i, j):
                        continue
                    # TODO: does pi need to be deterministic? how to handle ties?
                    old_action = max(self.policy[(i, j)], key=self.policy[(i, j)].get)
                    self.policy_improvement(i, j, gamma)
                    new_action = max(self.policy[(i, j)], key=self.policy[(i, j)].get)
                    if old_action != new_action:
                        stable = False
            if stable:
                break

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
mean = agent.simulate(1_000_000)
print(mean)