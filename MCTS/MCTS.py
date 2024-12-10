import csv
import math
import random
from collections import defaultdict
import numpy as np
import random
from ReinforcementLearningAlgorithms.Enviorments.CatVSMonsters import CatVSMonsters
from ReinforcementLearningAlgorithms.Enviorments.GridWorld687 import GridWorld687
import numpy as np

class Node:
    id = 0
    visits = defaultdict(lambda: 0)

    def __init__(self, env, parent, state, q, UCT, reward=0.0, action=None):
        self.env = env
        self.parent = parent
        self.state = state
        self.id = Node.id
        Node.id += 1
        self.q = q
        self.UCT = UCT
        self.reward = reward
        self.action = action
        self.children = {}
    
    def fullExpand(self):
        if len(self.env.actions) == len(self.children):
            return True
        else:
            return False

    def select(self):
        if not self.fullExpand() or self.state == self.env.food:
            return self
        else:
            actions = list(self.children.keys())
            action = self.UCT.select(self.state, actions, self.q)
            return self.getNode(action).select()

    def expand(self):
        if self.state != self.env.food:
            actions = self.env.actions - self.children.keys()
            action = random.choice(list(actions))
            self.children[action] = []
            return self.getNode(action)
        return self

    def backPropagate(self, reward, child):
        action = child.action
        Node.visits[self.state] = Node.visits[self.state] + 1
        Node.visits[(self.state, action)] = Node.visits[(self.state, action)] + 1
        delta = (1 / (Node.visits[(self.state, action)])) * (
            reward - self.q.getQVal(self.state, action)
        )
        self.q.update(self.state, action, delta)
        if self.parent != None:
            self.parent.backPropagate(self.reward + reward, self)

    def getNode(self, action):
        nextStates, probs = self.env.p(self.state, action)
        ind = np.random.choice(len(nextStates), p=probs)
        nextS = nextStates[ind]
        for (child, _) in self.children[action]:
            if nextS == child.state:
                return child
        newNode = Node(
            self.env, self, nextS, self.q, self.UCT, self.env.reward(nextS), action
        )
        probability = 0.0
        nextStates, probs = self.env.p(self.state, action)
        for state, probability in zip(nextStates, probs):
            if state == nextS:
                self.children[action] += [(newNode, probability)]
                return newNode
    
class MCTS:
    def __init__(self, env, q, UCT, epsilon=0.0, decay=False):
        self.env = env
        self.q = q
        self.UCT = UCT
        self.epsilon = epsilon
        self.decay = decay

    def mcts(self, rootNode, iterations=10000):
        for i in range(iterations):
            selNode = rootNode.select()
            if selNode != self.env.food:
                child = selNode.expand()
                reward = self.simulate(child)
                selNode.backPropagate(reward, child)
            if (i % 250 == 0 and i != 0) or i == 9999:
                if self.decay:
                    self.epsilon -= 0.02375
                bestAction = None
                maxVal = float('-inf')
                for action in self.env.actions:
                    actionVal = self.q.getQVal(rootNode.state, action)
                    if actionVal > maxVal:
                        bestAction = action
                        maxVal = actionVal
                maxQVals[i][rootNode.state] = [bestAction, maxVal]
        return rootNode

    def simulate(self, node):
        state = node.state
        cumReward = 0.0
        depth = 0
        while state != self.env.food:
            if self.epsilon != 0.0:
                curQ = {action: self.q.getQVal(state, action) for action in self.env.actions}
                optActions = [action for action, val in curQ.items() if val == max(curQ.values())]
                curPi = {action: (1 - self.epsilon) / len(optActions) + self.epsilon / len(self.env.actions) if action in optActions else self.epsilon / len(self.env.actions) for action in self.env.actions}
                actions = list(curPi.keys())
                probabilities = list(curPi.values())
                action = random.choices(actions, probabilities, k=1)[0]
            else:
                action = random.choice(self.env.actions)
            nextStates, probs = self.env.p(state, action)
            ind = np.random.choice(len(nextStates), p=probs)
            nextS = nextStates[ind]
            cumReward += pow(self.env.gamma, depth) * self.env.reward(nextS)
            depth += 1
            state = nextS
        return cumReward
    
class ActionValueFunction:
    def __init__(self):
        self.qVals = defaultdict(lambda: defaultdict(float))

    def getAllQs(self):
        return self.qVals

    def getQVal(self, state, action):
        return self.qVals[state][action]

    def getMaxQVal(self, state, actions):
        return max([self.qVals(state, action) for action in actions], default=0.0)

    def update(self, state, action, delta):
        self.qVals[state][action] += delta

class UpperConfidenceTree:
    def __init__(self, c=1/math.sqrt(2)):
        self.c = c
        self.actionVisits = defaultdict(lambda: defaultdict(int))

    def select(self, state, actions, q):
        totVisits = sum(self.actionVisits[state][a] for a in actions)

        def UCB1(action):
            qVal = q.getQVal(state, action)
            visits = self.actionVisits[state][action]
            UCB1Score = 2 * self.c * (math.sqrt(math.log(2 * (totVisits + 1)) / (visits + 1)))
            return qVal + UCB1Score

        bestA = max(actions, key=UCB1)
        self.actionVisits[state][bestA] += 1  
        return bestA

"""
maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=2.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/regularMCTSExploreTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])  

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=math.sqrt(2))
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/regularMCTSExploreSqrtTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])         

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=1.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/regularMCTSExploreOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree()
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/regularMCTSExploreOneOverSqrtTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.05)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/regularMCTSExplorePointZeroFive.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.001)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/regularMCTSExplorePointZeroZeroOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.001)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=0.2).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CCatVSMonstersResults/EpsilonPointTwoMCTSExplorePointZeroZeroOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.001)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=0.1).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/EpsilonPointOneMCTSExplorePointZeroZeroOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.001)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=0.05).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/EpsilonPointZeroFiveMCTSExplorePointZeroZeroOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = CatVSMonsters()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.001)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=1.0, decay=True).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./MCTS/CatVSMonstersResults/EpsilonDecayMCTSExplorePointZeroZeroOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=2.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/regularMCTSExploreTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])  

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=math.sqrt(2))
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/regularMCTSExploreSqrtTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])         

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=1.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/regularMCTSExploreOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree()
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/regularMCTSExploreOneOverSqrtTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.05)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/regularMCTSExplorePointZeroFive.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=0.001)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/regularMCTSExplorePointZeroZeroOne.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=2.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=0.2).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/EpsilonPointTwoMCTSExploreTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])
"""
maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=2.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=0.1).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/EpsilonPointOneMCTSExploreTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=2.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=0.05).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/EpsilonPointZeroFiveMCTSExploreTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])

maxQVals = defaultdict(lambda: defaultdict(list))
env = GridWorld687()
q = ActionValueFunction()
UCT = UpperConfidenceTree(c=2.0)
counter = 0
for i in range(5):
    for j in range(5):
        print(counter)
        counter += 1
        state = (i,j)
        if state in env.forbidden_furniture or state == env.food:
            continue
        rootNode = MCTS(env, q, UCT, epsilon=1.0, decay=True).mcts(iterations=10000, rootNode=Node(env, None, state, q, UCT))
csv_file = "./ReinforcementLearningAlgorithms/MCTS/GridWorld687Results/EpsilonDecayMCTSExploreTwo.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Outer Key", "Inner Key (Tuple)", "Value"])
    for outer_key, inner_dict in maxQVals.items():
        for inner_key, value in inner_dict.items():
            writer.writerow([outer_key, inner_key, value])