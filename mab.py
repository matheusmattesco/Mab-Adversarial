import random
import math
import numpy as np
class Arm:
    def create_arms(self, numArms, maxReward):
        self.numArms = numArms
        fakeArms = []
        for _ in range(numArms):
            num1 = round(random.uniform(0, maxReward), 4)
            num2 = round(random.uniform(num1, maxReward), 4)
            fakeArm = [num1, num2]
            fakeArms.append(fakeArm)
        self.fakeRewards = fakeArms
        self.maxReward = maxReward
        self.sumFakeRewards = sum([sum(fakeArm) for fakeArm in fakeArms])
        
    def get_makespan(self, arm, adversarial_prob):
        if random.random() < adversarial_prob:
            return round(random.uniform(0, self.maxReward), 4)
        else:
            return round(random.uniform(self.fakeRewards[arm][0], self.fakeRewards[arm][1]), 4)
    
    def get_reward(self, arm):
        return round(random.uniform(self.fakeRewards[arm][0], self.fakeRewards[arm][1]), 4)

class MabAdversarial:
    def __init__(self, numArms, maxReward, gamma, T, adversarial_prob):
        self.arms = Arm()
        self.arms.create_arms(numArms, maxReward)
        self.gamma = gamma
        self.T = T
        self.weights = [1.0] * self.arms.numArms
        self.rewards = [1.0] * self.arms.numArms
        self.arm_counts = [0] * self.arms.numArms
        self.arm_counts2 = [0] * self.arms.numArms
        self.rewards_history = [[] for _ in range(self.arms.numArms)]
        self.fakeRewards = self.arms.fakeRewards
        self.choicesArms = []
        self.allRewards = []
        self.allRewards2 = []
        self.regret = []
        self.regret2 = []
        self.mean_rewards = []
        self.adversarial_prob = adversarial_prob
        self.results = []
        self.results2= []

    def select_arm(self, t):
        exploration_prob = max(0, self.gamma * t / self.T) 
        if random.random() < exploration_prob:
            return random.randrange(self.arms.numArms)
        else:
            return max(range(self.arms.numArms), key=lambda arm: self.weights[arm])

    def run(self):
        for t in range(self.T):
            arm = self.select_arm(t)
            reward = self.arms.get_makespan(arm, self.adversarial_prob)
            self.rewards[arm] += reward
            self.weights[arm] = math.exp(self.gamma * (self.rewards[arm] / max(1, self.arm_counts[arm])))
            self.arm_counts[arm] += 1
            self.rewards_history[arm].append(reward)
            self.choicesArms.append(arm)           
            self.results.append((t+1, arm, reward))
        
        self.mean_rewards = [np.mean(rewards) if rewards else 0 for rewards in self.rewards_history]
        self.allRewards = self.rewards_history
        self.regret = [[max(self.fakeRewards[i]) - r for r in rewards] for i, rewards in enumerate(self.allRewards)]
        weights_means = [weight / sum(self.weights) for weight in self.weights]
        
        return self.mean_rewards, self.arm_counts, self.allRewards, self.regret, self.fakeRewards, self.results
    
    def run_no_adversarial(self):
        for t in range (self.T):
            arm = self.select_arm(t)
            reward = self.arms.get_reward(arm)
            self.rewards[arm] += reward
            self.weights[arm] = math.exp(self.gamma * (self.rewards[arm] / max(1, self.arm_counts2[arm])))
            self.arm_counts2[arm] += 1
            self.rewards_history[arm].append(reward)
            self.choicesArms.append(arm)           
            self.results2.append((t+1, arm, reward))
        
        self.mean_rewards2 = [np.mean(rewards) if rewards else 0 for rewards in self.rewards_history]
        self.allRewards2 = self.rewards_history
        self.regret2= [[max(self.fakeRewards[i]) - r for r in rewards] for i, rewards in enumerate(self.allRewards2)]
        weights_means = [weight / sum(self.weights) for weight in self.weights]
        
        return self.mean_rewards2, self.arm_counts2, self.allRewards2, self.regret2, self.results2
