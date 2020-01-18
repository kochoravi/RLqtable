import numpy as np
import pandas as pd
import random

class Game:
    rewards = None
    positionCol = None
    positionRow = None
    
    def __init__(self, startCol=1, startRow=1):
        self.rewards = pd.DataFrame({1:[0,1,2,3,4], 2:[1,2,3,4,5], 3:[2,3,4,5,6], 4:[3,4,5,6,7], 5:[4,5,6,7,8]}, index={1,2,3,4,5})
        self.positionCol = startCol
        self.positionRow = startRow
        
    def move(self, direction):
        reward = 0
        end = False
        if direction=='Up':
            self.positionRow -= 1    
        elif direction=='Down':
            self.positionRow += 1
        elif direction=='Left':
            self.positionCol -= 1  
        else:
            self.positionCol += 1
        
        #check if we lost
        if self.positionRow < 1 or self.positionRow > 5 or self.positionCol < 1 or self.positionCol > 5:
            end = True
            reward = -1000   
        #check if we have reached the end
        elif self.positionCol == 5 and self.positionRow == 5:
            end = True
            reward = self.rewards[self.positionCol][self.positionRow]
        else:
            end = False
            reward = self.rewards[self.positionCol][self.positionRow]
        
        #return reward and end of game indicator
        return (reward, end)

#states are in columns and actions are in rows
learning_rate=1
discount=0
random_explore=0.1
qtable = pd.DataFrame(100, index=['Up', 'Down', 'Left', 'Right'], columns=[11,12,13,14,15,21,22,23,24,25,31,32,33,34,35,41,42,43,44,45,51,52,53,54,55])

for i in range(1000):
    #print ("Game # " + str(i))
    game = Game()
    end_of_game = False
    while not end_of_game:
        #get current state 
        #state names are integers for ease of coding, it will be a two digit number with Col number and Row number in 1st and 2nd digits
        current_state = (game.positionCol*10)+game.positionRow
        #select the action with maximum reward
        max_reward_action = qtable[current_state].idxmax()
        #replace with random action to promote exploration and not get stuck in a loop
        if random.random() < random_explore:
            max_reward_action = qtable.index[random.randint(0,3)]
        #play the game with that action
        reward, end_of_game = game.move(max_reward_action)
        if (current_state==12):
            print("CS:" + str(current_state) + ", Action: " + max_reward_action + ", Reward: " + str(reward))
        #if end of game, then if the game is over, then no need to update the q value for the action taken, but is set to the reward value observed
        if end_of_game:
            qtable.loc[max_reward_action,current_state] = reward
            print ("----End----")
        else:
            #if not end of game, then get the next state's max q value - this is the estimate of optimal future value
            opimtal_future_value = qtable[(game.positionCol*10)+game.positionRow].max()
            #mulitpy this with the discount factor
            discounted_opimtal_future_value = discount * opimtal_future_value
            #so the new learned value will be observed immediate reward plus discounted future value estimate
            learned_value = reward + discounted_opimtal_future_value
            #the new refreshed q value for the action taken is old value plus new value adjusted for learning rate
            qtable.loc[max_reward_action,current_state] = (1 - learning_rate) * qtable[current_state][max_reward_action] + learning_rate * learned_value
        
