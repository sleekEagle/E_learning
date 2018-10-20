#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:17:42 2018

@author: sleek_eagle
"""
import numpy as np
import environment.mapList
import environment
import random
import matplotlib.pyplot as plt
from time import sleep




#state - my (x,y) and other player (x,y). what else we need other than you and me ?
#state action - state - my_action - other_action

NUM_ACTIONS = 5
LEFT,RIGHT,UP,DOWN,STAY = 0,1,2,3,4

# selfish_Q is defined as sefish_Q[my_x][my_y][other_x][other_y][my_action][other_action]
#these values are in the perspective of player 1. when we want to update plaer 2's value have to convert the
#state-action vector to player 1's perspective with flip_state_action function
selfish_Q = np.zeros((len(mapList[0]),len(mapList),len(mapList[0]),len(mapList),NUM_ACTIONS,NUM_ACTIONS))

E = [0.6,0.2]
NUM_PLAYERS = 2
ALPHA = 0.3
GAMMA = 0.9

root = '/home/sleek_eagle/research/gameTheory/E learning/'


#flip state and actions (the game is symmetric for two players)
def flip_state(state):
  flipped = [-1,-1,-1,-1]
  #same x because LEFT and RIGHT actions are same
  flipped[0] = state[2]
  flipped[2] = state[0]
  
  #flip y on midle coord
  flipped[1] = 3 + (3 - state[3])
  flipped[3] = 3 + (3 - state[1])
  return flipped

#convert UP -> DOWN, DOWN -> UP and swap actions
def flip_action(action):
  if (action == DOWN):
    flipped = UP
  elif (action == UP):
    flipped = DOWN
  else:
    flipped = action
  return flipped

#given the state-action in p1 's perspective, this gives that in p2's perspective
def flip_state_action(state_action):
  flipped = [-1,-1,-1,-1,-1,-1]
  flipped[0:4] = flip_state(state_action[0:4])
  
  flipped[4] = flip_action(state_action[5])
  flipped[5] = flip_action(state_action[4])
    
  return flipped
  
def get_state_reward_matrix(state):
    #testing
    return np.array([[0,1,5,4,-2],[3,-3,1,6,2],[4,2,-3,1,0],[0,0,2,-1,2],[3,6,-2,0,1]])
    #return selfish_Q[state[0]][state[1]][state[2]][state[3]]
def get_state_p_values(state,player):
    if (player == 1):
        return player1_p[state[0]][state[1]][state[2]][state[3]]
    elif (player == 2):
        return player2_p[state[0]][state[1]][state[2]][state[3]]
    
#calculate empathy q values
def calc_empathy(state_action,e,p_values):
    emp_rewards = np.zeros((NUM_ACTIONS))
    flipped_state_action = flip_state_action(state_action)
    for i in range(0,NUM_ACTIONS):
        my_reward = 0
        other_reward = 0
        for j in range(0,NUM_ACTIONS):
            my_reward += p_values[j] * get_state_reward_matrix(state_action)[i][j]
            other_reward += p_values[j] * get_state_reward_matrix(flipped_state_action)[j][i]
        emp_rewards[i] += (1-e)*my_reward + e*other_reward
    return emp_rewards

#deterministic action selection           
def select_action(prob):
    prob_mul=prob*100
    r = random.randint(1,100)
    action = -1
    top,bottom = 0,0
    for i in range(0,len(prob_mul)):
        top += prob_mul[i]
        if ((r <= top) and (r >= bottom)):
            action = i
            break
        bottom += prob_mul[i]
    return action

#calc this for player 1 or 2
def calc_state_value(state,player):
  goalPos = player.goalPos
  dist = findDistToGoal(mapList,[state[0],state[1]],goalPos)
  
def est_other_action_prob(est_e, state):
  my_reward = get_my_reward(state)
  other_reward = get_other_reward(state)
 
def get_emp_rewards(state,e,p_values):
  emp_rewards = np.zeros((NUM_ACTIONS))
  my_rewards = get_my_reward(state)
  other_reward = get_other_reward(state)
  for i in range(0,NUM_ACTIONS):
      my_reward = 0
      other_reward = 0
      for j in range(0,NUM_ACTIONS):
          my_reward += p_values[j] * my_rewards[i][j]
          other_reward += p_values[j] * other_rewards[i][j]
      emp_rewards[i] += (1-e)*my_reward + e*other_reward
  return emp_rewards
  
 
def get_p1_reward(state):
  p1_rewards = selfish_Q[state[0]][state[1]][state[2]][state[3]]
  return p1_rewards

def get_p2_reward(state):
  flipped = flip_state(state)
  p2_reward = selfish_Q[flipped[0]][flipped[1]][flipped[2]][flipped[3]]
  return p2_reward
  
def get_empathy_matrix(state,e):
  p1_reward = get_p1_reward(state)
  p2_reward = get_p2_reward(state)
  emp_matrix = np.zeros((NUM_ACTIONS,NUM_ACTIONS))
  for i in range(0,NUM_ACTIONS):
    for j in range(0,NUM_ACTIONS):
      emp_matrix[i][j] = (1-e)*p1_reward[i][j] + e*p2_reward[flip_action(j)][flip_action(i)]
  return emp_matrix

def get_max_row_col(mat):
  ind = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
  return ind

'''
define various types of policies
make them interchangeable! so we can swap them with one another on the go!
'''

def get_minimax_action(mat):
  shape = mat.shape
  mins = []
  for i in range(0,shape[0]):
    mins.append(min(mat[i]))
  argmax = np.argmax(mins)
  return argmax,mins[argmax]
    

#max actions for both players
#this is the policy!!
#change this if I want to change the policy!
def get_max_actions(state):
  p1_empathy_matrix = get_empathy_matrix(state,E[0])
  p2_empathy_matrix = get_empathy_matrix(state,(1-E[1]))
  p1_max = get_max_row_col(p1_empathy_matrix)[0]
  p2_max = get_max_row_col(p2_empathy_matrix)[1]
  return [p1_max,p2_max],[p1_empathy_matrix[p1_max][p2_max],p2_empathy_matrix[p1_max][p2_max]]

def get_max_avg_actions(state):
  p1_avgs,p2_avgs = get_avg_empathy_values(state)
  p1_action,p2_action = [np.argmax(p1_avgs),np.argmax(p2_avgs)]
  p1_max_avg,p2_max_avg = [p1_avgs[p1_action],p2_avgs[p2_action]]
  return p1_action,p2_action,p1_max_avg,p2_max_avg
  

def get_avg_empathy_values(state):
  p1_empathy_matrix = get_empathy_matrix(state,E[0])
  p2_empathy_matrix = get_empathy_matrix(flip_state(state),(1-E[1]))
  p1_avgs = p1_empathy_matrix.mean(1)
  p2_avgs = p2_empathy_matrix.mean(0)
  return p1_avgs,p2_avgs
  

def select_possible_rand_actions(state):
  random_actions = [random.randint(0,4),random.randint(0,4)]
  p1_possible_actions = getPossibleActions([state[0],state[1]])
  p2_possible_actions = getPossibleActions([state[2],state[3]])
  possible = False
  itr = 0
  while(not possible):
    itr+=1
    random_actions = [random.randint(0,4),random.randint(0,4)]
    if ((p1_possible_actions[random_actions[0]] == 1) and (p2_possible_actions[random_actions[1]] == 1)):
      possible = True
      return random_actions
    if (itr > 100):
      return [-1,-1]
  
  

def select_actions(state,prob,e1,e2):
  prob *=100
  r = [random.randint(0,100),random.randint(0,100)]
  max_actions = get_max_actions(state)[0]
  random_actions = select_possible_rand_actions(state)
  actions = [-1,-1]
  for i in range(0,NUM_PLAYERS):
    if(r[i]<prob):
      actions[i] = random_actions[i]
      print("random")
    else:
      actions[i] = max_actions[i]
      print("max_action")
  return actions

def get_state_values(state):
  [p1_action,p2_action],[p1_max,p2_max] = get_max_actions(state)
  return p1_max,p2_max
 
def Q_update(state,actions,reward,next_state_value):
  global selfish_Q
  p1_goal,p2_goal = check_reached_goals(next_state_value)
  selfish_Q[state[0]][state[1]][state[2]][state[3]][actions[0]][actions[1]] = (1-ALPHA)*selfish_Q[state[0]][state[1]][state[2]][state[3]][actions[0]][actions[1]] + ALPHA*(reward+GAMMA*next_state_value)

def update_Q_values(state,actions,next_state,rewards):
  state_action = [state[0],state[1],state[2],state[3],actions[0],actions[1]]
  p1_state = state
  p1_next_state = next_state
  
  p2_state = flip_state(state)
  p2_state_action = flip_state_action(state_action)
  p2_next_state = flip_state(next_state)
  
  next_state_values = get_state_values(next_state)
  
  Q_update(p1_state,actions,rewards[0],next_state_values[0])
  
  Q_update(p2_state_action[0:4],p2_state_action[4:6],rewards[1],next_state_values[1])

def get_avg_Q_value():
  return np.mean(selfish_Q)

  

#stochastic action selection
def action_probabilities(emp_rewards):
    #normalize
    minimum = min(emp_rewards)
    scaled = (emp_rewards - minimum)+1
    #rescale so sum is 1
    scaled/=sum(scaled)
    return scaled
     
            
my_reward = np.array([[1,2,3,4,2],[-1,-3,2,6,3],[6,3,-5,10,2],[1,1,1,0,0],[0,0,3,6,-3]])
other_reward = np.array([[-2,3,4,5,1],[0,0,2,3,4],[-10,-3,2,3,4],[0,0,2,3,4],[1,1,1,3,3]])
e=0.1

def init():
  init_simulation()
  init_agents()
  update()

l = []
def run_episode(explore,delay):
  init()
  itr = 0
  while(1):
    if(delay > 0):
      sleep(delay)
    itr +=1
    state = get_state()
    actions = select_actions(state,explore,E[0],E[1])
    is_agent1,is_agent2 = check_reached_goals(state)
    
    if (is_agent1):
      actions = [4,actions[1]]
    if (is_agent2):
      actions = [actions[0],4]

    next_state , p1_reward, p2_reward = move_agents(actions)
    if (is_agent1):
      p1_reward = 0
    if (is_agent2):
      p2_reward = 0
    rewards =  [p1_reward,p2_reward]
    update_Q_values(state,actions,next_state,rewards)
    update()
    avg = get_avg_Q_value()
    l.append(avg)
               
    if((is_agent1 and is_agent2)):
      print("iterations = " + str(itr))
      return avg

    
def reset_Q_Learning():
  global l
  global selfish_Q
  l=[]
  selfish_Q = np.zeros((len(mapList[0]),len(mapList),len(mapList[0]),len(mapList),NUM_ACTIONS,NUM_ACTIONS))


  

prev_avg = -5
explore = 1
while(1):
  avg = run_episode(explore)
  if (prev_avg == 0): prev_avg = avg
  per = (avg-prev_avg)/prev_avg
  if (per > 0.8):
    explore-=0.1
  else:
    explore+=0.1
  if(explore >=1):
    explore = 0.9
  if(explore <=0):
    explore = 0.1
  prev_avg = avg
  
while(1):
  run_episode(0,0.4)

reset_Q_Learning()  
quit_simulation()
ar = np.array(l)
plt.plot(ar[:,0])


# for agent 1
a = selfish_Q[state[0]][state[1]][state[2]][state[3]][actions[0]][actions[1]]
b = selfish_Q[next_state[0]][next_state[1]][next_state[2]][next_state[3]][next_state[0]][next_state[1]]
new_value = (1-alpha) * a + alpha * (p1_reward + b)

#for agent 2
a = selfish_Q[state[2]][state[3]][state[0]][state[1]][actions[0]][actions[1]]
b = selfish_Q[next_state[2]][next_state[3]][next_state[0]][next_state[1]][next_state[0]][next_state[1]]
new_value = (1-alpha) * a + alpha * (p1_reward + b)



