#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:09:25 2018

@author: sleek_eagle
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:00:42 2018

@author: sleek_eagle
"""

 
import pygame
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy



from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.breadth_first import BreadthFirstFinder




# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (50, 50, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
COLORS = [BLACK,WHITE,BLUE,RED,YELLOW]
 


# Screen dimensions
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 700

 
#agents
NUM_AGENTS=10
AGENT_SIZE=15 #here width = height for simplicity
MAX_VELO = 5000 # this is in mm


NUM_ACTIONS = 5
LEFT,RIGHT,UP,DOWN,STAY = 0,1,2,3,4



MAP_PATH = "/home/sleek_eagle/research/gameTheory/E_learning/map.txt"
#read the low res map from file
def readMap():
    mapList = []
    file = open(MAP_PATH, 'r') 
    for line in file:
        if (len(line) > 1):
            chars = list(line)
            nums = []
            for i in range(0,len(chars)):
                if (chars[i].isdigit()):
                    nums.append(int(chars[i]))        
            mapList.append(nums)
    return mapList

#expand the low res map to fit into the screen
def createWalls(mapList):
    square_walls = []
    #calculate scale in y axis
    yscale = SCREEN_HEIGHT/len(mapList)
    if(len(mapList) > 0):
        xscale = SCREEN_WIDTH/len(mapList[0])
    else:
        print("data is not consistant!")
        return
    
    for i in range(len(mapList)):
        for j in range(len(mapList[i])):
            if (mapList[i][j] == 1):
                wall = Wall((j*xscale),(i*yscale), xscale, yscale)
                square_walls.append(wall)
    return square_walls
 
#action is a vector [player1_action,player2_action]
def move_agents(actions):
  state = get_state()
  next_state,player1_reward,player2_reward = get_next_state_and_selfish_reward(state,actions[0],actions[1])
  for i in range(len(agents)):
    agents[i].set_pos([next_state[i*2],next_state[i*2+1]])
  return next_state,player1_reward,player2_reward

def get_actions_taken(state,next_state,actions):
  isSame = True
  actions_taken = [4,4]
  n = 0
  for i in range(0,len(state)):
    if(not(state[i] == next_state[i])):
      isSame = False
    if ((i+1)%2 == 0):
      if(isSame):
        actions_taken[n] = actions[n]
        n += 1
        isSame = True
    return actions_taken

#calculate distance to goal to calculate reward
def findDistToGoal(mapList,pos,goal):
    #invert map for A* purposes. In this library, 0 is obstacle 1 is free. so invert
    mapMatrix = np.ones((len(mapList),len(mapList[0])))
    for i in range(0,len(mapList)):
        for j in range(0,len(mapList[0])):
            if (mapList[i][j] == 0):
                mapMatrix[i][j] = 1
            else:
                mapMatrix[i][j] = 0
            
    grid = Grid(matrix = mapMatrix)
    start = grid.node(pos[0],pos[1])
    end = grid.node(goal[0],goal[1])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        
    path, runs = finder.find_path(start, end, grid)
    print(grid.grid_str(path=path, start=start, end=end))
    print(path)
    dist = 0
    if (len(path) > 2):
        dist = len(path)-1
    elif (len(path) == 2):
        dist = 1
    elif (len(path) == 1):
      dist = 0
    else:
        dist = -1
    return dist
  
def get_state():
  state = []
  for i in range(len(agents)):
    center = agents[i].get_center_coords()
    pos = getMapCoordsFromPixcels(center)
    pos = [pos[0],pos[1]]
    state.append(pos[0])
    state.append(pos[1])
  return state

def check_reached_goals(state):
  is_agent1 = False
  is_agent2 = False
  if(state[0] == goals[0][0] and state[1] == goals[0][1]):
    is_agent1 = True
  if(state[2] == goals[1][0] and state[3] == goals[1][1]):
    is_agent2 = True
  
  return is_agent1,is_agent2

#I am both players! I can be at two places at onece!!! (semi-omniprecense)  
def get_next_state_and_selfish_reward(state,player1_action,player2_action):
  player1_reward = -1
  player2_reward = -1
  next_state = state
  isCollision = False
  player1_next_pos,player1_wall_collide = get_next_pos([state[0],state[1]],player1_action)
  #handle undefined cases. actually theres no reson for next_pos to be -1. just for kicks!
  if (min (player1_next_pos) < 0):
    return -1
  player2_next_pos,player2_wall_collide  = get_next_pos([state[2],state[3]],player2_action)
  if(min (player2_next_pos) < 0):
    return -1
  #punishment for hitting walls
  if(player1_wall_collide):
    player1_reward-=5
  if(player2_wall_collide):
    player2_reward-=5
    
  #detect and resolve collitions
  collision = False
  #case1 - players try to move to the same grid
  player_dist = sum([abs(player1_next_pos[0] - player2_next_pos[0]),abs(player1_next_pos[1] - player2_next_pos[1])])
  if(player_dist == 0):
    collision = True
  #case2 - players try to walk through each other and they end up at each others previous grids
  if ((player1_next_pos == [state[2],state[3]]) and (player2_next_pos == [state[0],state[1]])):
    collision = True
  #collision!
  #if collision, move back to last state (no effective change)
  if (collision):
    player1_reward-=10
    player2_reward-=10
    next_state = state
  else:
    next_state = [player1_next_pos[0],player1_next_pos[1],player2_next_pos[0],player2_next_pos[1]]
    
  #did players reach goals due to the last actions they took ?
  already_goal = check_reached_goals(state)
  next_goal = check_reached_goals(next_state)
  if ((not already_goal[0]) and next_goal[0]):
    player1_reward+=100
  if ((not already_goal[1]) and next_goal[1]):
    player2_reward+=100
    
    '''  
  #calc player1 reward
  if ((not player1_wall_collide) and (not isCollision) and not (player1_action == STAY)):# if player 1 moved it gets this reward
    dist = findDistToGoal(mapList,[player1_next_pos[0],player1_next_pos[1]],agents[0].goalPos)
    dist_reward = 20 - dist
    player1_reward += dist_reward
    #calc player2 reward
  if ((not player2_wall_collide) and (not isCollision) and not (player2_action == STAY)):# if player 1 moved it gets this reward
    dist = findDistToGoal(mapList,[player2_next_pos[0],player2_next_pos[1]],agents[1].goalPos)
    dist_reward = 20 - dist
    player2_reward += dist_reward 
    '''
    
  return next_state,player1_reward,player2_reward
    

def get_next_pos(pos,action):
  wall_collide = False
  possible = getPossibleActions(pos)
  new_pos = [-1,-1]
  if (action >= NUM_ACTIONS):
    return  new_pos
  if(possible[action] == 0):
    wall_collide = True
    return pos,wall_collide
  if (action == LEFT):
    new_pos = [pos[0]-1,pos[1]]
  elif (action == RIGHT):
    new_pos = [pos[0]+1,pos[1]]
  elif (action == UP):
    new_pos = [pos[0],pos[1]+1]
  elif (action == DOWN):
    new_pos = [pos[0],pos[1]-1]
  else:
    new_pos = pos
  return new_pos,wall_collide

def getPixcelRectFromMapCoords(coords):
    left = GRID_WIDTH*coords[0]
    top = GRID_HEIGHT*coords[1]
    rect = pygame.Rect(left,top,GRID_WIDTH,GRID_HEIGHT)
    return rect

#point is y,x coords 
def getMapCoordsFromPixcels(point):
    grid_coord = [math.floor(point[0]/GRID_HEIGHT) , math.floor(point[1]/GRID_WIDTH)]
    return grid_coord

class Point(pygame.sprite.Sprite):
    # Constructor function
    def __init__(self,rect,col):
        # Call the parent's constructor
        super().__init__()
        [x,y] = [rect.left,rect.top]
        # Set height, width
        self.image = pygame.Surface([rect.width, rect.width])
        self.image.fill(col)
 
        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
 
        
class Agent(pygame.sprite.Sprite):
 
    # Constructor function
    def __init__(self,pos,col):
      # Call the parent's constructor
      super().__init__()
      rect = getPixcelRectFromMapCoords(pos)
      [x,y] = [rect.left,rect.top]
      print("init rect" + str([x,y]))
      # Set height, width
      self.image = pygame.Surface([rect.width, rect.width])
      self.image.fill(col)
 
      # Make our top-left corner the passed-in location.
      self.rect = self.image.get_rect()
      self.rect.y = y
      self.rect.x = x
 
      self.goalPos = []
      self.reached_goal = False
      self.action = 4
      self.pos = pos
    
    #actions 0-left,1-right,2-up,3-down,4-stay
    def setAction(self,action):
      self.action = action
      print("set action to " + str(action))
    def setGoal(self,goalPos):
      self.goalPos = goalPos
    def get_center_coords(self):
      return [(self.rect.right + self.rect.left)/2,(self.rect.bottom + self.rect.top)/2]
    def set_pos(self,pos):
      self.pos = pos
        
    def update(self):
      rect = getPixcelRectFromMapCoords(self.pos)
      print("pos = " + str(self.pos))
      self.rect.x = rect.x
      self.rect.y = rect.y
      print("rect x,y = "  + str(rect.x) + " " + str(rect.y))
      
      
      '''
      print("in update!")
      if (self.action == 0):
          #move left
          self.rect.x -= GRID_WIDTH
      elif (self.action == 1):
          #move right
          self.rect.x += GRID_WIDTH
      elif (self.action == 2):
          #move up
          self.rect.y += GRID_HEIGHT
      elif(self.action == 3):
          #move down
          self.rect.y -= GRID_HEIGHT
          '''
          
            

class Wall(pygame.sprite.Sprite):
    """ Wall the player can run into. """
    def __init__(self, x, y, width, height):
        """ Constructor for the wall that the player can run into. """
        # Call the parent's constructor
        super().__init__()
 
        # Make a blue wall, of the size specified in the parameters
        self.image = pygame.Surface([width, height])
        self.image.fill(BLUE)
 
        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
        

def isOccupied(rect):   
    #now check other players and walls 
    sprites = all_sprite_list.sprites() 
    for i in range(0,len(sprites)):
        if(rect.colliderect(sprites[i].rect)):
            return True
    return False

def initAgent(pos,col):
  agent = Agent(pos,col)
  return agent

def createPlayers(playerPos,goals):
    agents = []
    for i in range(0,len(playerPos)):
        agent = initAgent(playerPos[i],COLORS[3+i])
        agent.setGoal(goals[i])
        all_sprite_list.add(agent)
        agents.append(agent)
    return agents
    
def chooseGoals(mapList):
    width  = len(mapList[0])-2 #minus 2 because of wall width. walls surround the map like Vatican city
    height = len(mapList)-2
    #goal positions are in the top and bottom most empty cells
    #goalPos =  [[1,random.randint(0,width-1)],[5,random.randint(0,width-1)]]
    goalPos = [[2,5],[2,1]]
    return goalPos    

def choosePlayerInitPos(mapList):
    playerPos = []    
    #pre-determined player_pos
    playerPos = [[2,1],[2,5]]
    return playerPos
  
def choosePlayerInitPos_rand(mapList):
    playerPos = []
    #random player_pos
    
    width  = len(mapList[0])
    height = len(mapList)
    #player initial positions are any two random empty positions on the map
    #first find an empty spot
    for i in range(0,2):
        x = random.randint(0,width-1)
        y = random.randint(0,height-1)
        isRecorded = [x,y] in playerPos
        itr = 0
        while((mapList[y][x] == 1) or (isRecorded)):
          itr+=1
          x = random.randint(0,width-1)
          y = random.randint(0,height-1)
          isRecorded = [x,y] in playerPos
          if (itr > height*width):
            return [-1,-1]
        playerPos.append([x,y])
    
    return playerPos
  
'''        
for i in range(0,1000):
  print(i)
  pos =    choosePlayerInitPos_rand(mapList)
  if ((pos[0] == pos [1]) or (mapList[pos[0][1]][pos[0][0]] == 1) or (mapList[pos[1][1]][pos[1][0]] == 1)):
    print("caut !!1")
    break
'''
  
     
def getPossibleActions(pos):
    x,y = pos[0],pos[1]
    left,right,up,down,stay = 0,0,0,0,1
    if(not((x-1) < 0)):
        if(mapList[y][x-1] == 0):
            left = 1
    if(not((x+1) >= len(mapList[0]))):
        if(mapList[y][x+1] == 0):
            right = 1
    if(not((y-1) < 0)):
        if(mapList[y-1][x] == 0):
            down = 1
    if(not((y+1) >= len(mapList))):
        if(mapList[y+1][x] == 0):
            up = 1
    return left,right,up,down,stay



def init_simulation():
  global screen
  global all_sprite_list
  global wall_list
  global mapList
  global clock

  pygame.init()    
  screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])    
  # Set the title of the window
  pygame.display.set_caption('Skynet')
 
  # List to hold all the sprites
  all_sprite_list = pygame.sprite.Group()
 
  # Make the walls. (x_pos, y_pos, width, height)
  wall_list = pygame.sprite.Group()

  #read low res map from file
  mapList = readMap()
  #create scales version of walls from the low res map data
  wallList = createWalls(mapList)
  for i in range(0,len(wallList)):
    wall_list.add(wallList[i])
    all_sprite_list.add(wallList[i])
 

  clock = pygame.time.Clock()
  
  #two more global variables
  global GRID_WIDTH
  global GRID_HEIGHT
  GRID_WIDTH = SCREEN_WIDTH/len(mapList[0])
  GRID_HEIGHT = SCREEN_HEIGHT/len(mapList)

  
  
def init_agents():
  global goals
  goals = chooseGoals(mapList)
  playerPos = choosePlayerInitPos(mapList)
  global agents
  agents = createPlayers(playerPos,goals)
  return agents
  
def update():
  all_sprite_list.update()
  screen.fill(BLACK)
  all_sprite_list.draw(screen)
  pygame.display.flip()
  
def quit_simulation():
  pygame.quit()
  
    
          
# Call this function so the Pygame library can initialize itself
init_simulation()
init_agents()
update()

move_agents([1,4])
update()
get_state()








