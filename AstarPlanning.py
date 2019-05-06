import gym
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable


#Defining edge cost and heuristic functions
def edgeCost(action):     
    return np.abs(action)
     
def heuristic(curr_theta, curr_thetadot, action):
    heuristic_cost = (pow(curr_theta,2)+0.1*pow(curr_thetadot,2)+0.001*pow(action,2))   
    return heuristic_cost
  
#Defining model architecture to load weights
model = nn.Sequential(
           nn.Linear(4,128),
           nn.ReLU(),
           nn.Linear(128,3))

model.load_state_dict(torch.load("pendulum_weights.pt"))
model.eval()

#Initializing pendulum gym environment
env = gym.make('Pendulum-v0')

#Discretizing action space
actions = np.linspace(-2,2,num=10)
bins={}

for i in range(99):    
    bins[i] = float('inf')

#Running a loop for planning in discretized space (here n indicated state space discretized resolution)
for i in range(99):
    
    # n indicates nuber of bins
    n = i+2
    state_end_points_theta = np.linspace(-3*np.pi/2,np.pi/2,num=(n+1))
    state_end_points_thetadot = np.linspace(-8,8,num=(n+1))
    
    #Generating discrete possible state
    discrete_states_theta = []
    discrete_states_thetadot = []
    
    #Representing discrete bins with the average value of interval
    for j in range(n):
        discrete_mean_theta = (state_end_points_theta[j]+state_end_points_theta[j+1])/2
        discrete_mean_thetadot = (state_end_points_thetadot[j]+state_end_points_thetadot[j+1])/2
        theta_temp = (np.cos(discrete_mean_theta),np.sin(discrete_mean_theta))
        discrete_states_theta.append(theta_temp)
        discrete_states_thetadot.append(discrete_mean_thetadot)
    
    #defining a start position
    curr_state = (-1,0,0)

    #Assigning start position to the closest discrete state
    min_dist_theta = np.argmin(np.linalg.norm(np.asarray(discrete_states_theta).T-np.reshape(np.asarray(curr_state[:2]),(2,1)),axis=0))
    min_dist_thetadot = np.argmin(np.abs(np.asarray(discrete_states_thetadot)-curr_state[2]))
    curr_discrete_state_theta = discrete_states_theta[min_dist_theta]
    curr_thetadot = discrete_states_thetadot[min_dist_thetadot]
    curr_theta = np.arctan2(curr_discrete_state_theta[1],curr_discrete_state_theta[0])
    
    #Defining goal state
    goal_state = (1,0,0)

    #Assigining goal state to closest discrete state
    min_dist_goal_theta = np.argmin(np.linalg.norm(np.asarray(discrete_states_theta).T-np.reshape(np.asarray(goal_state[:2]),(2,1)),axis=0))
    min_dist_goal_thetadot = np.argmin(np.abs(np.asarray(discrete_states_thetadot)-goal_state[2]))
    goal_discrete_state_theta = discrete_states_theta[min_dist_goal_theta]
    goal_thetadot = discrete_states_thetadot[min_dist_goal_thetadot]
    
    #Defining tolerance for exit condition
    tol = 1e-4
    
    #Initializing open, closed list and Parent nodes for Astar planning
    CList = set()
    OList = set([(curr_discrete_state_theta,curr_thetadot)])
    ParentNode = {}
    
    #Storing edgecost from start to current position
    G = {}
    G[(curr_discrete_state_theta,curr_thetadot)] = 0 
    
    #Storing heuristic cost from current to goal position
    F = {} 
    F[(curr_discrete_state_theta,curr_thetadot)] = heuristic(curr_theta, curr_thetadot,0)
    
    while(len(OList)>0):   
        
        #Replacing current state with the state with lowest F cost
        currState = ()
        curreStateFvalue = 0
        for states in OList:
            if len(currState)==0 or F[states] < curreStateFvalue:
                curreStateFvalue = F[states]
                currState = states
                curr_theta = np.arctan2(currState[0][1],currState[0][0])
                curr_thetadot = currState[1]
        
        #If goal state is reached retrace the path
        if np.all(np.abs(np.asarray(currState[0])-np.asarray(goal_discrete_state_theta))< tol) and np.abs(currState[1]-goal_thetadot)< tol:
            
            #Retrace the path backward using parent nodes
            path = [currState]
            for currState in ParentNode:
                currState = ParentNode[currState]
                path.append(currState)
            path.reverse()
            bins[n] = len(path)
            print("Path length",len(path)) 
            print("Path ratio",len(path)/n) 
            break
        
        #Finding possible next discrete states using the learned model and updating the parent nodes and Gscore of the next states
        poss_next_theta = []
        poss_next_thetadot = []
        
        for a in actions:
            state_temp = np.zeros(4)
            state_temp[:-2] = currState[0]
            state_temp[-2] = currState[1]
            state_temp[-1] = a        
            state_tensor = torch.from_numpy(state_temp).float()
        
            poss_next_state_tensor = model(state_tensor)
            
            poss_next_temp = Variable(poss_next_state_tensor.data.clone(), requires_grad=False)
            poss_next_state = poss_next_temp.numpy()
            
            min_dist_next_theta = np.argmin(np.linalg.norm(np.asarray(discrete_states_theta).T-np.reshape(poss_next_state[:2],(2,1)),axis=0))
            min_dist_next_thetadot = np.argmin(np.abs(np.asarray(discrete_states_thetadot)-poss_next_state[2]))
            
            poss_next_theta.append(discrete_states_theta[min_dist_next_theta])
            poss_next_thetadot.append(discrete_states_thetadot[min_dist_next_thetadot])
    
            neighbour_theta = discrete_states_theta[min_dist_next_theta]
            next_theta = np.arctan2(neighbour_theta[1],neighbour_theta[0])
            
            next_thetadot  = discrete_states_thetadot[min_dist_next_thetadot]
            
            neighbour = (neighbour_theta,next_thetadot)
            
            if neighbour in CList: 
                continue 
            Gscore = G[currState] + edgeCost(a)
 
            if neighbour not in OList:
                OList.add(neighbour) 
            elif Gscore >= G[neighbour]:
                continue 

            ParentNode[neighbour] = currState
            G[neighbour] = Gscore
            H = heuristic(next_theta, next_thetadot,a)
            F[neighbour] = G[neighbour] + H
            
        OList.remove(currState)
        CList.add(currState)
        
lists = sorted(bins.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.xlabel('Discretization Resolution')
plt.ylabel('Number of steps to goal')
plt.title('For number of actions : 10')
plt.savefig('BinsVsSteps.png')
plt.show()
