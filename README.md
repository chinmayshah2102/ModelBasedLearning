This Project implements Model based learning for pendulum environment in OpenAI gym.
First a data set is generated to learn the dynamics of the system and then neural network architecture is designed to learn the unknown dynamics of the environment using the data-set.
Finally an Astar planning algorithm is implemented on this environment. The state and action space for the problem are continuous. One of the methods for solving the problem is discretizing the state and action space. The problem statement here is to understand the effects of this discretization resolution with the complexity of planning. 

Astar Planning State space discretization
The algorithm compares the discretization resolution of state space vs number of steps taken by the pendulum to reach the goal state (zero degree vertical).

The graph of discretization resolution for state space vs number of steps to reach the goal state obtained was:
![BinsVsSteps](https://user-images.githubusercontent.com/38117206/57319247-25b78680-70ca-11e9-8a57-2dda5cd4cf38.png)

Astar Plannning Action space discretization
The algorithm compares the discretization resolution of action space vs number of steps taken by the pendulum to reach the goal state (zero degree vertical).

