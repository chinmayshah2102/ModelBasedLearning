This Project implements Model based learning for pendulum environment in OpenAI gym.

-> First a data set is generated based on the unknown dynamics of the system.  
-> Then a neural network is designed to learn this unknown dynamics of the environment using the dataset.  
-> Finally an Astar planning algorithm is implemented to search a sequence of state -> action -> state to make the pendulum stand upright.  
-> The state and action space for this problem is a continuous space. One of the methods for solving the problem is discretizing the state and action space.

The problem statement here is to understand the effects of discretization resolution with the complexity of planning. 

State space discretization:
The graph below shows the number of steps taken by the pendulum to reach the goal state (zero degree vertical) for different values of the discretization resolution of state space.
![BinsVsSteps](https://user-images.githubusercontent.com/38117206/57319247-25b78680-70ca-11e9-8a57-2dda5cd4cf38.png)

Action space discretization:
The graph below shows the number of steps taken by the pendulum to reach the goal state (zero degree vertical) for different values of the discretization resolution of action space.
![ActionBinsVsSteps](https://user-images.githubusercontent.com/38117206/57319769-5c41d100-70cb-11e9-9cdb-d01be2f37ea6.png)


