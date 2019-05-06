This Project implements Model based learning for pendulum environment in OpenAI gym.
First a data set is generated to learn the dynamics of the system and then neural network is used to learn the weights using the data-set.
Finally an Astar plannign algorithm is implemented on the pendulum environment by discretizing the state and action space. The code compared the discretization resolution vs number of steps taken by the pendulum to reach the goal state (zero degree vertical)
