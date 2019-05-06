import gym
import numpy as np

env = gym.make('Pendulum-v0')

#Initializing variables
state = np.zeros([3,2501,10])
a = np.zeros([2500,10])

#Generating 10 batches of 2500 observations each to generate stochastic data
for i in range(10):
    state[:,0,i] = env.reset()
    for j in range(2500):
        action = env.action_space.sample()
        state[:,j+1,i], reward, done, _ = env.step(action)
        a[j,i] = action
                
env.close()

#Pulling training data and label from the data generated
train_data = np.zeros([3,20000])
train_action = np.zeros(20000)
train_label = np.zeros([3,20000])

for i in range(10):
    train_data[:,(i*2000):((i+1)*2000)] = state[:,:2000,i]
    train_action[(i*2000):((i+1)*2000)] = a[:2000,i]
    train_label[:,(i*2000):((i+1)*2000)] = state[:,1:2001,i]
   
#Pulling test data and label from the data generated
test_data = np.zeros([3,5000])
test_action = np.zeros(5000)
test_label = np.zeros([3,5000])

for i in range(10):
    test_data[:,(i*500):((i+1)*500)] = state[:,2000:2500,i]
    test_action[(i*500):((i+1)*500)] = a[2000:2500,i]
    test_label[:,(i*500):((i+1)*500)] = state[:,2001:2501,i]

#Saving files for later use
np.save("Train_data",train_data)
np.save("Train_action",train_action)
np.save("Train_label",train_label)
np.save("Test_data",test_data)
np.save("Test_action",test_action)
np.save("Test_label",test_label)

