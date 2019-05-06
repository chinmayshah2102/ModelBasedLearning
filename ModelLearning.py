import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gym


training_data = np.zeros([4,20000])
training_data[0:3,:] = np.load("Train_data1.npy")
training_data[3,:] = np.load("Train_action1.npy")
training_label = np.zeros([3,20000])
training_label = np.load("Train_label1.npy")

training_data = torch.from_numpy(training_data).float()
training_label = torch.from_numpy(training_label).float()

model = nn.Sequential(
           nn.Linear(4,128),
           nn.ReLU(),
           nn.Linear(128,3))
loss_fn = nn.MSELoss()

learning_rate = 1e-4

optimizer = optim.Adam(model.parameters(), learning_rate)
loss_train = np.zeros(20000)

for i in range(20000):
    pred_label = model(training_data[:,i])
    loss = loss_fn(pred_label,training_label[:,i])
    model.zero_grad()
    loss.backward()
    optimizer.step()
    loss_train_temp = Variable(loss.data.clone(), requires_grad=False)
    loss_train[i] = loss_train_temp.numpy()

mean_loss_train = np.mean(loss_train)
print("mean_loss_train:",mean_loss_train)
print("Last_loss:",loss_train[-1])
    
test_data = np.zeros([4,5000])
test_data[0:3,:] = np.load("Test_data1.npy")
test_data[3,:] = np.load("Test_action1.npy")
test_label = np.zeros([3,5000])
test_label[0:3,:] = np.load("Test_label1.npy")

test_data = torch.from_numpy(test_data).float()
test_label = torch.from_numpy(test_label).float()
loss_test = np.zeros(5000)

for i in range(5000):
    pred_test_label = model(test_data[:,i])
    losst = loss_fn(pred_test_label,test_label[:,i])
    loss_test_temp = Variable(losst.data.clone(), requires_grad=False)
    loss_test[i] = loss_test_temp.numpy()
    
plt.figure(1)
plt.subplot(211)
plt.plot(loss_train)

plt.subplot(212)
plt.plot(loss_test)
plt.show()

mean_loss_test = np.mean(loss_test)
print("mean_loss_test:",mean_loss_test)

torch.save(model.state_dict(), "pendulum_weights.pt")