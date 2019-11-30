# Import the relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random input data to train on
# the number of observations
observations = 1000 # we observe 1000 points
# observations = 10000 # we observe 10000 points



# Generate random input data to train on
## Generate observations for input 01
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
print('Input xs has shape - {}'.format(xs.shape))
# print(xs)
# plt.plot(xs)
# plt.show()
## Generate observations for input 02
zs = np.random.uniform(low=-10, high=10, size=(observations, 1))
print('Input zs has shape - {}'.format(zs.shape))
# Create the inputs for the deep learning models
inputs = np.column_stack((xs,zs))
print('Input for model has shape - {}'.format(inputs.shape))
# print(inputs)
# plt.plot(inputs)
# plt.show()

# Create the targets
## Model
'''
targets = f(x,z) = 2*xs - 3*zs + 5 + noise
'''
## create the targets we will aim at
noise = np.random.uniform(low=-1,high=1,size=(observations,1))
targets = 13*xs +7*zs -12 + noise # 2D Array
print('The shape of the targets - {}'.format(targets.shape))
# print(targets[:10])

# # Plot the training data
# targets = targets.reshape(observations,) # 1D array
# print('The shape of the targets - {}'.format(targets.shape))
# #print(targets[:10])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(xs, zs, targets)
# ax.set_xlabel('xs')
# ax.set_ylabel('zs')
# ax.set_zlabel('Targets')
# ax.view_init(azim=100)
# plt.show()

# Determine weights and biases
## inital weights and biases will be picked randonly from interval [-0.1, 0.1]
init_range = 0.1

weights = np.random.uniform(low=-init_range, high=init_range, size=(2,1))
biases = np.random.uniform(low=-init_range, high=init_range, size=(1,1))
print('The initial weights is {}'.format(weights))
print('The initial biases is {}'.format(biases))

# Set a learning Generate
learning_rate= .01

# Training
## Training the model
## Loop through the 100 epochs
loss_data = []
weights01_data = []
weights02_data = []
biases_data = []
for i in range(1000):
    # calcualte the output
    outputs = np.dot(inputs, weights) + biases
    # calcualte the deltas
    deltas = outputs - targets
    # Calculate the loss function
    loss = np.sum(deltas ** 2) / 2 / observations #L2-norm loss function
    loss_data.append(loss)
    weights01_data.append(weights[0])
    weights02_data.append(weights[1])
    biases_data.append(biases[0])
    print(loss)

    # scale the deltas
    deltas_scaled = deltas / observations
    # Update weights
    weights = weights - learning_rate*np.dot(inputs.T,deltas_scaled)
    # Update baises
    biases = biases - learning_rate * np.sum(deltas_scaled)

plt.plot(loss_data)
plt.show()
plt.plot(weights01_data)
plt.show()
plt.plot(weights02_data)
plt.show()
plt.plot(biases_data)
plt.show()
