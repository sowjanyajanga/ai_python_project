# Importing the libraries
import numpy as np

# Setting conversion rates and the number of samples
conversionRates = [0.15, 0.04, 0.13, 0.11, 0.05]
N = 1500
d = len(conversionRates)

# Creating the dataset
X = np.zeros((N, d))
print('X.shape ' + str(X.shape))
# X has N( number of attempts) rows
# d columns where each column is for each slot machine
# X[1][2] Attempt is 2
#         Slot machine 3
#         tells what are the chances the third slot machine winning for the second attempt

for i in range(N):
    for j in range(d):
        if np.random.rand() < conversionRates[j]:
            X[i][j] = 1

# Making arrays to count our losses and wins on each slot machine
# ex: nPosReward[0]= 400 means that there were 400 wins when played on slot machine 1
nPosReward = np.zeros(d)
nNegReward = np.zeros(d)

# Taking our best slot machine through beta distribution and updating its losses and wins
for i in range(N):
    selected = 0
    maxRandom = 0
    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j] + 1, nNegReward[j] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j
        if X[i][selected] == 1:
            nPosReward[selected] += 1
        else:
            nNegReward[selected] += 1

# Showing which slot machine is considered the best
nSelected = nPosReward + nNegReward
for i in range(d):
    print('Machine number ' + str(i + 1) + ' was selected ' + str(nSelected[i]) + ' times')

print('Conclusion: Best machine is machine number ' + str(np.argmax(nSelected) + 1))