# %%
import numpy as np

lights = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 1],
                  [0, 1, 1], [0, 0, 1], [0, 1, 0]])
weights = np.array([.5, .8, 1])

goals = np.array([1, 1, 1, 0, 0, 0])

alpha = 0.01

for i in range(80):
    sum_error = 0
    for j in range(len(goals)):
        input = lights[j]
        goal = goals[j]

        pred = input.dot(weights)

        error = (goal - pred) ** 2
        sum_error += error

        delta = pred - goal

        weights = weights - (alpha * (input * delta))

testInput = np.array([0, 0.4, 1])

res = testInput.dot(weights)


# %%

if res > 0.5:
    print('ИДТИ!')
else:
    print('СТОЯТЬ!')


# %%
