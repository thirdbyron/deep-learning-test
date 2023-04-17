# %%
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
import numpy as np

tf.random.set_seed(1)

model = Sequential([
    Dense(1, input_shape=(1,), activation='linear')
])

X = np.array([[1], [3], [2], [10], [4], [7], [8]])
y = np.array([[3, 9, 6, 30, 12, 21, 24]]).T


# %%
w0, w1 = model.get_weights()

print(w0, w1)

model.compile(optimizer='sgd', loss='mse', metrics='mae')

model.fit(X, y, epochs=100)

# %%
user_inp1, user_inp2 = 5, -9
print(f"Проверка на новых данных: {user_inp1} {user_inp2}")
print("Предсказание нейронной сети: ")
print(model.predict(np.array([[user_inp1], [user_inp2]])))


