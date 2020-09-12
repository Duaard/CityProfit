# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from src.predict import predict
from src.gradient_descent import descent
from src.compute_cost import cost_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython

# %%
from IPython import get_ipython

# %% [markdown]
#  # City Profit
# %% [markdown]
#  First, we'll import the required libraries:
#
#  Matplotlib will be used in plotting graphs.
#
#  Pandas and Numpy will be used for data handling and data operations

# %%
get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
#  Next, we'll create our dataframe and peek at the top  5 entries in our dataset

# %%
df = pd.read_csv('data/data.txt', names=['City Population', 'Profit'])
df.head()

# %% [markdown]
#  Using matplotlib, we'll use plt.scatter to visualize each data entry as individual points.

# %%
# Read our X values and Y values
df_x, df_y = df.iloc[:, 0], df.iloc[:, 1]

# Visualize data
fig, ax = plt.subplots()
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
ax.scatter(df_x, df_y)


# %%
# Set up all variables in 1 cell
m = len(df)
X = df_x.to_numpy()
Y = df_y.to_numpy()


# Will be used for gradient descent
alpha = 0.01
iterations = 500

# %% [markdown]
#  Next, we'll add i
# n a 1's column to our X features. This 1 is needed for the X-intercept.

# %%
# Add in 1 columns for X-intercept
X_1 = np.c_[np.ones(m), X]

# %% [markdown]
#  To compute for the cost function, we created a *cost_function* in compute_cost.py
#  ```python
#  def cost_function(X: np.array, Y: np.array, theta: np.array, alpha):
#      m = len(Y)
#      h = X.dot(theta)
#      return ((h - Y) ** 2).sum() / (2 * m)
#  ```
#  We'll now call this function, to confirm JCost is working we'll call it using a initial theta=\[0, 0\]
#
#  The expected result should be 32.07

# %%
# Initial theta values
theta = np.zeros(2)
print(cost_function(X_1, Y, theta, alpha))

# %% [markdown]
#  Since our JCost is functioning correctly, we'll now proceed with Gradient Descent
#
#  Using the gradient_descent.py, we'll import the *descent* function
# ```python
#  def descent(X: np.array, Y: np.array, theta: np.array, alpha):
#      m = len(Y)
#      h = X.dot(theta)
#      theta = theta - (X.transpose().dot(h - Y) * (alpha / m))
# ```
#  We can now use this descent function to minimize our JCost for a certain number of iterations. To see this, we'll store every value of JCost in each iteration in JHistory

# %%
theta = np.zeros(2)
j_hist = []
theta_hist = [theta]
for i in range(iterations):
    j_hist.append(cost_function(X_1, Y, theta, alpha))
    theta = descent(X_1, Y, theta, alpha)
    theta_hist.append(theta)

print("Training done!")


# %%
ax.plot(X, X_1.dot(theta), 'r')
fig
# %% [markdown]
# ## Predicting Profit
# We can know predict the profit using the computed theta
# Using the helper script predict, we can input any number of City Population and get our prediction, the script automatically normalizes the value to fit the prediction
# ```python
# def predict(population, theta):
#     # Normalize given population
#     population /= 10000
#     x = np.array([1, population])
#     return x.transpose().dot(theta) * 10000
# )
# ```

# %%
# Feel free to play with this cell to input any population
population = 75000
p = predict(population, theta)
print(f'The predicted profit for City with population {population}:\n${p:.2f}')

# Plot the result in respect with linear prediction
res_fig, res_ax = plt.subplots()
x, y = population/10000, p/10000
res_ax.plot(x, y, 'bx', mew=3, ms=15)
res_ax.scatter(df_x, df_y, c='m')
x_lim = res_ax.get_xlim()
x_lim_1 = np.c_[np.ones(len(x_lim)), np.array(x_lim)]
y_lim = x_lim_1.dot(theta)
res_ax.plot(x_lim, y_lim, 'r')
