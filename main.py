import numpy as np
from load_data import load_file
import plot as plt
from compute_cost import cost_function
from gradient_descent import descent
from predict import predict


def main():
    # First load our data to X and Y
    [X, Y] = load_file("data.txt")
    # Visualize data using matplotlib
    plt.plot_data(X, Y)

    # Get the number of training examples and features
    m = len(X)
    n = 1

    # Add in the ones for X intercept
    X = np.c_[np.ones(m), X]
    Y = np.array(Y)
    # Initalize fitting parameters
    theta = np.zeros(n + 1).transpose()
    # Set inital values for alpha and num of iterations
    alpha = 0.01
    iterations = 500

    # Get inital J cost
    J = cost_function(X, Y, theta, alpha)
    print(f"Inital cost function is: {J}")

    # Record past J costs
    J_hist = []

    # Perform gradient descent
    for i in range(iterations):
        J_hist.append(cost_function(X, Y, theta, alpha))
        theta = descent(X, Y, theta, alpha)

    # Predict profits for city with 35000 population
    prof = predict(35000, theta)
    print(f'Profit for City with Population of 35,000: ${prof * 10000}')

    # Draw the linear fit
    plt.plot_line(X[:, 1], X.dot(theta))
    plt.plot_show()


if __name__ == "__main__":
    # Call main function
    main()
