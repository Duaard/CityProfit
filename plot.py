import matplotlib.pyplot as plt


def plot_data(X, Y):
    plt.scatter(X, Y)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.draw()


def plot_line(X, Y):
    plt.plot(X, Y)
    plt.draw()


def plot_show():
    plt.show()
