from matplotlib import pyplot as plt
import numpy as np

from nmutil.fit import PolynomialFit

anscombes = np.array([
    [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
    [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]])

def test_poly():

    model = PolynomialFit(order=2)
    model.fit(anscombes[0], anscombes[1])
    x = np.linspace(0.0, 20.0, 1000)
    y = model.predict(x)

    plt.scatter(anscombes[0], anscombes[1], c="C0")
    plt.plot(x, y, c="C1")
    plt.show()


if __name__ == "__main__":
    test_poly()
