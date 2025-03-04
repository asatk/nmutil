from matplotlib import pyplot as plt
import numpy as np

from nmutil.ode import Euler, EulerCromer
from nmutil.ode import Solution

if __name__ == "__main__":
    
    nstep = 1000
    dt = 0.1
    grav = np.array([0, -10])
    fns = [lambda _: grav, lambda data, _: data[1]]
    initc = np.array([[0, 0], [1, 1]])
    
    euler = Euler(fns, nstep, 2, 2, dt, initc)
    eulercromer = EulerCromer(fns, nstep, 2, 2, dt, initc)

    data_e, time_e = euler.solve().get_soln()
    data_ec, time_ec = eulercromer.solve().get_soln()

    ax = plt.subplot()
    ax.plot(data_e[:,0], data_e[:,1], label="Euler")
    ax.plot(data_ec[:,0], data_e[:,1], labe="Euler-Cromer")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory of Ball w/ Gravity")
    ax.legend()

    
