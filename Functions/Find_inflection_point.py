import numpy as np
import matplotlib.pyplot as plt


class Inflection_Points:

    def __init__(self, precision=0.0001):
        self.precision = precision

    def find_inflection(self, file_id):
        data = np.loadtxt(file_id, skiprows=1, dtype=float)
        energy = data[:, 2]
        eqe = data[:, 3]

        sec_diff = np.diff(np.diff(eqe))
        new_set = np.vstack((energy[:len(sec_diff)], sec_diff))

        norm_eqe = eqe / np.max(eqe)
        plt.plot(energy, norm_eqe)
        norm_set = new_set[1, :] / np.max(new_set[1, :])
        plt.plot(new_set[0, :], norm_set)
        y = np.zeros(len(energy))
        plt.plot(energy, y, 'black', linewidth=1)
        plt.legend(['Normalized EQE', 'Normalized numerical second derivative of EQE'], fontsize=8)
        plt.xlim([1.4, 3.5])
        plt.show()

        for n in range(len(new_set[1, :]) - 1, 2, -1):
            a = new_set[1, n]
            b = new_set[1, n - 1]
            c = new_set[1, n - 2]
            if a > 0 and b < 0 and c < 0 and abs(a - b) > self.precision:
                inflection_pt = (energy[n] + energy[n - 1]) / 2
                return inflection_pt
        return None
