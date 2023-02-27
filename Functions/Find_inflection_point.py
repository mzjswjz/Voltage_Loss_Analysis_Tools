import numpy as np
import matplotlib.pyplot as plt


class Inflection_Points:

    def __init__(self, precision=0.0001):
        self.precision = precision

    def find_inflection(self, file_id):

        data = np.loadtxt(file_id, delimiter = ',', skiprows=1) #load data from file_id and read in ad float
        energy = data[:, 2]
        eqe = data[:, 3]

        sec_diff = np.diff(np.diff(eqe))
        new_set = np.vstack((energy[:len(sec_diff)], sec_diff))

        norm_eqe = eqe / np.max(eqe) # normalize EQE maximum value to 1
        plt.plot(energy, norm_eqe)  #plot normalised EQE

        norm_set = new_set[1, :] / np.max(new_set[1, :]) #normalize the second derivative of EQE
        plt.plot(new_set[0, :], norm_set) #plot normalised second derivatiev of EQE

        y = np.zeros(len(energy)) #creat y = 0 points at each energy value
        plt.plot(energy, y, 'black', linewidth=1) #plot y = 0

        plt.legend(['Normalized EQE', 'Normalized numerical second derivative of EQE'], fontsize=8) #define legends
        plt.xlim([1.4, 3.5])
        plt.show() #show the plotdsd

        for n in range(len(new_set[1, :]) - 1, 2, -1):
            a = new_set[1, n]
            b = new_set[1, n - 1]
            c = new_set[1, n - 2]
            if a > 0 and b < 0 and c < 0 and abs(a - b) > self.precision:
                inflection_pt = (energy[n] + energy[n - 1]) / 2
                return inflection_pt
        return None
