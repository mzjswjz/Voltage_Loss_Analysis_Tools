import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Inflection_Points:

    def __init__(self, file_id):
        self.file_id = file_id
        #self.rangemin = rangemin
        #self.rangemax = rangemax

    def plot_sec_diff(self):

        data = np.loadtxt(self.file_id, delimiter=',', skiprows=1)  # load data from file_id and read in ad float
        energy = data[:, 2][::-1]
        eqe = data[:, 3][::-1]

        interpolated_eqe = interp1d(energy, eqe, kind='cubic')

        # Generate new energy values for interpolation
        new_energy = np.linspace(energy.min(), energy.max(), 2000)
        # Evaluate interpolated function at new energy values
        new_eqe = interpolated_eqe(new_energy)

        sec_diff = np.diff(np.diff(new_eqe))
        self.new_set = np.vstack((new_energy[:len(sec_diff)], sec_diff))

        norm_eqe = new_eqe / np.max(new_eqe)  # normalize EQE maximum value to 1
        plt.plot(new_energy, norm_eqe)  # plot normalised EQE

        self.norm_set = self.new_set[1, :] / np.max(self.new_set[1, :])  # normalize the second derivative of EQE
        plt.plot(self.new_set[0, :], self.norm_set)  # plot normalised second derivatiev of EQE

        y = np.zeros(len(new_energy))  # creat y = 0 points at each energy value
        plt.plot(new_energy, y, 'black', linewidth=1)  # plot y = 0

        plt.legend(['Normalized EQE', 'Normalized numerical second derivative of EQE'], fontsize=8)  # define legends
        plt.xlim(new_energy[0], new_energy[-1])
        plt.show()  # show the plotd


    def find_inflection(self,rangemin, rangemax):

        indexmin = np.min(np.where(self.new_set[0,:] > rangemin))
        indexmax = np.max(np.where(self.new_set[0, :] < rangemax))

        for n in range(indexmin,indexmax, 1):
            a = self.norm_set[n]
            b = self.norm_set[n + 1]
            if a > 0 and b < 0 :
                inflection_pt = (self.new_set[0, n] + self.new_set[0, n + 1]) / 2
                return inflection_pt
        return None
