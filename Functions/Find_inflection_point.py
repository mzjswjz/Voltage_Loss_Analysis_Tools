import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Inflection_Points:

    # Constructor function that takes in the file_id for the EQE data file
    def __init__(self, file_id):
        self.file_id = file_id

    # Function that plots the normalized EQE and its normalized numerical second derivative
    def plot_sec_diff(self):

        # Load the EQE data from the file_id and read it in as a float using numpy
        data = np.loadtxt(self.file_id, delimiter=',', skiprows=1)
        energy = data[:, 2][::-1]  # extract energy values from the data
        eqe = data[:, 3][::-1]  # extract EQE values from the data

        # Interpolate the EQE data using a cubic spline
        interpolated_eqe = interp1d(energy, eqe, kind='cubic')

        # Generate new energy values for interpolation
        new_energy = np.linspace(energy.min(), energy.max(), 2000)
        # Evaluate the interpolated function at the new energy values
        new_eqe = interpolated_eqe(new_energy)

        # Calculate the second derivative of the EQE data using numpy's diff function
        sec_diff = np.diff(np.diff(new_eqe))

        # Store the new set of energy and second derivative values in an instance variable
        self.new_set = np.vstack((new_energy[:len(sec_diff)], sec_diff))

        # Normalize the EQE data to a maximum value of 1
        norm_eqe = new_eqe / np.max(new_eqe)
        # Plot the normalized EQE data
        plt.plot(new_energy, norm_eqe)

        # Normalize the second derivative of the EQE data
        self.norm_set = self.new_set[1, :] / np.max(self.new_set[1, :])
        # Plot the normalized second derivative of the EQE data
        plt.plot(self.new_set[0, :], self.norm_set)

        # Plot a horizontal line at y = 0 for visual reference
        y = np.zeros(len(new_energy))
        plt.plot(new_energy, y, 'black', linewidth=1)

        # Add a legend to the plot with font size of 8
        plt.legend(['Normalized EQE', 'Normalized numerical second derivative of EQE'], fontsize=8)
        # Set the x-axis limits to be the minimum and maximum energy values
        plt.xlim(new_energy[0], new_energy[-1])
        # Show the plot
        plt.show(dpi=500)

    # Function that finds the inflection point of the EQE data within a specified energy range
    def find_inflection(self, rangemin, rangemax):

        # Find the index corresponding to the minimum and maximum energy values in the specified range
        indexmin = np.min(np.where(self.new_set[0, :] > rangemin))
        indexmax = np.max(np.where(self.new_set[0, :] < rangemax))

        # Iterate through the range of energy values and find the inflection point where the second derivative changes sign
        for n in range(indexmin, indexmax, 1):
            a = self.norm_set[n]
            b = self.norm_set[n + 1]
            if a > 0 and b < 0:
                inflection_pt = (self.new_set[0, n] + self.new_set[0, n + 1]) / 2
                return inflection_pt
        return None
