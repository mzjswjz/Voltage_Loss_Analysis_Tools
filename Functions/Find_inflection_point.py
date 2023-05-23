import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


class Inflection_Points:

    # Constructor function that takes in the file_id for the EQE data file
    def __init__(self, file_id):
        self.file_id = file_id
        # Load the EQE data from the file_id and read it in as a float using numpy
        self.EQE_data = np.loadtxt(self.file_id, delimiter=',', skiprows=1)
        self.raw_energy = self.EQE_data[:, 2][::-1]  # extract energy values from the data
        self.raw_eqe = self.EQE_data[:, 3][::-1]  # extract EQE values from the data

        # Interpolate the EQE data using a cubic spline
        interpolated_eqe = interp1d(self.raw_energy, self.raw_eqe, kind='cubic', bounds_error=False, fill_value=0.0)
        # Generate new energy values for interpolation
        new_energy = np.linspace(self.raw_energy.min(), self.raw_energy.max(), 2000)
        # Evaluate the interpolated function at the new energy values
        new_eqe = interpolated_eqe(new_energy)

        self.int_energy = new_energy
        self.int_eqe = new_eqe

    def plot_raw_norm_EQE(self):

        # Normalize the EQE data to a maximum value of 1
        norm_eqe = self.raw_eqe / np.max(self.raw_eqe)
        # Plot the normalized EQE data
        plt.plot(self.raw_energy, norm_eqe)
        # set x limit
        plt.xlim(self.raw_energy[0], self.raw_energy[-1], 0.5)
        # set x-ticks and font size
        plt.xticks(np.arange(0.7, 3.7, 0.2), fontsize=5)
        # Add a legend to the plot with font size of 8
        plt.legend(['Normalized EQE'], fontsize=8)
        # turn on grid
        plt.grid(axis='x')
        # Show the plot
        plt.show(dpi=200)

    def plot_sec_diff_from_SG_smooth(self, smooth_window=200, smooth_order=5, savefig=False):
        """
        Calculate the plot the second derivative of EQE after applying SG signal filter to remove high frequency part

        Arguments:
        smooth_window -- the window size for smoothing the EQE spectrum with the Savitzky-Golay filter (default 21)
        smooth_order -- the order of the polynomial used in the Savitzky-Golay filter (default 3)

        Returns:
        None

        """

        # Check if energy and EQE arrays have the same length
        if len(self.int_energy) != len(self.int_eqe):
            print('Error: energy and EQE arrays have different lengths.')
            return None

        # Smooth the EQE spectrum using the Savitzky-Golay filter
        eqe_smooth = savgol_filter(self.int_eqe, window_length=smooth_window, polyorder=smooth_order)
        # Normalize the EQE data to a maximum value of 1
        self.smooth_norm_eqe = eqe_smooth / np.max(abs(eqe_smooth))

        # # DEBUG USE: Plot the original and smoothed EQE spectra
        # fig_EQE = plt.figure()
        # plt.plot(self.int_energy, self.int_eqe, label='Original EQE')
        # plt.plot(self.int_energy, eqe_smooth, '--', label='Smoothed EQE')
        # plt.xlabel('Energy (eV)')
        # plt.ylabel('EQE')
        # plt.title('Comparison of smoothed and raw EQE (Smooth window = 200, order = 5)')
        # plt.legend()
        # fig_EQE.savefig('/Users/jswjzhm/Desktop/EQE_Compare.png', dpi=200)

        # Calculate the second derivative of the EQE spectrum w/o and with smoothing filter
        sec_diff_wo_filter = np.gradient(np.gradient(self.int_eqe, self.int_energy), self.int_energy)
        sec_diff = np.gradient(np.gradient(eqe_smooth, self.int_energy), self.int_energy)
        # Store the new set of energy and second derivative values in an instance variable
        self.new_set_wo_filter = np.vstack((self.int_energy[:len(sec_diff_wo_filter)], sec_diff_wo_filter))
        self.new_set = np.vstack((self.int_energy[:len(sec_diff)], sec_diff))

        # Normalize the second derivative of the EQE and smoothed EQE
        self.norm_set_wo_filter = self.new_set_wo_filter[1, :] / np.max(abs(self.new_set_wo_filter[1, :]))
        self.norm_set = self.new_set[1, :] / np.max(abs(self.new_set[1, :]))


        # Plot to check the validity of smoothed EQE
        fig_Sec_D = plt.figure()
        # Plot the normalized second derivative of the EQE data
        plt.plot(self.new_set_wo_filter[0, :], self.norm_set_wo_filter)
        plt.plot(self.new_set[0, :], self.norm_set)

        # Plot a horizontal line at y = 0 for visual reference
        y = np.zeros(len(self.int_energy))
        plt.plot(self.int_energy, y, 'black', linewidth=1)

        # Add a legend to the plot with font size of 8
        plt.legend(['Normalized numerical second derivative of EQE (w/o Savitzky-Golay filter)', 'Normalized numerical second derivative of EQE'], fontsize=8)
        # Set the x-axis limits to be the minimum and maximum energy values
        plt.xlim(self.int_energy[0], self.int_energy[-1])
        plt.xticks(np.arange(1, 3.75, 0.25))
        plt.xlabel('Energy (eV)')
        plt.ylabel('Normalized value')
        plt.grid()

        if savefig != False:
            fig_Sec_D.savefig('/Users/jswjzhm/Desktop/Second_Derivative_EQE_Compare.png', dpi=200)


        # Show the plot
        plt.show(dpi=300)

    #  Additionally fitting a Gaussian to EQE then find second derivative

    def plot_sec_diff_from_raw(self):
        # Normalize the EQE data to a maximum value of 1
        self.norm_eqe = self.int_eqe / np.max(self.int_eqe)

        # Plot a horizontal line at y = 0 for visual reference
        y = np.zeros(len(self.int_energy))

        plt.figure()
        # Plot the normalized EQE data
        plt.plot(self.int_energy, self.norm_eqe)
        plt.plot(self.new_set_wo_filter[0, :], self.norm_set_wo_filter)
        plt.plot(self.int_energy, y, 'black', linewidth=1)

        # Add a legend to the plot with font size of 8
        plt.legend(['Normalized EQE', 'Normalized numerical second derivative of EQE'], fontsize=8)
        # Set the x-axis limits to be the minimum and maximum energy values
        plt.xlim(self.int_energy[0], self.int_energy[-1])
        plt.xticks(np.arange(1, 3.5, 0.25))
        plt.grid()
        # Show the plot
        plt.show(dpi=200)

    # Function that finds the inflection point of the EQE data within a specified energy range
    def find_inflection(self, rangemin, rangemax, savefig=False):

        # Find the index corresponding to the minimum and maximum energy values in the specified range
        indexmin = np.min(np.where(self.new_set[0, :] > rangemin))
        indexmax = np.max(np.where(self.new_set[0, :] < rangemax))

        # Iterate through the range of energy values and find the inflection point where the second derivative changes sign
        for n in range(indexmin, indexmax, 1):
            a = self.norm_set[n]
            b = self.norm_set[n + 1]
            if a > 0 and b < 0:
                inflection_pt = (self.new_set[0, n] + self.new_set[0, n + 1]) / 2
                formatted_inflection_pt = '{:.3f}'.format(inflection_pt)

                # Plot inflection point
                x_inflect = np.full(100, inflection_pt)
                y_inflect = np.linspace(-0.1, 1.1, 100)

                # Plot y=0 base line to see cross point
                y = np.zeros(len(self.int_energy))

                # Plot the second derivative with Original EQE
                fig = plt.figure()
                plt.plot(self.int_energy, self.smooth_norm_eqe, label=f'EQE')
                plt.plot(x_inflect, y_inflect, '--', linewidth=1, label=f'{formatted_inflection_pt} eV')
                plt.plot(self.new_set[0, :], self.norm_set, label=f'Second derivative of the EQE')
                plt.plot(self.int_energy, y, 'black', linewidth=1)

                # Set the x-axis limits to be the minimum and maximum energy values
                # Add a legend to the plot with font size of 8
                plt.legend(fontsize=8)
                plt.xlabel('Energy (eV)')
                plt.ylabel('Normalized value')
                plt.xlim(self.int_energy[0], 2.5)
                plt.xticks(np.arange(1, 3.5, 0.25))
                plt.yticks(np.arange(-1, 1.25, 0.25))

                if savefig != False:
                    fig.savefig('/Users/jswjzhm/Desktop/EQE_and_Sec_EQE.png', dpi=200)

                plt.show(dpi=300)

                return formatted_inflection_pt
        return None

    def Debug_plot_sec_diff_from_SG_smooth(self):
        """
        Calculate the plot the second derivative of EQE after applying SG signal filter to remove high frequency part

        Arguments:
        smooth_window -- the window size for smoothing the EQE spectrum with the Savitzky-Golay filter (default 21)
        smooth_order -- the order of the polynomial used in the Savitzky-Golay filter (default 3)

        Returns:
        None

        """

        # Check if energy and EQE arrays have the same length
        if len(self.int_energy) != len(self.int_eqe):
            print('Error: energy and EQE arrays have different lengths.')
            return None

        # Define the range of smooth windows and smooth orders
        smooth_windows = range(100, 301, 100)
        smooth_orders = range(1, 6)

        # Figure 1: Varying smooth window

        # Initialize the figure and subplot
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        # Calculate the second derivative of the EQE spectrum w/o smoothing filter
        sec_diff_wo_filter = np.gradient(np.gradient(self.int_eqe, self.int_energy), self.int_energy)

        # Plot the second derivative of the EQE w/o smoothing filter
        ax1.plot(self.int_energy, sec_diff_wo_filter, label='W/O SG filter')

        # Loop through the smooth windows and plot the second derivative with the corresponding smooth window
        for smooth_window in smooth_windows:
            eqe_smooth = savgol_filter(self.int_eqe, window_length=smooth_window, polyorder=5)
            sec_diff = np.gradient(np.gradient(eqe_smooth, self.int_energy), self.int_energy)
            ax1.plot(self.int_energy, sec_diff, label=f'Window={smooth_window}')

        # Set plot properties
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Second Derivative')
        ax1.set_title('Varying Smooth Window (Order = 5)')
        ax1.legend()
        ax1.grid()
        ax1.set_xlim(1, 2.5)

        # Figure 2: Varying smooth order

        # Initialize the figure and subplot
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        # Calculate the second derivative of the EQE spectrum w/o smoothing filter
        sec_diff_wo_filter = np.gradient(np.gradient(self.int_eqe, self.int_energy), self.int_energy)

        # Plot the second derivative of the EQE w/o smoothing filter
        ax2.plot(self.int_energy, sec_diff_wo_filter, label='W/O SG filter')

        # Loop through the smooth orders and plot the second derivative with the corresponding smooth order
        for smooth_order in smooth_orders:
            eqe_smooth = savgol_filter(self.int_eqe, window_length=200, polyorder=smooth_order)
            sec_diff = np.gradient(np.gradient(eqe_smooth, self.int_energy), self.int_energy)
            ax2.plot(self.int_energy, sec_diff, label=f'Order={smooth_order}')

        # Set plot properties
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Second Derivative')
        ax2.set_title('Varying Smooth Order (Window = 200)')
        ax2.legend()
        ax2.grid()
        ax2.set_xlim(1, 2.5)

        # Save the figures on the desktop
        fig1.savefig('/Users/jswjzhm/Desktop/figure1.png', dpi=300)
        fig2.savefig('/Users/jswjzhm/Desktop/figure2.png', dpi=300)

        # Show the plots
        plt.show(dpi=300)

