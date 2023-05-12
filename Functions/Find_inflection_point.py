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

    def interpolate_eqe(self, num_points=2000):
        energy = self.raw_energy
        eqe = self.raw_eqe

        # Interpolate the EQE data using a cubic spline
        interpolated_eqe = interp1d(energy, eqe, kind='cubic')

        # Generate new energy values for interpolation
        new_energy = np.linspace(energy.min(), energy.max(), num_points)
        # Evaluate the interpolated function at the new energy values
        new_eqe = interpolated_eqe(new_energy)

        self.int_energy = new_energy
        self.int_eqe = new_eqe


    def plot_EQE(self):

        # Normalize the EQE data to a maximum value of 1
        norm_eqe = self.raw_eqe / np.max(self.raw_eqe)
        # Plot the normalized EQE data
        plt.plot(self.raw_energy, norm_eqe)
        # set x limit
        plt.xlim(self.raw_energy[0], self.raw_energy[-1], 0.5)
        #set x-ticks and font size
        plt.xticks(np.arange(0.7, 3.7, 0.1), fontsize=5)
        # Add a legend to the plot with font size of 8
        plt.legend(['Normalized EQE'], fontsize=8)
        # turn on grid
        plt.grid(axis='x')
        # Show the plot
        plt.show(dpi=200)

    def calculate_bandgap_energy(self, smooth_window=21, smooth_order=3):
        """
        Calculates the bandgap energy of a solar cell from its EQE spectrum using the zero crossing
        of the second derivative method.

        Arguments:
        energy -- an array of photon energies in eV
        eqe -- an array of EQE values
        smooth_window -- the window size for smoothing the EQE spectrum with the Savitzky-Golay filter (default 21)
        smooth_order -- the order of the polynomial used in the Savitzky-Golay filter (default 3)

        Returns:
        A float representing the bandgap energy in eV, or None if a zero crossing of the second derivative
        could not be found.
        """
        # Check if energy and EQE arrays have the same length
        if len(self.raw_energy) != len(self.raw_eqe):
            print('Error: energy and EQE arrays have different lengths.')
            return None

        # Smooth the EQE spectrum using the Savitzky-Golay filter
        eqe_smooth = savgol_filter(self.raw_eqe, window_length=smooth_window, polyorder=smooth_order)

        # Plot the original and smoothed EQE spectra
        plt.figure()
        plt.plot(self.raw_energy, self.raw_eqe, label='Original EQE')
        plt.plot(self.raw_energy, eqe_smooth, label='Smoothed EQE')
        plt.xlabel('Photon energy (eV)')
        plt.ylabel('EQE')
        plt.legend()

        # Check for NaN values in the smoothed EQE spectrum
        if np.isnan(eqe_smooth).any():
            # Replace NaN values with the mean of the surrounding non-NaN values
            eqe_smooth[np.isnan(eqe_smooth)] = np.mean(eqe_smooth[np.logical_not(np.isnan(eqe_smooth))])


        # Calculate the second derivative of the EQE spectrum
        eqe_second_deriv = np.gradient(np.gradient(eqe_smooth, self.raw_energy), self.raw_energy)

        # Find the zero crossing of the second derivative
        zero_crossings = np.where(np.diff(np.sign(eqe_second_deriv)))[0]

        # Check if there is a zero crossing
        if len(zero_crossings) > 0:
            zero_crossing_energy = self.raw_energy[zero_crossings[0]]


            # Plot the second derivative of the EQE spectrum with the normalized EQE spectrum
            eqe_norm = eqe_smooth / np.max(eqe_smooth)
            eqe_second_deriv_norm = eqe_second_deriv / np.max(np.abs(eqe_second_deriv))
            plt.figure()
            plt.plot(self.raw_energy, eqe_norm, label='Normalized EQE')
            plt.plot(self.raw_energy, eqe_second_deriv_norm, label='Normalized Second Derivative EQE')
            plt.xlabel('Photon energy (eV)')
            plt.ylabel('Normalized EQE/Second Derivative EQE')
            plt.legend()

            return zero_crossing_energy
        else:
            print('Could not find a zero crossing of the second derivative.')
            return None



    def gaussian(self, energy, a, b, c):
        return a * np.exp(-(energy - b)**2 / (2 * c**2))

    def Find_best_Gaussian_Fit(self, rangemin, rangemax, fluctuation=0.2):

        # Interpolate EQE data to 1000 points
        energy_interp = np.linspace(self.raw_energy.min(), self.raw_energy.max(), num=2000)
        eqe_interp = interp1d(self.raw_energy, self.raw_eqe, kind='linear')(energy_interp)

        # Define energy range
        start_min = rangemin - fluctuation / 2
        start_max = rangemin + fluctuation / 2
        end_min = rangemax - fluctuation / 2
        end_max = rangemax + fluctuation / 2

        # Calculate Gaussian fits with different start and end values
        fits = []
        r_squareds = []
        energy_for_fits = []
        raw_eqe_for_fits = []

        for start, end in [(start_min, end_min), (start_min, end_max), (start_max, end_min), (start_max, end_max)]:
            # Select energy range
            mask = (energy_interp >= start) & (energy_interp <= end)
            energy = energy_interp[mask]
            eqe = eqe_interp[mask]

            # Initial guess for Gaussian parameters
            a0 = np.max(eqe)
            b0 = energy[np.argmax(eqe)]
            c0 = (end - start) / 10
            p0 = [a0, b0, c0]
            try:
                # Fit Gaussian
                popt, pcov = curve_fit(self.gaussian, energy, eqe, p0=p0, maxfev=20000)
                fit_eqe_coeffs = popt

                # Calculate R-squared of the fit
                fit_eqe = self.gaussian(energy_interp, *popt)
                residuals = eqe_interp - fit_eqe
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((eqe_interp - np.mean(eqe_interp)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                fits.append(fit_eqe_coeffs)
                r_squareds.append(r_squared)
                energy_for_fits.append(energy_interp)
                raw_eqe_for_fits.append(eqe_interp)


            except RuntimeError:
                # If the fit fails, skip this iteration
                continue

        if not r_squareds:
            # If all fits fail, raise an exception
            raise RuntimeError('Unable to fit a Gaussian to the given range.')

        # Choose fit with highest R-squared
        best_fit_coeffs = fits[np.argmax(r_squareds)]
        best_r_squared = r_squareds[np.argmax(r_squareds)]
        best_fit_energy = energy_for_fits[np.argmax(r_squareds)]
        best_fit_eqe = raw_eqe_for_fits[np.argmax(r_squareds)]

        # Calculate fitted EQE based on Gaussian coefficients for best fit
        fitted_energy = np.linspace(rangemin - 0.5, rangemax + 0.5, num=1000)
        fitted_eqe = self.gaussian(fitted_energy, *best_fit_coeffs)

        # Plot results
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(energy_interp, eqe_interp, label='EQE')
        ax.plot(fitted_energy, fitted_eqe, label='Gaussian fit')
        ax.legend()
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('EQE (a.u.)')



        # Function that plots the normalized EQE and its normalized numerical second derivative
    def plot_sec_diff_from_raw(self):

        energy = self.raw_energy  # extract energy values from the data
        eqe = self.raw_eqe  # extract EQE values from the data

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
        plt.xlim(new_energy[0], new_energy[-1], 0.25)
        plt.grid()
        # Show the plot
        plt.show(dpi=200)



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
