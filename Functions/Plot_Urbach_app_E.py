import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
import os


class Urbach_E:
    def __init__(self, file_id, temperature=300):
        self.temperature = temperature
        self.k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
        self.kT = self.k_B * self.temperature * 1000  # kT in meV
        self.file_id = file_id

        # Process the file path to extract the folder path and file name
        self.folder_path, self.EQE_file_name = self._process_file_path(file_id)
        # Load the EQE data from the file
        self.raw_energy, self.raw_eqe = self._load_eqe_data(file_id)
        # Interpolate the EQE data for smoother curve
        self.int_energy, self.int_eqe = self._interpolate_eqe(self.raw_energy, self.raw_eqe)

        # Calculate the Urbach energy from the raw EQE data
        self.urbach_energy = self._calculate_urbach_energy(self.raw_energy, self.raw_eqe)

    def _process_file_path(self, file_id):
        """
        Process the file path to extract the folder path and file name without extension.
        """
        folder_path = os.path.dirname(file_id)
        base_name = os.path.basename(file_id)
        file_name, _ = os.path.splitext(base_name)
        return folder_path, file_name

    def _load_eqe_data(self, file_id):
        """
        Load the EQE data from a file.
        Assumes the file is in CSV format with energy values in the third column and EQE values in the fourth column.
        """
        EQE_data = np.loadtxt(file_id, delimiter=',', skiprows=1)
        energy = EQE_data[:, 2][::-1]  # extract and reverse energy values
        eqe = EQE_data[:, 3][::-1]  # extract and reverse EQE values
        return energy, eqe

    def _interpolate_eqe(self, energy, eqe):
        """
        Interpolate the EQE data using a cubic spline for smoother plotting.
        """
        interpolated_eqe = interp1d(energy, eqe, kind='cubic', bounds_error=False, fill_value=0.0)
        new_energy = np.linspace(energy.min(), energy.max(), 2000)  # generate new energy values for interpolation
        new_eqe = interpolated_eqe(new_energy)  # evaluate the interpolated function
        return new_energy, new_eqe

    def _calculate_urbach_energy(self, energy, eqe):
        """
        Calculate the Urbach energy from the natural logarithm of EQE and its derivative.
        """
        ln_eqe = np.log(eqe)
        dlnEQE_dE = np.gradient(ln_eqe, energy)  # calculate the derivative of ln(EQE) with respect to energy
        urbach_energy = np.abs(dlnEQE_dE) ** -1 * 1000  # Urbach energy in meV
        return urbach_energy

    def plot_urbach_energy(self, savefig=False, grid_handle=True, highlight_energy=None, highlight_value=None):
        """
        Plot the Urbach energy and optionally save the plot to a file.
        """

        fontsize = 18
        with mpl.rc_context({'axes.linewidth': 2}):

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.tick_params(labelsize=fontsize, direction='in', axis='both', which='major', length=6, width=1, top=True,
                          right=False, left=True)
            ax.tick_params(labelsize=fontsize, direction='in', axis='both', which='minor', length=3, width=1, left=True,
                          bottom=True, top=True)
            ax.minorticks_on()
            ax.set_xlim(1.1, 2)  # set the x-axis limits
            ax.set_ylim(0, 100)  # set the y-axis limits
            ax.set_xticks(np.arange(1.2, 2.1, 0.1))
            ax.plot(self.raw_energy, self.urbach_energy, label=r'$E^{app}_U$')  # plot Urbach energy
            ax.axhline(y=self.kT, color='grey', linestyle='--', label=r'$kT_{300K}$' f'= {self.kT:.2f} meV')  # plot kT line

            if highlight_energy is not None and highlight_value is not None:
                # Highlight the specified point
                ax.text(highlight_energy+0.05, highlight_value-4, f'{highlight_value:.2f} meV', color='#990000', ha='right')


            ax.set_xlabel('Energy (eV)', fontsize=fontsize)
            ax.set_ylabel('Apparent Urbach Energy $E^{app}_U$ (meV)', fontsize=fontsize)
            ax.legend(fontsize=fontsize-2, frameon=False, loc='lower right')
            ax.grid(grid_handle)

        if savefig:
            # Ensure the save directory exists
            save_dir = os.path.join(self.folder_path, 'Urbach_Energy')
            os.makedirs(save_dir, exist_ok=True)
            # Save the plot to the specified path
            save_path = os.path.join(save_dir, f'Urbach_E_vs_E_{self.EQE_file_name}.png')
            fig.savefig(save_path, dpi=200)
            print(f"Plot saved to {save_path}")

        plt.show()

    def find_urbach_energy_at_edge(self, start, end, method='parabolic', plot=True, savefig=False, grid_handle=False):
        """
        Find the Urbach energy at the edge of the EQE low energy tail within a specified range.
        The method can be 'parabolic' or 'plateau'.
        """
        # Filter the data within the specified range
        mask = (self.raw_energy >= start) & (self.raw_energy <= end)
        energy_range = self.raw_energy[mask]
        urbach_energy_range = self.urbach_energy[mask]

        if method == 'parabolic':
            # Find the lowest point in the parabolic-like curve
            min_index = np.argmin(urbach_energy_range)
            # Take the average value with the 2 nearest points
            nearest_points = urbach_energy_range[max(0, min_index - 1):min_index + 2]
            urbach_energy_edge = np.mean(nearest_points)
            label_energy = energy_range[min_index]

            if plot:
                # Reuse the plot_urbach_energy method to highlight the calculated point
                self.plot_urbach_energy(savefig, grid_handle, highlight_energy=label_energy,
                                        highlight_value=urbach_energy_edge)


        elif method == 'plateau':
            # Find the lowest point in the defined region
            min_index = np.argmin(urbach_energy_range)
            lowest_value = urbach_energy_range[min_index]

            # Initialize the region with the lowest point
            plateau_region = [lowest_value]

            # Check nearest points to see if they have similar values
            i = min_index - 1
            while i >= 0 and np.abs(urbach_energy_range[i] - lowest_value) < 0.05 * lowest_value:
                plateau_region.append(urbach_energy_range[i])
                i -= 1
            i = min_index + 1
            while i < len(urbach_energy_range) and np.abs(urbach_energy_range[i] - lowest_value) < 0.05 * lowest_value:
                plateau_region.append(urbach_energy_range[i])
                i += 1
            # Calculate the average value in this plateau region
            urbach_energy_edge = np.mean(plateau_region)
            print(plateau_region)
            label_energy = energy_range[min_index]

            if plot:
                # Reuse the plot_urbach_energy method to highlight the calculated point
                self.plot_urbach_energy(savefig, grid_handle, highlight_energy=label_energy,
                                        highlight_value=urbach_energy_edge)
        else:
            raise ValueError("Method must be either 'parabolic' or 'plateau'.")

        return urbach_energy_edge
