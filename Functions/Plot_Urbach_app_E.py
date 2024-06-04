import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Urbach_E:

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

    def plot_Urbacch_app_E(self, savefig=False, Temperature = 300):
        k_B = 8.617333262145e-5
        kT = k_B * Temperature * 1000
        # Calculate the natural logarithm of EQE
        ln_eqe = np.log(self.raw_eqe)

        # Calculate the derivative of ln(EQE) with respect to energy
        dlnEQE_dE = np.gradient(ln_eqe, self.raw_energy)

        # Calculate Urbach energy in meV
        self.urbach_energy = (dlnEQE_dE ** -1)*1000

        # Plotting the Urbach energy

        fig = plt.figure(figsize=(10, 6))
        plt.xlim(1, 2)
        plt.ylim(0, 100)
        plt.plot(self.raw_energy, self.urbach_energy, label='Urbach Energy')
        plt.axhline(y=kT, color='grey', linestyle='--', label=f'$kT$ = {kT:.2f} meV')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Urbach Energy (meV)')
        #plt.title('Urbach Energy vs. Energy')
        plt.legend()
        plt.grid(True)

        if savefig != False:
            fig.savefig('/Users/jswjzhm/Desktop/Urbach_Energy.png', dpi=200)

        plt.show()

