import numpy as np
from scipy.interpolate import interp1d


class Jsc:
    """
    This is a class that computes the short-circuit current density (Jsc) of a photovoltaic device from its external
    quantum efficiency (EQE) data and the spectral irradiance of the sun.
    """
    def __init__(self, file_id_EQE):
        """
        Constructor function that takes in the file_id for the EQE data file

        Args:
            file_id_EQE (str): the file ID of the EQE data file.
        """
        self.file_id_EQE = file_id_EQE
        # Load the EQE data from the file_id and read it in as a float using numpy
        self.EQE_data = np.loadtxt(self.file_id_EQE, delimiter=',', skiprows=1)
        self.raw_wavelength = self.EQE_data[:, 1]  # extract energy values from the data
        self.raw_eqe = self.EQE_data[:, 3]  # extract EQE values from the data
        self.EQE = interp1d(self.raw_wavelength, self.raw_eqe, kind='cubic', bounds_error=False, fill_value=0.0)

        # Read AM1.5G
        spectrum = np.loadtxt('AM15G.txt', delimiter='\t', skiprows=1)
        self.AM15G_wavelength = spectrum[:, 0] # nm
        self.AM15G_irradiance = spectrum[:, 1] # W*m^-2*nm^-1

    def calculate_Jsc(self):
        """
        This method calculates the Jsc of the photovoltaic device.

        Returns:
            jsc (float): the Jsc of the photovoltaic device, in mA/cm^2.
        """
        # Interpolate EQE to 2000 points
        interp_wavelength = np.linspace(self.raw_wavelength.min(), self.raw_wavelength.max(), 2000)
        EQE_interp = self.EQE(interp_wavelength)

        # Interpolate AM1.5G to match EQE wavelength range
        AM15G_interp = interp1d(self.AM15G_wavelength, self.AM15G_irradiance, kind='linear',
                                            bounds_error=False, fill_value=0.0)
        AM15G_interp = AM15G_interp(interp_wavelength)

        # Calculate Jsc
        e = 1.60217662e-19  # C
        h = 6.62607004e-34  # J*s
        c = 299792458  # m/s
        qe = EQE_interp  # unit of 1
        wl = interp_wavelength  # nm
        spectrum = AM15G_interp  # W/m^2/nm

        # calculate photon flux density
        flux_density = spectrum / (h * c / wl)

        # calculate current density
        j_density = e * np.trapz(flux_density * qe, wl)

        # convert unit to mA/cm^2
        jsc = j_density*1e-10

        # print Jsc to console
        print(f"Jsc: {jsc:.3f} mA/cm^2")

        return jsc





