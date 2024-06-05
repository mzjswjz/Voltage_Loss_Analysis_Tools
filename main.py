# Import the Find_inflection_point function from the Functions module
from Functions import Find_inflection_point as FindInflect
from Functions import Calculate_Jsc
from Functions import Plot_Urbach_app_E

# Import the os module to check if a file exists
import os

def main():


####### Find Eg_PV: Uncomment the block below


    # # Ask the user to enter the path of the EQE raw file
    # q1 = 'Where the EQE raw file is located? (including file name)'
    # while True:
    #     file_id = input(q1)
    #     # Check if the file exists
    #     if os.path.isfile(file_id):
    #         break
    #     print('Invalid file path. Please enter a valid file path.')
    #
    # try:
    #     # Create an instance of the Inflection_Points class with the file ID
    #     fip = FindInflect.Inflection_Points(file_id)
    #     # Plot the second derivative of the EQE data
    #     fip.plot_sec_diff_from_SG_smooth(savefig=False, smooth_window=20)
    #
    #     # Ask the user to set a range to search for the inflection point
    #     q3 = 'Set a range to find inflection point (separated by comma)'
    #     search_range = input(q3)
    #     parts = search_range.split(",")
    #     parts = [float(part) for part in parts]
    #     # Set the range as the minimum and maximum values entered by the user
    #     rangemin = min(parts)
    #     rangemax = max(parts)
    #
    #     # Find the inflection point within the specified range
    #     inflection_pt = fip.find_inflection(rangemin, rangemax, savefig=False)
    #     print(f'Inflection point: {inflection_pt}')
    #
    # except Exception as e:
    #     # Print any errors that occur during the processing
    #     print(f'Error: {e}')



######### Calculate Jsc: uncomment the block below
    #
    # # Ask the user to enter the path of the EQE raw file
    # q1 = 'Where the EQE raw file is located? (including file name)'
    # while True:
    #     file_id = input(q1)
    #     # Check if the file exists
    #     if os.path.isfile(file_id):
    #         break
    #     print('Invalid file path. Please enter a valid file path.')
    #
    # try:
    #     # Create an instance of the Inflection_Points class with the file ID
    #     CJsc = Calculate_Jsc.Jsc(file_id)
    #     CJsc.calculate_Jsc()
    #
    # except Exception as e:
    #     # Print any errors that occur during the processing
    #     print(f'Error: {e}')



###### Find Urbach_energy at the edge and plot : Uncomment the block below

    q1 = 'Where the EQE raw file is located? (including file name)'
    while True:
        file_id = input(q1)
        # Check if the file exists
        if os.path.isfile(file_id):
            break
        print('Invalid file path. Please enter a valid file path.')

    try:
        # Create an instance of the Urbach class with the file ID
        Urbach = Plot_Urbach_app_E.Urbach_E(file_id, temperature=300)
        # Plot Urbach_energy
        Urbach.plot_urbach_energy(savefig=False, grid_handle=True)

        # Take user input for the energy range and method
        start_energy = float(input("Enter the start energy value (in eV): "))
        end_energy = float(input("Enter the end energy value (in eV): "))
        method = input("Enter the method ('parabolic' or 'plateau'): ").strip().lower()

        # Validate method input
        if method not in ['parabolic', 'plateau']:
            print("Invalid method. Please enter 'parabolic' or 'plateau'.")
            return

        # Calculate the Urbach energy edge
        urbach_energy_edge = Urbach.find_urbach_energy_at_edge(start=start_energy, end=end_energy, method=method,
                                                               plot=True, savefig=False, grid_handle=False)

        # Print the calculated Urbach energy edge
        print(f"Urbach energy at the edge ({method} method): {urbach_energy_edge:.2f} meV")



    except Exception as e:
        # Print any errors that occur during the processing
        print(f'Error: {e}')



if __name__ == '__main__':
    # Call the main function if this module is run as a script
    main()
