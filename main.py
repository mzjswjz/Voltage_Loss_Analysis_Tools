# Import the Find_inflection_point function from the Functions module
from Functions import Find_inflection_point as FindInflect
from Functions import Calculate_Jsc

# Import the os module to check if a file exists
import os

def main():


    # Find Eg_PV: Uncomment the block below
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
    #     fip.plot_sec_diff_from_SG_smooth(savefig=False, smooth_window=10)
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



    # Calculate Jsc: uncomment the block below

    # Ask the user to enter the path of the EQE raw file
    q1 = 'Where the EQE raw file is located? (including file name)'
    while True:
        file_id = input(q1)
        # Check if the file exists
        if os.path.isfile(file_id):
            break
        print('Invalid file path. Please enter a valid file path.')

    try:
        # Create an instance of the Inflection_Points class with the file ID
        CJsc = Calculate_Jsc.Jsc(file_id)
        CJsc.calculate_Jsc()

    except Exception as e:
        # Print any errors that occur during the processing
        print(f'Error: {e}')



if __name__ == '__main__':
    # Call the main function if this module is run as a script
    main()
