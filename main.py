from Functions import Find_inflection_point as FindInflect
import os

def main():
    q1 = 'Where the EQE raw file is located? (including .txt)'
 #   q2 = 'What is the precision you set to find inflection point? '

 #   precision = float(input(q2))
    while True:
        file_id = input(q1)
        if os.path.isfile(file_id):
            break
        print('Invalid file path. Please enter a valid file path.')

    try:
        fip = FindInflect.Inflection_Points(file_id)
        fip.plot_sec_diff()

        q3 = 'Set a range to find inflection point (separated by comma)'
        search_range = input(q3)
        parts = search_range.split(",")
        parts = [float(part) for part in parts]
        rangemin = min(parts)
        rangemax = max(parts)

        inflection_pt = fip.find_inflection(rangemin, rangemax)
        print(f'Inflection point: {inflection_pt}')


    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()





