from Functions import Find_inflection_point as FindInflect
import os

def main():
    q1 = 'Where the EQE raw file is located? (including .txt)'
    q2 = 'What is the precision you set to find inflection point? '

    file_id = input(q1)
    precision = float(input(q2))



    try:
        assert os.path.isfile(file_id), 'Invalid file path'
        fip = FindInflect.Inflection_Points(precision)
        inflection_pt = fip.find_inflection(file_id)
        print(f'Inflection point: {inflection_pt}')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()





