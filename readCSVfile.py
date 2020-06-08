import csv
import pandas
with open('finalProjectDataset/trainingDataset.csv', mode='r', errors='ignore') as csv_file:
    # csv_reader = csv.DictReader(csv_file)
    # line_count = 0
    # for row in csv_reader:
    #     if line_count == 0:
    #         print(f'Column names are {", ".join(row)}')
    #         line_count += 1
    #     line_count += 1
    #     print(row)
    # print(f'Processed {line_count} lines.')
    trainingdata = pandas.read_csv(csv_file, sep=';', header=None)
    # print(trainingdata)
    # print(trainingdata.filter(like="Hoax"))
#
for index, row in trainingdata.iterrows():
     print(row[1])

