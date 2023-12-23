import csv


def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        data = []
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            pages = row
            data.append(pages)

        return data
