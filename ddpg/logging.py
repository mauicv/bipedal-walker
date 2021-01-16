import csv


class Logging:
    def __init__(self, headers):
        self.headers = headers
        with open('save/logs', 'w') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(headers)

    def log(self, row):
        print('')
        for item, header in zip(row, self.headers):
            print(header, ': ', item)

        with open('save/logs', 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(row)
