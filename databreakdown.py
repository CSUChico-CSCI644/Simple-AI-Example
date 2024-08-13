import csv
testpercent = 0.3

rows = []
with open('spam.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        rows.append(row)

rowheader = rows[0]
rows = rows[1:]

n = len(rows)
test = int(n * testpercent)
train = n - test
train_rows = rows[:train]
test_rows = rows[train:]

test_outfile = 'spam_test.csv'
train_outfile = 'spam_train.csv'

with open(test_outfile, 'w') as test_file:
    writer = csv.writer(test_file)
    writer.writerow(rowheader)
    writer.writerows(test_rows)

with open(train_outfile, 'w') as train_file:
    writer = csv.writer(train_file)
    writer.writerow(rowheader)
    writer.writerows(train_rows)