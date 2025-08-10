import csv

def print_csv_header_and_count(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader, None)
        last_row = None
        row_count = 1 if first_row else 0

        for row in reader:
            last_row = row
            row_count += 1

        print("Header:", header)
        print("Total records (excluding header):", row_count)
        if first_row:
            print("First row:", first_row)
        if last_row:
            print("Last row:", last_row)

# Example usage:
csv_file = "cleaned_comments_full.csv"
print_csv_header_and_count(csv_file)