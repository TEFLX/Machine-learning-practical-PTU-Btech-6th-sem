import csv

def write_student_records_to_csv(file_name, records):
    with open(file_name, mode='w', newline='') as csv_file:
        fieldnames = ['Name', 'Maths', 'Science', 'English']  # Define field names
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()  # Write header row

        for record in records:
            writer.writerow(record)

if __name__ == "__main__":
    # Example student records
    student_records = [
        {'Name': 'John', 'Maths': 85, 'Science': 90, 'English': 88},
        {'Name': 'Alice', 'Maths': 78, 'Science': 92, 'English': 85},
        {'Name': 'Bob', 'Maths': 90, 'Science': 85, 'English': 82}
    ]

    # Name of the CSV file to be created
    csv_file_name = "student_records.csv"

    # Write student records to CSV file
    write_student_records_to_csv(csv_file_name, student_records)

    print("Student records written to CSV file successfully.")
