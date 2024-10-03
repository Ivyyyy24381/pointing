import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Open the CSV file
    with open(csv_file_path, 'r') as csv_file:
        # Read the CSV data
        csv_reader = csv.DictReader(csv_file)
        # Convert CSV data to a list of dictionaries
        data = list(csv_reader)
    
    # Write the JSON data to a file
    with open(json_file_path, 'w') as json_file:
        # Convert the data to JSON format and write it to the JSON file
        json.dump(data, json_file, indent=4)

# Example usage:
csv_file_path = '/Users/ivyhe/Downloads/Pointing Data Processing - Data-2.csv'  # Replace with the path to your CSV file
json_file_path = '/Users/ivyhe/Downloads/raw.json'  # Replace with the desired path for your JSON file
csv_to_json(csv_file_path, json_file_path)
