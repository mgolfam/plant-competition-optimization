import csv
from typing import List, Dict, Any

class CSVManager:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def write_headers(self, headers: List[str]) -> None:
        """Write headers to the CSV file."""
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()

    def append_row(self, row: Dict[str, Any]) -> None:
        """Append a single row of data to the CSV file."""
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writerow(row)

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all rows from the CSV file and return as a list of dictionaries."""
        with open(self.file_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            return [row for row in reader]

    def clear_file(self) -> None:
        """Clear the contents of the CSV file."""
        with open(self.file_path, mode='w', newline='') as file:
            file.truncate(0)
