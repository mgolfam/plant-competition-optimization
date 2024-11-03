import os

class FileManager:
    def __init__(self, base_path="data"):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def get_full_path(self, filename):
        return os.path.join(self.base_path, filename)

    def write_text(self, filename, content):
        """Write text content to a file."""
        try:
            full_path = self.get_full_path(filename)
            with open(full_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"File '{filename}' written successfully.")
        except Exception as e:
            print(f"Error writing to file '{filename}': {e}")

    def append_text(self, filename, content):
        """Append text content to a file."""
        try:
            full_path = self.get_full_path(filename)
            with open(full_path, 'a', encoding='utf-8') as file:
                file.write(content)
            print(f"Content appended to file '{filename}' successfully.")
        except Exception as e:
            print(f"Error appending to file '{filename}': {e}")

    def read_text(self, filename):
        """Read and return text content from a file."""
        try:
            full_path = self.get_full_path(filename)
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
            return None

    def file_exists(self, filename):
        """Check if a file exists."""
        full_path = self.get_full_path(filename)
        return os.path.exists(full_path)

    def delete_file(self, filename):
        """Delete a file."""
        try:
            full_path = self.get_full_path(filename)
            if os.path.exists(full_path):
                os.remove(full_path)
                print(f"File '{filename}' deleted successfully.")
            else:
                print(f"File '{filename}' does not exist.")
        except Exception as e:
            print(f"Error deleting file '{filename}': {e}")

    def list_files(self):
        """List all files in the base directory."""
        try:
            files = os.listdir(self.base_path)
            return files
        except Exception as e:
            print(f"Error listing files in '{self.base_path}': {e}")
            return []

# Example Usage
if __name__ == "__main__":
    fm = FileManager()
    fm.write_text("example.txt", "This is an example content.")
    print(fm.read_text("example.txt"))
    fm.append_text("example.txt", "\nAppended content.")
    print(fm.read_text("example.txt"))
    print("Files in directory:", fm.list_files())
    fm.delete_file("example.txt")
