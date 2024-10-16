import tika
from tika import parser

class PDFTextExtractor:
    """Extracts text from PDF files by page number."""

    def __init__(self):
        tika.initVM()

    def extract_text(self, file_path):
        """Extracts text from a specified page of a PDF file.

        Args:
            file_path: The path to the PDF file.
            page_number: The page number to extract text from (starting from 1).

        Returns:
            The extracted text from the specified page.
        """

        parsed_data = parser.from_file(file_path)
        text_content = parsed_data['content']
        metadata = parsed_data['metadata']
        num_pages = metadata.get('xmp:numberOfPages', metadata.get('xmpTPg:NPages' , 0 ) )
        return text_content , num_pages

class FileProcessor:
    """Processes files, extracting file name and sending PDFs to PDFTextExtractor."""

    def __init__(self):
        self.pdf_extractor = PDFTextExtractor()

    def process_file(self, file_path):
        """Processes a file, extracting file name and sending PDFs to PDFTextExtractor.

        Args:
            file_path: The path to the file.
        """

        file_name = file_path.split('/')[-1]  # Extract file name from path
        file_extension = file_name.split('.')[-1]  # Extract file extension

        if file_extension == "pdf":
            extracted_text , number_of_pages = self.pdf_extractor.extract_text(file_path)  # Extract text from page 1
            print(f"Extracted text from {file_name}: {number_of_pages}")
            return extracted_text , number_of_pages
        else:
            print(f"{file_name} is not a PDF file.")
            return 0 , None

# Example usage
if __name__ == "__main__":
    processor = FileProcessor()
    file_path = "dictionary.pdf"  # Replace with the actual file path
    processor.process_file(file_path)
