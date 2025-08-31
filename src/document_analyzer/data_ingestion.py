# Standard Library Imports
import os
import uuid
from datetime import datetime, timezone

# Third-Party Imports
import fitz  # PyMuPDF for PDF reading

# Local Application Imports
from logger.custom_logger import CustomLogger
from exception.custom_exception import AllAboutDocumentsException


class DocumentHandler:
    """
    Handles PDF-related operations including saving and reading PDF content.

    - Initializes a unique session directory per instance to store uploaded files.
    - Validates and saves PDF files to disk.
    - Extracts text from PDF files page-by-page using PyMuPDF.
    """

    def __init__(self, file_location=None, session_id=None) -> None:
        """
        Initialize DocumentHandler by setting up file paths and session directory.

        Args:
            file_location (str, optional): Path to store session files. Uses default if not provided.
            session_id (str, optional): Custom session ID. Generates one if not supplied.

        Raises:
            AllAboutDocumentsException: If initialization fails.
        """
        try:
            self.logger = CustomLogger().get_logger(__name__)

            # Set base file storage directory
            self.file_location = file_location or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "data", "document_analyzer"),
            )

            # Generate a new or use provided session ID
            self.session_id = self.generate_session_id(session_id)

            # Create a session-specific directory for storing PDF files
            self.session_directory = os.path.join(self.file_location, self.session_id)
            os.makedirs(self.session_directory, exist_ok=True)

            # Log successful initialization
            self.logger.info(
                "DocumentHandler initialized successfully.",
                session_id=self.session_id,
                session_directory=self.session_directory,
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize DocumentHandler: {e}.")
            raise AllAboutDocumentsException(
                "Failed to initialize DocumentHandler.", e
            ) from e

    @staticmethod
    def generate_session_id(session_id=None):
        """
        Generate a unique session ID using timestamp and a short UUID.

        Args:
            session_id (str, optional): Provided session ID.

        Returns:
            str: Unique or passed session ID.
        """
        if session_id:
            return session_id

        # Uncomment below if user/hostname-based session IDs are needed
        # hostname = socket.gethostname()
        # user_identifier = os.getenv("USER", "anonymous")
        # timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        # unique_part = uuid.uuid4().hex[:8]
        # return f"session_{user_identifier}_{hostname}_{timestamp}_{unique_part}"
        # return f"session_{user_identifier}_{timestamp}_{unique_part}"

        return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"

    def save_pdf(self, file_to_be_analyzed):
        """
        Save a PDF file to the current session directory after validating the file type.

        Args:
            file_to_be_analyzed: A file-like object with `.name` and `.getbuffer()` attributes.

        Returns:
            str: Absolute path to the saved file.

        Raises:
            AllAboutDocumentsException: If file is not a PDF or if save fails.
        """
        try:
            pdf_file_name = os.path.basename(file_to_be_analyzed.name)

            # Validate file extension
            if not file_to_be_analyzed.name.lower().endswith(".pdf"):
                raise AllAboutDocumentsException(
                    "Invalid file type: only PDF files are supported."
                )

            # Construct full save path
            the_pdf_is_saved_here = os.path.join(self.session_directory, pdf_file_name)

            # Save file to disk
            with open(the_pdf_is_saved_here, "wb") as pdf_file:
                pdf_file.write(file_to_be_analyzed.getbuffer())

            self.logger.info(
                "PDF file saved successfully.",
                saved_file=pdf_file_name,
                saved_path=the_pdf_is_saved_here,
                session_for_analysis=self.session_id,
            )

            return the_pdf_is_saved_here

        except Exception as e:
            self.logger.error(f"Error while saving PDF file: {e}.")
            raise AllAboutDocumentsException("Failed to save PDF file.", e) from e

    def read_pdf(self, path_of_the_file_to_be_analyzed):
        """
        Read and extract all text from a PDF file using PyMuPDF.

        Args:
            path_of_the_file_to_be_analyzed (str): Full path to the PDF file.

        Returns:
            str: Extracted content, page-wise.

        Raises:
            AllAboutDocumentsException: If file reading fails.
        """
        try:
            text_chunks = []

            # Open PDF and extract text page by page
            with fitz.open(path_of_the_file_to_be_analyzed) as document:
                for page_number, page in enumerate(document, start=1):
                    text_chunks.append(
                        f"\n--- Page {page_number} ---\n{page.get_text()}"
                    )

            text = ".\n".join(text_chunks)

            self.logger.info(
                "PDF file read successfully.",
                saved_path=path_of_the_file_to_be_analyzed,
                session_for_analysis=self.session_id,
                number_of_pages=len(text_chunks),
            )

            return text

        except Exception as e:
            self.logger.error(f"Error while reading the PDF file: {e}.")
            raise AllAboutDocumentsException("Failed to read PDF file.", e) from e


# ----------------------------
# Example usage (only runs when script is executed directly)
# ----------------------------
if __name__ == "__main__":
    from pathlib import Path

    path_of_the_file_to_be_analyzed = (
        r"D:\\00.LLMOPS\\allaboutdocuments\\data\\document_analyzer\\Generative AI.pdf"
    )

    class DummyPdfDocument:
        """
        Simulates a file-like object with .name and .getbuffer() for testing.
        """

        def __init__(self, path_of_the_file_to_be_analyzed):
            self.name = Path(path_of_the_file_to_be_analyzed).name
            self._path_of_the_file_to_be_analyzed = path_of_the_file_to_be_analyzed

        def getbuffer(self):
            with open(self._path_of_the_file_to_be_analyzed, "rb") as f:
                return f.read()

    # Initialize dummy PDF file and handler
    dummy_pdf_document = DummyPdfDocument(path_of_the_file_to_be_analyzed)
    handler = DocumentHandler(session_id=None)

    try:
        # Save and read PDF
        saved_path = handler.save_pdf(dummy_pdf_document)
        print(saved_path)

        content = handler.read_pdf(saved_path)
        print("PDF Content:\n")
        print(content[:500])  # Print only first 500 characters

    except Exception as e:
        print(f"Error: {e}")
