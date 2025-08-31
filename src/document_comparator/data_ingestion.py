# Standard Library Imports
from datetime import datetime, timezone
from pathlib import Path
import uuid

# Third-Party Imports
import fitz  # PyMuPDF for PDF reading

# Local Application Imports
from logger.custom_logger import CustomLogger
from exception.custom_exception import AllAboutDocumentsException


class DocumentIngestion:
    """
    Handles PDF document ingestion and management for comparison or analysis.
    Responsibilities:
    - Saving uploaded PDFs in session-specific folders
    - Reading PDF text content
    - Combining content from multiple documents
    - Cleaning up old session folders
    """

    def __init__(
        self, base_dir: str = "data\document_comparator", session_id=None
    ) -> None:
        """
        Initialize the DocumentIngestion instance.

        Args:
            base_dir (str): Base directory to store session folders.
            session_id (str, optional): If provided, use this session ID; otherwise generate one.
        """
        self.logger = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)

        # Generate unique session ID if not provided
        self.session_id = (
            session_id
            or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
            # f"session_{os.getenv('USER', 'anonymous')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
        )

        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Document ingestion session initialized.",
            session_id=self.session_id,
            session_path=str(self.session_path),
        )

    def save_uploaded_files(self, first_document, second_document):
        """
        Save two uploaded PDF files to the session directory.

        Args:
            first_document: A file-like object with a `.name` and `.getbuffer()` method.
            second_document: Same as above.

        Returns:
            Tuple[Path, Path]: Paths where the documents were saved.
        """
        try:
            # Ensure both documents are PDFs
            if not first_document.name.endswith(
                ".pdf"
            ) or not second_document.name.endswith(".pdf"):
                raise ValueError("Only PDF files are supported.")

            # Define save paths
            first_document_path = self.session_path / first_document.name
            second_document_path = self.session_path / second_document.name

            # Save both files
            with open(first_document_path, "wb") as f:
                f.write(first_document.getbuffer())
            with open(second_document_path, "wb") as f:
                f.write(second_document.getbuffer())

            self.logger.info(
                "Uploaded documents saved successfully.",
                first_document=str(first_document_path),
                second_document=str(second_document_path),
            )

            return first_document_path, second_document_path

        except Exception as e:
            self.logger.error("Failed to save uploaded documents.", error=str(e))
            raise AllAboutDocumentsException(
                "Error while saving uploaded PDF files.", e
            )

    def read_document(self, pdf_path: Path) -> str:
        """
        Extract and return text content from a given PDF file.

        Args:
            pdf_path (Path): Path to the PDF file to read.

        Returns:
            str: Extracted text from all pages.
        """
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError("The document is encrypted and cannot be read.")

                text_extract = []
                for page_number in range(doc.page_count):
                    page = doc.load_page(page_number)
                    page_text = page.get_text()

                    if page_text.strip():
                        text_extract.append(
                            f"\n---- Page {page_number + 1} ----\n{page_text}"
                        )

            self.logger.info(
                "PDF document read successfully.",
                file=str(pdf_path),
                pages=len(text_extract),
            )

            return "\n".join(text_extract)

        except Exception as e:
            self.logger.error("Failed to read PDF document.", error=str(e))
            raise AllAboutDocumentsException("Error while reading the PDF file.", e)

    def combine_documents(self) -> str:
        """
        Combine the text from all PDF documents in the current session folder.

        Returns:
            str: Combined text content from all documents.
        """
        try:
            content_dict = {}
            doc_parts = []

            # Iterate over only PDFs in the session folder
            for pdf_file in sorted(self.session_path.glob("*.pdf")):
                content = self.read_document(pdf_file)
                content_dict[pdf_file.name] = content

            for filename, content in content_dict.items():
                doc_parts.append(f"Document: {filename}\n{content}")

            combined_text = "\n\n".join(doc_parts)

            self.logger.info("Documents combined successfully.", count=len(doc_parts))
            return combined_text

        except Exception as e:
            self.logger.error("Failed to combine documents.", error=str(e))
            raise AllAboutDocumentsException("Error while combining documents.", e)

    def delete_old_sessions(self, max_retained_logs: int = 3):
        """
        Delete all but the most recent N session folders to save space.

        Args:
            max_retained_logs (int): Number of recent session folders to keep.
        """
        try:
            session_folders = sorted(
                [f for f in self.base_dir.iterdir() if f.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            # Delete older sessions beyond the retention limit
            for folder in session_folders[max_retained_logs:]:
                for file in folder.iterdir():
                    file.unlink()
                folder.rmdir()

                self.logger.info("Old session folder deleted.", path=str(folder))

        except Exception as e:
            self.logger.error("Failed to delete old sessions.", error=str(e))
            raise AllAboutDocumentsException(
                "Error while deleting old session folders.", e
            )
