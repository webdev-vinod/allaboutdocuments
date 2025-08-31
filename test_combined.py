############# Test code for document ingestion and analysis using a PDFHandler and DocumentMetadataAnalyzer #############

import io
from pathlib import Path
from datetime import datetime, timezone
import uuid

from src.document_analyzer.data_ingestion import DocumentHandler
from src.document_analyzer.document_metadata_analysis import DocumentMetadataAnalyzer
from src.document_comparator.data_ingestion import DocumentIngestion
from src.document_comparator.document_comparison import DocumentComparatorUsingLLM


# Path to the PDF you want to test
PDF_PATH = r"D:\\00.LLMOPS\\allaboutdocuments\\data\\document_analyzer\\350 NLP Projects with Code.pdf"


class DummyFile:
    """
    Dummy file wrapper to simulate an uploaded file interface (Streamlit style).

    Args:
        file_path (Path or str): Path to the file to wrap.
    """

    def __init__(self, file_path):
        self.name = Path(file_path).name
        self._file_path = file_path

    def getbuffer(self):
        """Return the binary content of the file."""
        return open(self._file_path, "rb").read()


def test_document_analyzer():
    """
    Test function to perform document ingestion and metadata analysis.
    Steps:
        1. Save PDF to session directory.
        2. Extract text content from PDF.
        3. Analyze extracted text for metadata using a pre-trained LLM.
        4. Print the analysis results.
    """
    try:
        # ---------- STEP 1: DATA INGESTION ----------
        print("Starting PDF ingestion...")
        dummy_pdf = DummyFile(PDF_PATH)

        handler = DocumentHandler(
            session_id=f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
        )

        saved_path = handler.save_pdf(dummy_pdf)
        print(f"PDF successfully saved at: {saved_path}")

        text_content = handler.read_pdf(saved_path)
        print(f"Extracted text length: {len(text_content)} characters\n")

        # ---------- STEP 2: DATA ANALYSIS ----------
        print("Starting metadata analysis...")
        analyzer = DocumentMetadataAnalyzer()  # Load LLM and parser

        analysis_result = analyzer.analyze_document(text_content)

        # ---------- STEP 3: DISPLAY RESULTS ----------
        print("\n=== METADATA ANALYSIS RESULT ===")
        for key, value in analysis_result.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Test failed with error: {e}")


# 📁 Define the directory containing PDFs
PDF_DIR = Path("data/document_analyzer")  # relative to project root or adjust as needed


def get_first_pdf_file(directory: Path) -> Path:
    """
    Retrieve the first PDF file found in the specified directory.

    Args:
        directory (Path): Directory to search for PDF files.

    Returns:
        Path: Path of the first PDF file found.

    Raises:
        FileNotFoundError: If the directory does not exist or no PDFs found.
    """
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdf_files = list(directory.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the directory.")

    return pdf_files[0]  # Modify if a different selection logic is needed


def test_document_analyzer_with_dir():
    """
    Test function that ingests and analyzes the first PDF file found in a specified directory.
    Demonstrates file selection, ingestion, text extraction, and metadata analysis.
    """
    try:
        # ---------- STEP 1: DATA INGESTION ----------
        print("Starting PDF ingestion...")

        pdf_path = get_first_pdf_file(PDF_DIR)
        print(f"Selected PDF: {pdf_path}")

        dummy_pdf = DummyFile(pdf_path)

        handler = DocumentHandler(
            session_id=f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
        )

        saved_path = handler.save_pdf(dummy_pdf)
        print(f"PDF successfully saved at: {saved_path}")

        text_content = handler.read_pdf(saved_path)
        print(f"Extracted text length: {len(text_content)} characters\n")

        # ---------- STEP 2: DATA ANALYSIS ----------
        print("Starting metadata analysis...")
        analyzer = DocumentMetadataAnalyzer()  # Load LLM and parser

        analysis_result = analyzer.analyze_document(text_content)

        # ---------- STEP 3: DISPLAY RESULTS ----------
        print("\n=== METADATA ANALYSIS RESULT ===")
        for key, value in analysis_result.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Test failed with error: {e}")


############# Testing code for document comparison using LLMs #############


def load_fake_uploaded_file(file_path: Path) -> io.BytesIO:
    """
    Helper to simulate a file upload by reading file bytes into an in-memory buffer.

    Args:
        file_path (Path): Path to the file to load.

    Returns:
        io.BytesIO: In-memory binary stream of file contents.
    """
    return io.BytesIO(file_path.read_bytes())


def test_compare_documents():
    """
    Test function to perform document comparison using LLMs.
    Steps:
      1. Simulate file uploads of two PDF versions.
      2. Save uploaded files and combine their text.
      3. Perform comparison via LLM-based comparator.
      4. Display preview of combined text and comparison results.
    """
    first_document_path = Path(
        "D:\\00.LLMOPS\\allaboutdocuments\\data\\document_comparator\\The Benefits of LLMs V1.pdf"
    )
    second_document_path = Path(
        "D:\\00.LLMOPS\\allaboutdocuments\\data\\document_comparator\\The Benefits of LLMs V2.pdf"
    )

    class FakeUpload:
        """
        Fake uploaded file simulator with a name and getbuffer method.
        """

        def __init__(self, file_path: Path) -> None:
            self.name = file_path.name
            self.buffer = file_path.read_bytes()

        def getbuffer(self):
            """Return the file's binary content."""
            return self.buffer

    comparator = DocumentIngestion()
    first_document_upload = FakeUpload(first_document_path)
    second_document_upload = FakeUpload(second_document_path)

    first_document, second_document = comparator.save_uploaded_files(
        first_document_upload, second_document_upload
    )
    combined_text = comparator.combine_documents()
    comparator.delete_old_sessions(max_retained_logs=3)

    print("\nCombined text preview (first 500 characters):\n")
    print(combined_text[:500])

    llm_comparator = DocumentComparatorUsingLLM()
    comparison_df = llm_comparator.compare_documents(combined_text)

    print("\nComparison DataFrame preview:\n")
    print(comparison_df.head())


if __name__ == "__main__":
    # Run document analyzer test on single file
    test_document_analyzer()

    # Run document analyzer test using directory scanning
    test_document_analyzer_with_dir()

    # Run document comparison test
    test_compare_documents()
