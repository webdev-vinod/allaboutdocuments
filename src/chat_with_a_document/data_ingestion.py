# Standard Library Imports
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Third Party Imports

# LangChain Core and Community Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Custom Application Imports
from logger.custom_logger import CustomLogger
from exception.custom_exception import AllAboutDocumentsException
from utils.model_loader import ModelLoader


class SingleDocumentIngestor:
    """
    Ingests a single document:
    - Saves uploaded PDF files with a unique name
    - Splits text into chunks
    - Creates a FAISS index for vector retrieval
    """

    def __init__(
        self,
        data_dir: str = "data/chat_with_a_document",
        faiss_dir: str = "faiss_index",
    ):
        """
        Initialize directories, logger, and model loader.
        """
        try:
            self.logger = CustomLogger().get_logger(__name__)

            # Initialize data directory
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

            # Initialize faiss directory
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()

            self.logger.info(
                "SingleDocumentIngestor initialized successfully.",
                data_directory=str(self.data_dir),
                faiss_directory=str(self.faiss_dir),
            )

        except Exception as e:
            self.logger.error(
                "Failed to initialize SingleDocumentIngestor.", error=str(e)
            )
            raise AllAboutDocumentsException(
                "Initialization failed for SingleDocumentIngestor.", sys
            )

    def ingest_files(self, uploaded_files):
        """
        Ingests uploaded file-like objects and builds retriever.

        Args:
            uploaded_files (list): List of file-like objects.

        Returns:
            retriever (BaseRetriever): A FAISS-based retriever object.
        """
        if not isinstance(uploaded_files, list):
            raise ValueError("uploaded_files must be a list of file-like objects.")

        try:
            documents = []

            for uploaded_file in uploaded_files:
                # Create a unique filename with timestamp and UUID
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                original_extension = Path(uploaded_file.name).suffix or ".pdf"
                unique_filename = (
                    f"{timestamp}_{uuid.uuid4().hex[:8]}{original_extension}"
                )
                temp_path = self.data_dir / unique_filename

                # Save the file locally
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                self.logger.info(
                    "Uploaded file saved locally.",
                    original_filename=uploaded_file.name,
                    saved_path=str(temp_path),
                )

                # Load document using LangChain PDF loader
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)

            self.logger.info(
                "Documents loaded successfully.", document_count=len(documents)
            )

            return self._create_retriever(documents)

        except Exception as e:
            self.logger.error("Document ingestion failed.", error=str(e))
            raise AllAboutDocumentsException("Failed to ingest document(s).", sys)

    def _create_retriever(self, documents):
        """
        Converts documents to vector store and creates retriever.

        Args:
            documents (list): List of LangChain Document objects.

        Returns:
            retriever (BaseRetriever): Retriever for document similarity search.
        """
        try:
            # Split documents into manageable chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=300
            )
            chunks = splitter.split_documents(documents)
            self.logger.info(
                "Documents split into chunks successfully.", chunk_count=len(chunks)
            )

            # Load embedding model
            embedding = self.model_loader.load_embeddings()

            # Create FAISS vector index
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)

            # Save FAISS index to disk
            vectorstore.save_local(str(self.faiss_dir))
            self.logger.info(
                "FAISS index created and saved locally.", path=str(self.faiss_dir)
            )

            # Return vectorstore as retriever object
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

            self.logger.info(
                "FAISS retriever created successfully.",
                retriever_type=str(type(retriever)),
            )
            return retriever

        except Exception as e:
            self.logger.error("Failed to create retriever.", error=str(e))
            raise AllAboutDocumentsException(
                "Error while creating FAISS retriever.", sys
            )
