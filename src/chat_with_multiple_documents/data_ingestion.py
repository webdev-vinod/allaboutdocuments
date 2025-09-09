# Standard Library Imports
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

# Third-Party Imports
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever

# Local Application Imports
from logger.custom_logger import CustomLogger
from exception.custom_exception import AllAboutDocumentsException
from utils.model_loader import ModelLoader


class MultipleDocumentsIngestor:
    """
    Handles the ingestion of multiple document types (PDF, DOCX, TXT),
    stores them locally, loads their contents, and prepares them for retrieval
    using a FAISS vector store.

    Attributes:
        temp_dir (str): Directory to temporarily store uploaded documents.
        faiss_dir (str): Directory to store FAISS index files.
        session_id (str): Unique session identifier for this ingestion session.
    """

    SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

    def __init__(
        self,
        temp_dir: str = "data/chat_with_multiple_documents",
        faiss_dir: str = "faiss_index",
        session_id: str | None = None,
    ):
        """
        Initialize the document ingestor, setting up directories and session identifiers.

        Args:
            temp_dir (str): Temporary directory for storing uploaded documents.
            faiss_dir (str): Directory for storing FAISS vector indices.
            session_id (str, optional): Unique identifier for the session. If not provided, generated automatically.
        """
        try:
            self.logger = CustomLogger().get_logger(__name__)

            # 📁 Create base directories for storage
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            # 🔑 Generate a unique session ID if not provided
            self.session_id = (
                session_id
                or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}"
            )

            # 📁 Create session-specific directories
            self.session_temp_dir = self.temp_dir / self.session_id
            self.session_temp_dir.mkdir(parents=True, exist_ok=True)

            self.session_faiss_dir = self.faiss_dir / self.session_id
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            # 🧠 Initialize model loader (for embeddings)
            self.model_loader = ModelLoader()

            self.logger.info(
                "✅ MultipleDocumentsIngestor initialized successfully.",
                temp_dir=str(self.temp_dir),
                faiss_dir=str(self.faiss_dir),
                session_id=str(self.session_id),
                temp_path=str(self.session_temp_dir),
                faiss_path=str(self.session_faiss_dir),
            )

        except Exception as e:
            self.logger.error(
                "❌ Error initializing MultipleDocumentsIngestor.", error=str(e)
            )
            raise AllAboutDocumentsException(
                "Initialization failed for MultipleDocumentsIngestor.", sys
            )

    def ingest_files(self, uploaded_files) -> VectorStoreRetriever:
        """
        Ingest and process uploaded files, extract their content,
        and return a retriever for querying the data.

        Args:
            uploaded_files (List[BinaryIO]): List of uploaded file-like objects.

        Returns:
            FAISS: A retriever object for performing semantic search on the ingested content.

        Raises:
            AllAboutDocumentsException: If no valid documents are loaded or ingestion fails.
        """
        try:
            documents = []

            for uploaded_file in uploaded_files:
            
                uploaded_file_ext = Path(uploaded_file.name).suffix.lower()

                # ⚠️ Skip unsupported file types
                if uploaded_file_ext not in self.SUPPORTED_EXTENSIONS:
                    self.logger.warning(
                        "⚠️ Unsupported file type skipped.", skipped_filename=uploaded_file.name
                    )
                    continue

                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                
                # 📄 Save the file to the session directory
                unique_filename = (
                    f"{timestamp}_{uuid.uuid4().hex[:8]}{uploaded_file_ext}"
                )
                temp_path = self.session_temp_dir / unique_filename

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())

                self.logger.info(
                    "📄 Uploaded file saved locally for ingestion.",
                    filename=uploaded_file.name,
                    saved_as=str(temp_path),
                    session_id=self.session_id,
                )

                # 🧹 Load file using appropriate loader
                if uploaded_file_ext == ".pdf":
                    loader = PyPDFLoader(str(temp_path))
                elif uploaded_file_ext == ".docx":
                    loader = Docx2txtLoader(str(temp_path))
                elif uploaded_file_ext == ".txt":
                    loader = TextLoader(str(temp_path), encoding="utf-8")
                else:
                    self.logger.warning("Unsupported file type encountered", filename=uploaded_file.name)
                    continue  # Should not happen due to earlier check

                docs = loader.load()
                documents.extend(docs)

            # ❌ Handle case where no valid documents were loaded
            if not documents:
                raise AllAboutDocumentsException("No valid documents were loaded.",sys)

            self.logger.info(
                "📚 All valid files loaded successfully.",
                total_documents=len(documents),
                session_id=self.session_id,
            )

            return self._create_retriever(documents)

        except Exception as e:
            self.logger.error("❌ Error ingesting files.", error=str(e))
            raise AllAboutDocumentsException(
                "Failed to ingest files in MultipleDocumentsIngestor.", sys
            )

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
                "📄 Documents split into chunks successfully.",
                chunk_count=len(chunks),
                session_id=self.session_id,
            )

            # Load embedding model
            embeddings = self.model_loader.load_embeddings()

            # Create FAISS vector index
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

            # Save FAISS index to disk in session folder
            vectorstore.save_local(str(self.session_faiss_dir))
            self.logger.info(
                "💾 FAISS index created and saved locally.",
                path=self.faiss_dir,
                session_id=self.session_id,
            )

            # Return vectorstore as retriever object
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )

            self.logger.info(
                "✅ FAISS retriever created successfully.",
                retriever_type=str(type(retriever)),
            )
            return retriever

        except Exception as e:
            self.logger.error("❌ Error creating retriever.", error=str(e))
            raise AllAboutDocumentsException("❌ Failed to create retriever.", sys)
