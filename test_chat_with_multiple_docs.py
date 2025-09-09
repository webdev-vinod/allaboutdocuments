#  Standard Library Imports
import sys
from pathlib import Path


#  Local Application Imports
from src.chat_with_multiple_documents.data_ingestion import MultipleDocumentsIngestor
from src.chat_with_multiple_documents.data_retrieval import ConversationalRAG

# Path to the FAISS index directory
FAISS_INDEX_PATH = Path("faiss_index")


def test_conversational_rag_multiple_documents() -> None:
    try:
        test_files = [
            "data/chat_with_multiple_documents/points to beat procrastination.docx",
            "data/chat_with_multiple_documents/Python Architecture Patterns.pdf",
            "data/chat_with_multiple_documents/STEPS TO PROCESS COLORS FILE IN EZ TITLES.txt",
            "data/chat_with_multiple_documents/The Benefits of LLMs V2.pdf"
        ]

        uploaded_files = []

        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No valid file to upload.")
            sys.exit(1)

        ingestor = MultipleDocumentsIngestor()
        retriever = ingestor.ingest_files(uploaded_files)

        for f in uploaded_files:
            f.close()

        session_id = "test_chat_with_multiple_documents"

        rag = ConversationalRAG(session_id=session_id, retriever=retriever)

        question = "What are the benefits of LLMs?"

        response = rag.invoke(question)

        print("\n++++++++++ RAG RESPONSE ++++++++++")
        print(f"\n Question: {question}")
        print(f"Answer: {response}")

    except Exception as e:
        print(f"❌ Test failed. Reason: {str(e)}.")
        sys.exit(1)


if __name__ == "__main__":
    test_conversational_rag_multiple_documents()
