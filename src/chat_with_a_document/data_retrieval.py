# Standard Library Imports
import os
import sys

# Third Party Imports
import streamlit as st
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Custom Application Imports
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import AllAboutDocumentsException
from prompt.prompt_library import PROMPT_REGISTRY
from models.models import PromptType


class ConversationalRAG:
    """
    ConversationalRAG sets up a Retrieval-Augmented Generation (RAG) system with support for session-based
    conversational memory and vector-based document retrieval. It integrates a language model, a retriever,
    and prompt templates for contextual question answering.
    """

    def __init__(self, session_id: str, retriever):
        """
        Initializes ConversationalRAG with a session ID and a retriever. Sets up the LLM, prompts, retriever,
        and full RAG chain including message history tracking.

        Args:
            session_id (str): Unique identifier for the conversation session.
            retriever: A document retriever (e.g., FAISS retriever).
        """
        # Set up logging, track the current session, and configure the retriever for data access
        self.logger = CustomLogger().get_logger(__name__)
        self.session_id = session_id
        self.retriever = retriever

        try:
            # Load the language model using the ModelLoader utility
            self.llm = self._load_llm()

            # Retrieve prompt templates from the prompt registry
            self.contextualize_prompt = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            # Create a retriever that considers prior chat history
            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.logger.info(
                "History-aware retriever created successfully.",
                session_id=self.session_id,
            )

            # Create a QA chain to synthesize answers from retrieved documents
            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

            # Combine retriever and QA chain into a complete RAG chain
            self.rag_chain = create_retrieval_chain(
                self.history_aware_retriever, self.qa_chain
            )
            self.logger.info(
                "Retrieval-Augmented Generation (RAG) chain initialized successfully.",
                session_id=self.session_id,
            )

            # Wrap the RAG chain with chat message history functionality
            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            self.logger.info(
                "RunnableWithMessageHistory initialized successfully.",
                session_id=self.session_id,
            )

        except Exception as e:
            self.logger.error("Error initializing ConversationalRAG.", error=str(e))
            raise AllAboutDocumentsException(
                "Failed to initialize ConversationalRAG.", sys
            )

    def _load_llm(self):
        """
        Loads the language model using the ModelLoader utility.

        Returns:
            llm: A language model instance.
        """
        try:
            llm = ModelLoader().load_llm()
            self.logger.info(
                "Language model loaded successfully.", class_name=llm.__class__.__name__
            )
            return llm
        except Exception as e:
            self.logger.error("Error loading language model.", error=str(e))
            raise AllAboutDocumentsException(
                "Failed to load language model via ModelLoader.", sys
            )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves the chat history object for the given session. Initializes one if not present.

        Args:
            session_id (str): Unique identifier for the session.

        Returns:
            BaseChatMessageHistory: In-memory or session-backed message history.
        """
        try:
            if "store" not in st.session_state:
                st.session_state.store = {}

            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
                self.logger.info(
                    "New chat session history initialized.", session_id=session_id
                )

            return st.session_state.store[session_id]

        except Exception as e:
            self.logger.error(
                "Error retrieving session history.", error=str(e), session_id=session_id
            )
            raise AllAboutDocumentsException("Failed to retrieve session history.", sys)

    def load_retriever_from_faiss(self, index_path: str):
        """
        Loads a FAISS-based retriever from the given local index path and convert to a retriever.

        Args:
            index_path (str): Local path where FAISS index is stored.

        Returns:
            retriever: Configured FAISS retriever.
        """
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(
                    f"FAISS index directory not found: {index_path}"
                )

            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(index_path, embeddings)
            self.logger.info(
                "FAISS retriever loaded successfully from index path.",
                index_path=index_path,
            )

            return vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

        except Exception as e:
            self.logger.error("Error loading FAISS retriever.", error=str(e))
            raise AllAboutDocumentsException(
                "Failed to load retriever from FAISS index.", sys
            )

    def invoke(self, user_input: str) -> str:
        """
        Invokes the RAG pipeline with user input and returns the model's response.

        Args:
            user_input (str): The user's current query.

        Returns:
            str: Generated answer or fallback string.
        """
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}},
            )

            answer = response.get("answer", "no_answer")

            if not answer:
                self.logger.warning(
                    "No answer returned from the model.", session_id=self.session_id
                )

            self.logger.info(
                "Conversation chain invoked successfully.",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=answer[:150],
            )

            return answer

        except Exception as e:
            self.logger.error(
                "Error invoking ConversationalRAG chain.",
                session_id=self.session_id,
                error=str(e),
            )
            raise AllAboutDocumentsException(
                "Failed to invoke ConversationalRAG chain.", sys
            )
