# Standard Library Imports
import os
import sys
from operator import itemgetter
from typing import List, Optional

# Third Party Imports
from langchain_community.vectorstores import FAISS
from langchain.schema import BaseMessage, StrOutputParser
# from langchain_core.runnables import RunnableLambda


# Custom Application Imports
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import AllAboutDocumentsException
from prompt.prompt_library import PROMPT_REGISTRY
from models.models import PromptType


class ConversationalRAG:
    def __init__(self, session_id: str, retriever=None):
        """
        Initializes ConversationalRAG with a session ID and a retriever. Sets up the LLM, prompts, retriever,
        and full RAG chain including message history tracking.

        Args:
            session_id (str): Unique identifier for the conversation session.
            retriever: A document retriever (e.g., FAISS retriever).
        """
        try:
            # Set up logging, track the current session, and configure the retriever for data access
            self.logger = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            # Load the language model using the ModelLoader utility
            self.llm = self._load_llm()

            # Retrieve prompt templates from the prompt registry
            self.contextualize_prompt = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            
            self.retriever = retriever
            
            if self.retriever is None:
                raise ValueError("Retriever cannot be None.")

            # Build a LCEL chain
            self._build_lcel_chain()
            self.logger.info(
                "ConversationalRAG initialized successfully.",
                session_id=self.session_id,
            )

        except Exception as e:
            self.logger.error("Error initializing ConversationalRAG.", error=str(e))
            raise AllAboutDocumentsException(
                "Failed to initialize ConversationalRAG.", sys
            )

    def load_retriever_from_faiss(self, index_path: str):
        """
        Loads a FAISS-based retriever from the given local index path and convert to a retriever.

        Args:
            index_path (str): Local path where FAISS index is stored.

        Returns:
            retriever: Configured FAISS retriever.
        """
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(
                    f"FAISS index directory not found: {index_path}"
                )
            
            vectorstore = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )

            self.retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )

            self.logger.info(
                "FAISS retriever loaded successfully from index path.",
                index_path=index_path,
                session_id=self.session_id,
            )

            # self._build_lcel_chain()

            return self.retriever

        except Exception as e:
            self.logger.error("Error loading FAISS retriever.", error=str(e))
            raise AllAboutDocumentsException(
                "Failed to load retriever from FAISS index.", sys
            )

    def invoke(
        self, user_input: str, chat_history: Optional[List[BaseMessage]] = None
    ) -> str:
        """
        Invokes the RAG pipeline with user input and returns the model's response.

        Args:
            user_input (str): The user's current query.

        Returns:
            str: Generated answer or fallback string.
        """
        try:
            chat_history = chat_history  or []

            payload = {"input": user_input, "chat_history": chat_history}

            answer = self.chain.invoke(payload)

            if not answer:
                self.logger.warning(
                    "No answer returned from the model.", session_id=self.session_id
                )
                return "No answer generated"

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

    def _load_llm(self):
        """
        Loads the language model using the ModelLoader utility.

        Returns:
            llm: A language model instance.
        """
        try:
            llm = ModelLoader().load_llm()

            if not llm:
                raise ValueError("LLM could not be loaded.")

            self.logger.info(
                "Language model loaded successfully.",
                class_name=llm.__class__.__name__,
                session_id=self.session_id,
            )
            return llm
        except Exception as e:
            self.logger.error("Error loading language model.", error=str(e))
            raise AllAboutDocumentsException("Failed to load LLM via ModelLoader.", sys)

    @staticmethod
    def _format_documents(docs):
        return "\n\n".join(d.page_content for d in docs)

    def _build_lcel_chain(self):
        try:
            # Rewrite question using chat history
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # Retrieve documents for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_documents

            # Feed context + original input + chat history to the answer prompt
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            self.logger.info(
                "LCEL chain built.",
                session_id=self.session_id,
            )

        except Exception as e:
            self.logger.error("Error building LCEL chain.", error=str(e))
            raise AllAboutDocumentsException("Failed to build LCEL chain.", sys)
