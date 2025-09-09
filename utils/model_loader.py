import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config_yaml
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from exception.custom_exception import AllAboutDocumentsException

# initialize logger to log events
modelLoaderCustomLogger = CustomLogger().get_logger(__name__)


# create a class to load model
class ModelLoader:
    """
    This is a class written to load and test environment and LLL models
    """

    def __init__(self):
        """This function loads .env file, calls the _validate_environments_variables function"""
        load_dotenv()
        self._validate_environment_variables()
        self.config = load_config_yaml()
        modelLoaderCustomLogger.info("Configuration loaded without any error.")

    def _validate_environment_variables(self):
        """
        Validates that all required API keys are present in the environment.
        Ensure all API keys are listed in .env file.
        """

        mandatory_api_keys_to_be_listed_in_config = ["GOOGLE_API_KEY", "GROQ_API_KEY"]

        # create dictionary of API keys and their secrets
        self.api_keys = {
            api_key: os.getenv(api_key)
            for api_key in mandatory_api_keys_to_be_listed_in_config
        }

        # Check if any API value is missing for any key from the list of mandatory variables
        missing_api_secret_keys_for_mandatory_api_keys = [
            api_key for api_key, api_value in self.api_keys.items() if not api_value
        ]

        # raise an exception on missing API keys
        if missing_api_secret_keys_for_mandatory_api_keys:
            modelLoaderCustomLogger.error(
                "Missing environment variables: ",
                missing_api_secret_keys_for_mandatory_api_keys,
            )
            raise AllAboutDocumentsException("Missing environment variables: ", sys)
        modelLoaderCustomLogger.info(
            "Environment variables validated",
            keys_mentioned_in_config_file=[
                api_key for api_key in self.api_keys if self.api_keys[api_key]
            ],
        )

    def load_embeddings(self):
        """
        This functions loads and returns the embedding model.
        """
        try:
            modelLoaderCustomLogger.info("Loading embedding model...")
            embedding_model_name = self.config["embedding_model"][
                "embedding_model_name"
            ]
            return GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        except Exception as e:
            modelLoaderCustomLogger.error(
                "Error while loading embedding model", error=str(e)
            )
            raise AllAboutDocumentsException(
                "Failure while load the embedding model", sys
            )

    def load_llm(self):
        """
        This functions loads LLM based on the provider details in the config and returns the embedding model.
        """

        # read LLMs block from config.YAML file
        llm_block = self.config["LLMs"]

        # log the information to log file
        modelLoaderCustomLogger.info("Loading LLM...")

        # since there is no value set for LLM_PROVIDER in .env file,
        # the fallback value 'groq' is set as the default llm provider
        llm_provider_key = os.getenv("LLM_PROVIDER", "google")

        # raise an exception if the LLM provider key is not present in config.YAML file
        if llm_provider_key not in llm_block:
            # if not provider or not llm_model_name:
            modelLoaderCustomLogger.error(
                "LLM provider not found in config", provider_key=llm_provider_key
            )
            raise ValueError(
                f"Missing required LLM configuration parameters : '{llm_provider_key}'"
            )

        # once the LLM provider key is available in the config.YAML file, read the LLM parameters into variables
        llm_config = llm_block[llm_provider_key]
        llm_provider = llm_config.get("provider")
        llm_model_name = llm_config.get("llm_model_name")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_output_tokens", 2048)

        modelLoaderCustomLogger.info(
            "Loading LLM",
            provider=llm_provider,
            model_name=llm_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if llm_provider == "google":
            llm = ChatGoogleGenerativeAI(
                model=llm_model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return llm

        elif llm_provider == "groq":
            llm = ChatGroq(
                model=llm_model_name,
                api_key=self.api_keys["GROQ_API_KEY"],
                temperature=temperature,
            )
            return llm
        else:
            modelLoaderCustomLogger.error(
                "Unsupported LLM provider", provider=llm_provider
            )
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")


if __name__ == "__main__":
    # create an instance of ModelLoader for testing
    test_loader = ModelLoader()

    # check if embeddings are loaded properly
    test_embeddings = test_loader.load_embeddings()
    print(f"Loaded embedding is {test_embeddings}")

    # check if LLMs are loaded properly as per YAML file
    test_llm = test_loader.load_llm()
    print(f"Loaded LLM is {test_llm}")

    # Query the LLM to see if everything is order and LLM is able to answer
    test_query = "What's your name"
    test_result = test_llm.invoke(test_query)
    # print(f"{test_result.content}")
    # print("\n test_query")
    modelLoaderCustomLogger.info(f"\n My answer is for {test_result.content}")
