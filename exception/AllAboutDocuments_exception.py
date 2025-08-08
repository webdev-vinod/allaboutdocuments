import sys
import traceback
from logger.AllAboutDocuments_logger import AllAboutDocumentsLogger


logger = AllAboutDocumentsLogger().get_AllAboutDocumentsLogger(__file__)


class AllAboutDocumentsException(Exception):
    """Custom Exception class for All About Documents poral"""

    def __init__(self, error_message, error_details: sys):
        _, _, exception_traceback = error_details.exc_info()
        self.file_name = exception_traceback.tb_frame.f_code.co_filename
        self.linenumber = exception_traceback.tb_lineno
        self.error_message = str(error_message)
        self.traceback_str = "".join(
            traceback.format_exception(*error_details.exc_info())
        )

    def __str__(self):
        return f"""
            Error in [{self.file_name}] at line number [{self.linenumber}]
            Message: {self.error_message}
            Traceback : {self.traceback_str}
            """


if __name__ == "__main__":
    try:
        # simulate exception of Divide by Zero
        a = 1 / 0
    except Exception as e:
        exception_encountered = AllAboutDocumentsException(e, sys)
        logger.error(exception_encountered)
        raise exception_encountered
