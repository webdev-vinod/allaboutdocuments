import logging
import os
from datetime import datetime


class AllAboutDocumentsLogger:
    """Thi is a custom logger class that accepts creates a log in the location from where it is being called."""

    def __init__(self, logs_directory="logs"):
        # building the log path
        self.where_are_the_logs = os.path.join(os.getcwd(), logs_directory)

        # creating logs folder
        os.makedirs(self.where_are_the_logs, exist_ok=True)

        # create log file name using system timestamp
        log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

        # create log file
        log_file_path = os.path.join(self.where_are_the_logs, log_file_name)

        # set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            filename=log_file_path,
            format="[ %(asctime)s ] %(levelname)s %(name)s (line:%(lineno)d) - %(message)s",
        )

    def get_AllAboutDocumentsLogger(self, name=__file__):
        # collect info on which module is logging the info
        return logging.getLogger(os.path.basename(name))


if __name__ == "__main__":
    # log the message in the log file
    logger = AllAboutDocumentsLogger()
    logger = logger.get_AllAboutDocumentsLogger(__file__)
    logger.info("This log is from All About Documents Logger")
