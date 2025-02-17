# Dependencies
import logging
from pathlib import Path

# Logger Setuo
def setup_logger(model_name:str , log_dir:str="logs" , log_file : str="training.log"):
    """
    Sets up a logger to save training information
    
    Arguments:
    ----------
    model_name {str} : Name of model
    log_dir {str}    : Derectory of log file
    file_log {str}   : Log file name
    """
    Path(log_dir).mkdir(parents  = True,
                        exist_ok = True)
    log_file = f"{model_name}_training.log"
    log_path = Path(log_dir)/log_file

    logging.basicConfig(filename = log_path,
                        filemode = "a",
                        format   = "%(asctime)s - %(levelname)s - %(message)s",
                        level    = logging.INFO
                        )
    return logging.getLogger()
