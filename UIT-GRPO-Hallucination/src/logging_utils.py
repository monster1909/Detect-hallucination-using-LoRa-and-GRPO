# src/logging_utils.py
import logging
import sys

def setup_logger(log_file='training.log'):
    """Thiết lập logger để ghi ra file và console."""
    
    # Xóa bất kỳ handler nào đã có để tránh log trùng lặp
    logging.getLogger().handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Định dạng log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler cho file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler cho console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger