import logging
import os

def setup_logger(save_dir, use_sampler):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(save_dir, 'training_log.txt'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'{"" if use_sampler else "Not"} Using Sampler\n')
    return logger
