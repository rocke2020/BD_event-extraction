import logging, os, sys

def get_logger(name=__name__, log_file=None, log_level=logging.DEBUG):
    logger = logging.getLogger(name)
    if name == 'app':
        fmt= '%(asctime)s %(filename)10s %(levelname)s L %(lineno)d: %(message)s'
    else:
        fmt= '%(asctime)s %(name)s %(levelname)s L %(lineno)d: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(log_level)
    return logger

