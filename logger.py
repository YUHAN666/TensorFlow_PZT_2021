import logging
import os
import time


class Logger(object):

    def __init__(self, param):
        self.param = param
        self.log_path = self.param["log_path"]
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        timer = time.strftime("%Y-%m-%d-%H-%M-%S_", time.localtime())
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
        txthandle = logging.FileHandler((self.log_path + '/' + timer + 'log.txt'))
        txthandle.setFormatter(formatter)
        self.logger.addHandler(txthandle)

    def info(self, string):
        self.logger.info(string)
