import os

from agent import Agent
from config import DefaultParam
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():

    param = DefaultParam
    logger = Logger(param)
    agent = Agent(param, logger)
    agent.run()


if __name__ == '__main__':
    main()
