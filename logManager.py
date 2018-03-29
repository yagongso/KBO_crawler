# logManager.py

import logging
import logging.config
import logging.handlers
import os
import inspect
import traceback

formatter = logging.Formatter(fmt='[%(asctime)s.%(msecs)03d %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')


error_msg = None


def getTracebackStr():
    lines = traceback.format_exc().strip().split('\n')
    if error_msg is not None:
        rl = [error_msg, lines[-1]]
    else:
        rl = [lines[-1]]
    lines = lines[1:-1]
    lines.reverse()
    for i in range(0, len(lines), 2):
        rl.append('  %s at %s' % (lines[i].strip(), lines[i + 1].strip()))
    return '\n'.join(rl)


class LogManager:
    def __init__(self, name=None):
        # main 에서 처음 실행할 때 호출
        # log directory 를 세팅하기 위해 사용
        if (name is not None) and (isinstance(name, str)):
            self.logger = logging.getLogger(name)
        else:
            self.logger = logging.getLogger('LogManager')

        self.logPath = '{}/log/'.format(os.getcwd())
        self.logFileName = 'default.log'

        # 주의 : root logger 에 defaultConfig 를 실행하면 console 출력 됨.

        # set root logger level
        logging.getLogger().setLevel(logging.DEBUG)

        # set my logger level
        self.logger.setLevel(logging.DEBUG)

    def setLogPath(self, lp=''):
        if lp is not '':
            self.logPath = lp
        # omit last slash in path
        if self.logPath[-1] == '/':
            self.logPath = self.logPath[:-1]

    def setLogFileName(self, name='log.txt'):
        if name is not '':
            self.logFileName = name

    def createLogHandler(self):
        if not os.path.isdir(self.logPath):
            os.mkdir(self.logPath)

        file_handler = logging.FileHandler('{}/{}'.format(self.logPath, self.logFileName,
                                                          encoding='utf-8'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def resetLogHandler(self):
        self.logger.handlers = []

    def killLogManager(self):
        logging.shutdown()

    def log(self, msg):
        assert isinstance(msg, str)
        message = '] > {}'.format(msg)

        self.logger.info(message)

    def bugLog(self, msg):
        assert isinstance(msg, str)
        message = 'in {} at {}:{}] > {}'.format(inspect.stack()[1].function,
                                                os.path.basename(inspect.stack()[1].filename),
                                                inspect.stack()[1].lineno,
                                                msg)
        self.logger.debug(message)

    def cleanLog(self):
        # 로그 파일 열기 전에 수행해야 함
        # setLogFilePath 이전에 수행해야 함
        # 파일 열려 있을 때 로그 파일 remove 시도하면 에러
        if os.path.isdir(self.logPath) and os.listdir(self.logPath) is not None:
            if os.path.isfile(self.logFileName):
                os.remove(os.path.join(self.logPath, self.logFileName))
