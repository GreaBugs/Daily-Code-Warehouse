import logging


class ColoredFormatter(logging.Formatter):
    COLOR_CODE = {
        'WARNING': '\033[93m',
        'INFO': '\033[92m',
        'DEBUG': '\033[94m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
        'ENDC': '\033[0m'
    }

    def format(self, record):
        log_color = self.COLOR_CODE.get(record.levelname, self.COLOR_CODE['ENDC'])
        return f"{log_color}{super().format(record)}{self.COLOR_CODE['ENDC']}"


def create_logger(logger_name):
    # ============================ 1、实例化 logger ============================
    # 实例化一个记录器，并将记录器的名字设为 'training_log'，并将日志级别为 info
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # ============================ 2、定义Handler ============================
    # 创建一个往 console打印输出的 Handler，日志级别为 debug
    consoleHandler = logging.StreamHandler()

    # 再创建一个往文件中打印输出的handler
    fileHandler = logging.FileHandler(filename='mnist.log', mode='w')

    # ============================ 3、定义打印格式 ============================

    simple_formatter = ColoredFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')

    # ============================ 4、绑定 ============================
    # 让 consoleHandler 使用 简单版日志打印格式
    consoleHandler.setFormatter(simple_formatter)
    # 让 fileHandler 使用 简单版日志打印格式
    fileHandler.setFormatter(simple_formatter)

    # 给记录器绑定上 consoleHandler 和 fileHandler
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    
    return logger
