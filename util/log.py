# logging 库提供了多个组件：Logger、Handler、Filter、Formatter
# Logger      对象提供应用程序可直接使用的接口，树形层级结构，name与Logger实例一一对应
#                   logging.getLogger 获取指定名称logger，没有则创建
#                   子默认逐层继承来自祖先的日志级别、Handler、Filter设置
#                   子会将消息分发给他的handler进行处理也会传递给所有的祖先Logger处理？
# Handler     负责具体发送日志到适当的目的地，一个logger可以有多个handler
# Filter      限制只有满足过滤规则的日志才会输出
# Formatter   指定日志显示格式

# 日志级别等级CRITICAL > ERROR > WARNING > INFO > DEBUG > NOTSET
# logging.debug()、logging.info()、logging.warning()、logging.error()、logging.critical()
# use logging.xx() 默认使用 root logger

# https://www.cnblogs.com/darkpig/p/5924820.html

import cgitb
import logging
import logging.handlers
import os
import sys
from pathlib import Path
import traceback
import random
import shutil
from datetime import datetime
import multiprocessing

_inited = False


# root logger 的包装对象
# 存在一个问题，会导致日志输出行出问题
class MyLogger:
    def __init__(self):
        self._logger = logging.getLogger()  # root logger包装，如果不设置root logger
        # 默认使用控制台打印配置
        self._logger.handlers = []  # reset handlers
        format_str = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt=format_str)  # ,datefmt=""
        # 再创建一个handler，用于输出到控制台 If stream is not specified, sys.stderr is used.
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)
        self._logger.setLevel(logging.INFO)

    @staticmethod
    def args2str(args):
        n = len(args)
        f = "%s " * n
        msg = f.strip() % args
        return msg

    def debug(self, *args, **kwargs):
        # self.log(logging.DEBUG, *args,  **kwargs)
        msg = self.args2str(args)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger._log(logging.DEBUG, msg, (), **kwargs)

    def info(self, *args, **kwargs):
        # self.log(logging.INFO, *args,  **kwargs)
        msg = self.args2str(args)
        if self._logger.isEnabledFor(logging.INFO):
            self._logger._log(logging.INFO, msg, (), **kwargs)

    def warning(self, *args, **kwargs):
        # self.log(logging.WARNING, *args,  **kwargs)
        msg = self.args2str(args)
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger._log(logging.WARNING, msg, (), **kwargs)

    def error(self, *args, **kwargs):
        # self.log(logging.ERROR, *args,  **kwargs)
        msg = self.args2str(args)
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger._log(logging.ERROR, msg, (), **kwargs)

    def log(self, level, *args, **kwargs):
        msg = self.args2str(args)
        if self._logger.isEnabledFor(level):
            self._logger._log(level, msg, (), **kwargs)

    def exception(self, *args, **kwargs):
        """
        Convenience method for logging an ERROR with exception information.
        """
        msg = self.args2str(args)
        self._logger._log(logging.ERROR, msg, (), exc_info=True, **kwargs)


logger = MyLogger()


# 设置root logger， 自动创建文件目录
def init_root_logger(file_path="./output/logging.txt", level=logging.DEBUG, ):
    global _inited
    if _inited: return True  # 避免重复初始化
    try:
        file_path = os.path.abspath(file_path)
        d = os.path.dirname(file_path)
        ensure_mkdirs(d)
        ensure_touch(file_path)

        root_logger = logging.getLogger()
        # if root_logger.hasHandlers():
        #     return  # 避免重复初始化，只初始化一次
        root_logger.handlers = []  # reset handlers

        # 定义handler的输出格式formatter
        # %(asctime)s       字符串形式的当前时间
        # %(name)s          Logger的名字
        # %(filename)s      调用日志输出函数的模块的文件名
        # %(lineno)d        调用日志输出函数的语句所在的代码行
        # %(levelname)s     文本形式的日志级别
        # %(process)d       进程ID 可能没有
        # %(threadName)s    线程名 可能没有
        # %(message)s            用户输出的消息
        format_str = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s'
        formatter = logging.Formatter(fmt=format_str)  # ,datefmt=""

        # 创建一个handler，用于写入日志文件
        # fh = logging.handlers.TimedRotatingFileHandler(file_path, when='h', interval=2,
        #                                                backupCount=100, encoding='utf8')
        fh = logging.handlers.RotatingFileHandler(file_path, maxBytes=1024 * 1024, backupCount=100, encoding='utf8')
        fh.setFormatter(formatter)

        # 再创建一个handler，用于输出到控制台 If stream is not specified, sys.stderr is used.
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)

        # 定义一个filter
        # filter = logging.Filter('mylogger.child1.child2')
        # root_logger.addFilter(filter)
        root_logger.addHandler(fh)
        root_logger.addHandler(ch)

        root_logger.setLevel(level)

        root_logger.info("init/reset root logger done")

        _inited = True
        return True
    except Exception as e:
        logger.info("---日志路径创建失败，检查是否权限或其他问题……")
        logger.error(e)
        return False


# 全局的异常捕获
def init_excepthook(exp_logdir="./logs/_exception_logs_", format='txt'):
    try:
        ensure_mkdirs(exp_logdir)
        # 覆盖默认的sys.excepthook函数，记录所有未补获异常信息
        cgitb.enable(logdir=exp_logdir, format=format)
        logger.info("init_except hook done: " + exp_logdir)
    except:
        logger.info("init exp_logdir fail")


def log_except(ex=None):
    # 记录异常到日志文件
    root_logger = logging.getLogger()
    if ex is not None: root_logger.exception(ex)
    info = traceback.format_exc()
    root_logger.error(info)
    return info


def auto_init_logger(log_dir="./logs/", prefix="pylog",
                     level=None, debug=False, work_dir_file=None, clean_log_file=True):
    """
    按照默认的一套规则初始化日志目录和配置
    :@param log_dir: 日志根目录，最好传绝对路径，默认是项目目录下的
    :@param prefix: 日志文件名前缀
    :@param level: 日志级别
    :@param debug: 日志级别
    :@param work_dir_file: 根目录下py文件的 __file__ 变量，如果需要自动切换work dir到项目根目录，就传这个
    :@param clean_log_file：是否自动清理之前留下的日志文件
    :@return
    """
    if _inited:
        logger.info("logger is inited, return")
        return

    if isinstance(work_dir_file, str):
        change_work_dir(work_dir_file)

    # 初始化目录
    ensure_mkdirs(log_dir)
    ensure_mkdirs(os.path.join(log_dir, "log_bak"))

    # 清理之前的log文件
    if is_subprocess():
        logger.info("subprocess, not clean log file")
    elif clean_log_file:
        logger.info("main process, start to clean pre log file")
        for name in os.listdir(log_dir):
            if (name.endswith(".log") or name.endswith(".txt")) and name.startswith(prefix):
                try:
                    shutil.move(os.path.join(log_dir, name), os.path.join(log_dir, "log_bak", name))
                except Exception as e:
                    logger.info(">>> ERROR: move log file failure", e)

    # 初始化文件
    if level is None:
        level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, "%s_%s_pid_%s.log" % (prefix, timestamp, os.getpid()))

    init_root_logger(path, level=level)
    init_excepthook(os.path.join(log_dir, "_exception_logs_"))
    logger.info("logger init done, path: " + path)


# 本来放在这里不合适的，但是log又依赖，为了做到无依赖，就这样把
def ensure_touch(path: [str, Path]):
    if isinstance(path, str):
        path = Path(path)
    path.touch(mode=0o777, exist_ok=True)  # 这个函数的777有问题 ...
    try:
        os.chmod(str(path), 0o777)  # 修改不属于当前用户的文件，会失败
    except Exception as e:
        pass


def ensure_mkdirs(path: str):
    os.makedirs(path, mode=0o777, exist_ok=True)  # 这个函数的777有问题 ...
    try:
        os.chmod(str(path), 0o777)  # 修改不属于当前用户的文件，会失败
    except Exception as e:
        pass


# 是否是python进程启动的子进程
def is_subprocess():
    p = multiprocessing.current_process()
    return isinstance(p, multiprocessing.Process)  # main process的类型是 MainProcess


def get_file_dirname(_file_):
    return os.path.dirname(os.path.abspath(_file_))


def change_work_dir(_file_):  # __file__
    path = get_file_dirname(_file_)
    os.chdir(path)
    logger.info(">>> change work dir to: " + path)


def random_log_name(prefix="log"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = random.randint(10000, 100000)
    return "%s_%s_%s.log" % (prefix, timestamp, rand)

# 这个问题比较多，暂时不启用
# import pwd
# import grp
# import getpass
# user_name = getpass.getuser() # 获取当前用户名
#
#
# def get_file_owner(file_path):
#     stat_info = os.stat(file_path)
#     uid = stat_info.st_uid
#     gid = stat_info.st_gid
#     user = pwd.getpwuid(uid)[0]
#     group = grp.getgrgid(gid)[0]
#     return user, group
