import logging
import asyncio

LOG_GOOD_FORMAT = "[%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d) %(message)s"

def init_root_logger(logger : logging.Logger = None) -> logging.Logger:
    '''Initialize just enough to get some output for root logger'''
    if logger is None:
        logger = logging.getLogger()
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        _stdout = logging.StreamHandler()
        _stdout.setFormatter(logging.Formatter(LOG_GOOD_FORMAT))
        logger.addHandler(_stdout)
    return logger


def aio_exc_hdl(loop : asyncio.AbstractEventLoop, context):
    logger = logging.getLogger(__name__)
    exception = context.get('exception')
    loop.default_exception_handler(context)
    if exception:
        if isinstance(exception, KeyboardInterrupt):
            logger.info("KeyboardInterrupt received")
            return
        logger.exception("Uncaught exception in asyncio task:", exc_info=exception)

class ColorfulFormatter(logging.Formatter):
    FORMAT_COLORS = {
        'black': 0, 'red': 1, 'green': 2, 'yellow': 3, 'blue': 4, 'magenta': 5, 'cyan': 6, 'white': 7,
        'reset_seq': "\033[0m",
        'color_seq': "\033[1;%dm",
        'bold_seq': "\033[1m"
    }
    LEVEL_COLORS = {
        'WARNING': FORMAT_COLORS['yellow'],
        'INFO': FORMAT_COLORS['green'],
        'DEBUG': FORMAT_COLORS['blue'],
        'CRITICAL': FORMAT_COLORS['yellow'],
        'ERROR': FORMAT_COLORS['red']
    }

    def __init__(self, *args, fmt : str = None, colormode : str = 'ansi', apply_color_sequence : bool = True, **kwargs):
        if fmt is None: fmt = LOG_GOOD_FORMAT
        if colormode == 'ansi' and apply_color_sequence: fmt = fmt.replace('%(levelname)s', '%(ansi_colorlevel)s%(levelname)s%(ansi_end)s')
        logging.Formatter.__init__(self, *args, fmt = fmt, **kwargs)
        self.colormode = colormode

    #Override the format call
    def format(self, record : logging.LogRecord):
        if self.colormode == 'ansi':
            #If we need to show colors, we can preset the format attributes for the record
            if record.levelname in self.LEVEL_COLORS:
                setattr(record, 'ansi_colorlevel', self.FORMAT_COLORS['color_seq'] % (30 + self.LEVEL_COLORS[record.levelname]))
            else:
                setattr(record, 'ansi_colorlevel', '')
            setattr(record, 'ansi_bold', self.FORMAT_COLORS['bold_seq'])
            setattr(record, 'ansi_end', self.FORMAT_COLORS['reset_seq'])
        else:
            #Otherwise, leave it blank so as to not raise an exception for missing attributes
            setattr(record, 'ansi_colorlevel', '')
            setattr(record, 'ansi_bold', '')
            setattr(record, 'ansi_end', '')
        return logging.Formatter.format(self, record)

