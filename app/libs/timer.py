import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self, name ):
        self._start_time = None
        self.info = ""
        self.DEBUG: bool = bool("True")
        self.logger = logging.getLogger( name )
        self.logger.setLevel( logging.INFO )
        file_handler = TimedRotatingFileHandler( "timings.log" , when="midnight")
        file_handler.setFormatter( logging.Formatter("%(message)s" ) )
        self.logger.addHandler( file_handler )
        self.logger.propagate = False

    def start(self, info):
        if self.DEBUG == False:
            return True
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()
        self.info = info

    def stop(self):
        if self.DEBUG == False:
            return True
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        msg = f"{self.info} , {elapsed_time:0.4f}"
        self.logger.info(msg)
        self.info = ""

if __name__ == "__main__":
    import time 

    testTimer = Timer("time_shit") 
    testTimer.start( f"job name")
    time.sleep(5)
    testTimer.stop()