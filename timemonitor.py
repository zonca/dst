import timeit
from collections import OrderedDict
import logging as l

_monitors = None
_format = "%-20s : %.3f s\n"
MyPID = None

def set_mypid(mypid):
    global MyPID
    MyPID = mypid

def reset():
    global _monitors
    _monitors = OrderedDict()

def summarize(format=_format):
    global MyPID
    if MyPID == 0:
        out = "Time Monitor\n"
        for k in sorted(_monitors.keys()):
            out += format % (k, _monitors[k])
        return out

class TimeMonitor:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = timeit.default_timer()
        global _monitors
        if not _monitors.has_key(self.name):
            _monitors[self.name] = 0
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.interval = self.end - self.start
        global _monitors
        global MyPID
        if MyPID == 0:
            _monitors[self.name] +=  self.interval

# set initial values for _monitors
reset()
