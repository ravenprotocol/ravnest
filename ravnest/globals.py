import os
from .singleton_utils import Singleton

@Singleton
class Globals(object):
    def __init__(self):
        self._forward_done = False

    @property
    def forward_done(self):
        return self._forward_done
    
    @forward_done.setter
    def forward_done(self, forward_done):
        self._forward_done = forward_done

g = Globals.Instance()