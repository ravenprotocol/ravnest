class Singleton:
    def __init__(self, cls):
        self._cls = cls

    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError("Singletons must be accessed through `Instance()`.")

    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)


@Singleton
class Globals(object):
    def __init__(self):
        self._name = None
        self._tid = 0

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def tid(self):
        return self._tid

    @tid.setter
    def tid(self, tid):
        self._tid = tid
    

g = Globals.Instance()