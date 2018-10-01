class A(object):
    def __init__(self):
        self.bork = 3

    def opt(self, arg):
        return self.bork + arg

class B(A):
    def opt(self, arg):
        self.arg = arg
        return A.opt(self, arg=self.arg)

