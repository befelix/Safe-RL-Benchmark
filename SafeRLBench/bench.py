

class Bench(object):
    def __init__(self, algos=None, measures=None):

        if algos is None:
            self.algos = []
        else:
            self.algos = algos

        if measures is None:
            self.measures = []
        else:
            self.measures = measures

    def __call__(self, parameters=None):
        self.benchmark(parameters)

    def benchmark(self, parameters=None):
        pass

    def addAlgorithm(self):
        pass

    def addMeasure(self):
        pass

    def plotMeasure(self):
        pass
