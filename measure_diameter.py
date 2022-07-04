class MeasureDiameter:

    defined = False

    def __init__(self, xStartRef, xEndRef, zMeanRef):
        self.xStartRef = xStartRef
        self.xEndRef = xEndRef
        self.zMeanRef = zMeanRef

    def isDiameter(self, xStart, xEnd, zMean):
        return xEnd - xStart >= 1 and self.xStartRef - 0.1 <= xStart and xEnd <= self.xEndRef + 0.1 and (self.zMeanRef - 0.05 <= zMean <= self.zMeanRef + 0.05)

    def setRealDiameter(self, zMean):
        self.zMean = zMean
        self.defined = True