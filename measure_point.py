class MeasurePoint:

    defined = False

    def __init__(self, xRef, zRef):
        self.xRef = xRef
        self.zRef = zRef

    def isPoint(self, x, z):
        return (self.xRef  - 0.1 <= x <= self.xRef + 0.1) and (self.zRef - 0.1 <= z <= self.zRef + 0.1)

    def setRealPoint(self, x, z):
        self.x = x
        self.z = z
        self.defined = True