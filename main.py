import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from measure_point import MeasurePoint
from measure_diameter import MeasureDiameter

piece = 1

pickPoints = False
interpolateMissingValues = False
meanGraph = False
multipleMeasures = False

data = pd.read_csv("../path/to/file/" + str(piece) + ".csv", header=None, names=["x", "y", "z"])

measures = np.zeros((3, 10)) #number of measurements, number of measures to be taken for each measurement
means = []
std = []
maxMinDiff = []

pickedPoints = []

def checkPoint(point, startingPoint, measurePoints):
        if not startingPoint.defined:
            if startingPoint.isPoint(point[0], point[1]):
                startingPoint.setRealPoint(point[0], point[1])
        for measurePoint in measurePoints:
            if not measurePoint.defined:
                if measurePoint.isPoint(point[0], point[1]):
                    measurePoint.setRealPoint(point[0], point[1])

def checkDiameter(diameter, startingDiameter, measureDiameters):
    if not startingDiameter.defined:
        if startingDiameter.isDiameter(diameter[0], diameter[1], diameter[2]):
            startingDiameter.setRealDiameter(diameter[2])
    for measureDiameter in measureDiameters:
        if not measureDiameter.defined:
            if measureDiameter.isDiameter(diameter[0], diameter[1], diameter[2]):
                measureDiameter.setRealDiameter(diameter[2])

def onClick(event):
    if event.button == 1 and event.dblclick:
        point = (round(event.xdata, 4), round(event.ydata, 4))
        pickedPoints.append(point)
        print(f"Point {len(pickedPoints)} picked: {point}")

def scan(y):
    run = data[data['y'] == (y / 1000)]

    runZ = run['z'].copy()
    runZ[runZ == '***'] = np.nan
    xStart = np.array(run['x'], dtype=np.float64)
    zStart = np.array(runZ, dtype=np.float64) / 1000
    runMat = np.matrix([xStart, zStart])

    if not pickPoints:
        try:
            points = np.genfromtxt("points_" + str(piece) + ".csv", delimiter=',')
        except FileNotFoundError:
            print("Pick points before run")
            return

        measurePoints = []
        measureDiameters = []
        for i in range(len(points)):
            point = points[i]
            if i == 0:
                startingPoint = MeasurePoint(point[0], point[1])
            elif i >= 1 and i <= 7:
                measurePoints.append(MeasurePoint(point[0], point[1]))
            elif i >= 8 and i <= 17:
                if i % 2 == 0:
                    lastPointX = point[0]
                else:
                    if i == 9:
                        startingDiameter = MeasureDiameter(lastPointX, point[0], point[1])
                    else:
                        measureDiameters.append(MeasureDiameter(lastPointX, point[0], point[1]))
            elif i == 18:
                lastDiam1 = point[0]
            elif i == 19:
                lastDiam2 = point[0]  
        
        refIndices = np.where(((lastDiam1 <= xStart) & (xStart <= lastDiam2)) | ((startingDiameter.xStartRef <= xStart) & (xStart <= startingDiameter.xEndRef)))[0]
        xHorizontalRef = xStart[refIndices]
        zHorizontalRef = zStart[refIndices]
        nonNanIndices = np.logical_not(np.isnan(zHorizontalRef))
        xHorizontalRef = xHorizontalRef[nonNanIndices]
        zHorizontalRef = zHorizontalRef[nonNanIndices]
        m, q = np.polyfit(xHorizontalRef, zHorizontalRef, 1)

        a = -np.arctan(m)
        rotMat = np.matrix([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        rotatedRun = np.matmul(rotMat, runMat)

        run = pd.DataFrame(rotatedRun.transpose(), columns=['x', 'z'])
    else:
        run = pd.DataFrame(runMat.transpose(), columns=['x', 'z'])
    
    if interpolateMissingValues:
        dataGraph = run.copy()
        dataGraph = dataGraph.drop(dataGraph[np.isnan(dataGraph['z'])].index)
        xGraph = dataGraph['x']
        zGraph = np.array(dataGraph['z'], dtype=np.float64)
    else:
        xGraph = run['x']
        zGraphArray = run['z'].copy()
        zGraph = np.array(zGraphArray, dtype=np.float64)

    run = run.drop(run[np.isnan(run['z'])].index)

    runShift = run.shift(1)
    runShift.drop(runShift.index[:1], inplace=True)
    zShift = np.array(runShift['z'], dtype=np.float64)

    run = run.drop(run.index[:1])
    x = np.array(run['x'], dtype=np.float64)
    z = np.array(run['z'], dtype=np.float64)

    if not pickPoints:
        splitIndex = np.where((z < zShift - 0.1) | (z > zShift + 0.1))[0]
        splitIndex = np.insert(splitIndex, 0, 0)
        splitIndex = np.append(splitIndex, len(z))

        splitIndexDiameters = np.where((z < zShift - 0.02) | (z > zShift + 0.02))[0]
        if splitIndexDiameters[0] != 0:
            splitIndexDiameters = np.insert(splitIndexDiameters, 0, 0)
        if splitIndexDiameters[-1] != len(z):
            splitIndexDiameters = np.append(splitIndexDiameters, len(z))    

        for i in range(len(splitIndex) - 1):
            xGroup = x[splitIndex[i]:splitIndex[i + 1]]
            zGroup = z[splitIndex[i]:splitIndex[i + 1]]
            checkPoint((xGroup[0], zGroup[0]), startingPoint, measurePoints)
            checkPoint((xGroup[-1], zGroup[-1]), startingPoint, measurePoints)

        zMeans = np.zeros(len(x))
        for i in range(len(splitIndexDiameters) - 1):
            zGroup = z[splitIndexDiameters[i]:splitIndexDiameters[i + 1]]
            mean = np.mean(zGroup)
            zMeans[splitIndexDiameters[i]:splitIndexDiameters[i + 1]] = mean
            xGroup = x[splitIndexDiameters[i]:splitIndexDiameters[i + 1]]
            checkDiameter([xGroup[0], xGroup[-1], mean], startingDiameter, measureDiameters)

        if startingPoint.defined:
            for i, measurePoint in enumerate(measurePoints):
                if measurePoint.defined:
                    measure = startingPoint.x - measurePoint.x
                    measures[y][i] = measure
                else:
                    print(f"Measure point {i + 1} not found for y={y / 1000}")

        if startingDiameter.defined:
            for i, measureDiameter in enumerate(measureDiameters):
                if measureDiameter.defined:
                    diameter = abs(measureDiameter.zMean - startingDiameter.zMean)
                    measures[y][len(measurePoints) + i] = diameter
                else:
                    print(f"Measure diameter {i + 1} not found for y={y / 1000}")

        if not multipleMeasures:
            print(measures[y])

            if meanGraph:
                xGraph = x
                zGraph = zMeans

            plt.plot(xGraph, zGraph)
            plt.xlim(np.nanmax(xGraph) + 3, np.nanmin(xGraph) - 3)
            plt.ylim(np.nanmin(zGraph) - 1, np.nanmax(zGraph) + 1)
            plt.show()

    else:
        fig, ax = plt.subplots()
        ax.plot(xGraph, zGraph)
        ax.set_xlim(np.nanmax(xGraph) + 3, np.nanmin(xGraph) - 3)
        ax.set_ylim(np.nanmin(zGraph) - 1, np.nanmax(zGraph) + 1)
        fig.canvas.mpl_connect('button_press_event', onClick)
        plt.show()
        if len(pickedPoints) == 20:
            np.savetxt("points_" + str(piece) + ".csv", pickedPoints, delimiter=',')
            print("Points file saved")
        

def main():
    if multipleMeasures:
        if pickPoints:
            print("PickPoints and MultipleMeasures cannot be used together")
            return

        for y in range(measures.shape[0]):
            scan(y)

        #np.savetxt("measures.csv", measures, delimiter=';')

        for dim in range(measures.shape[1]):
            allMeasures = measures[:, dim]
            allMeasures = allMeasures[allMeasures != 0]
            means.append(np.mean(allMeasures))
            std.append(np.std(allMeasures, ddof=1))
            maxMinDiff.append(np.max(allMeasures) - np.min(allMeasures))

        print("Means", means)
        print("Std", std)
        print("MaxMinDiff", maxMinDiff)
    else:
        scan(0) #editable

if __name__ == "__main__":
    main()
