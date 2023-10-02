import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import random
import time


def main():
    start = time.time()
    for i in range(10):
        print(i)
    time.sleep(1)
    print("Hello World!")
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :4]  # we only take the first two features.
    y = iris.target
    N=len(X)

    k=3
    maxIterations_2=500
    
    oldCentroidDict={}
    centroidDict={}
    converged=False
        
    # initialise clusters

    
    centroidList=[]
    centroidDict={}
    dataLabelDict={}
    
    # For specified number k, generate random starting cluster centroids.
    for i in range(k):
        tempCentroid=[]
        for j in range(4):
            
            column=[row[j] for row in X]
            element=random.choice(column)
            tempCentroid.append(element)
            
        centroidList.append(np.array(tempCentroid))
        centroidDict[i]=np.array(tempCentroid)
    
    labels=[]
    x = np.array(X[0])
    y = centroidDict[0]

    iterations_2=0
    while (iterations_2<maxIterations_2) and (converged==False):
        
        oldCentroidDict=centroidDict.copy()
        clusteredData=[]
        for i in range(k):
            clusteredData.append([])
        centroidDistanceList=[]

        # For each data point, find the smallest distance to a centroid and give the point a label.
        for i in range(N):
            distanceList=[]
            point=X[i]
            for key, value in centroidDict.items():
                
                distance=np.sqrt(sum((value - point) ** 2))
                distanceList.append(distance)

            minDist = min(distanceList)
            label = distanceList.index(minDist)
            centroidDistanceList.append(minDist)
            clusteredData[label].append(point)
            dataLabelDict[i]=label
        for key,value in centroidDict.items():
            tempArray=np.array(clusteredData[key])
            tempValue=(tempArray.sum(axis=0))/len(clusteredData[key])
            centroidDict[key]=tempValue
        iterations_2+=1  

        # If the previous iteration centroid matches the current centroid, then it is converged
        for i in range(k):
            converged=True
            if(np.array_equal(centroidDict[i],oldCentroidDict[i])):
                converged=True
            else:
                converged=False
                break
        
    iterations_1+=1
    # VARIANCE CALCULATIONS
    varianceLengthSum=0
    for a in range(k):
        N=len(clusteredData[a])
        SUMSQ=[]
        SUM=[]
        tempNPArray=np.array(clusteredData[a])
        SUM=tempNPArray.sum(axis=0)
        SUMSQ=(tempNPArray**2).sum(axis=0)
        variance=SUMSQ/N-(SUM/N)**2
        print(variance)
        varianceLength=np.linalg.norm(variance)
        print("Variance Length = ", end = ' ')
        print(varianceLength)
        print("---")
        varianceLengthSum=varianceLengthSum+varianceLength
        print("Variance Length Sum = ", end = ' ')
        print(varianceLengthSum)
    colours=[]
    pointsToPlotX=[]
    pointsToPlotY=[]
    for i in range(k):
        tempVec=[]
        for j in range(len(clusteredData[i])):
            colours.append(i)
            pointsToPlotX.append(clusteredData[i][j][0])
            pointsToPlotY.append(clusteredData[i][j][1])


    y = iris.target

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5


    # Plot the training points
    for i in range(k):
        plt.scatter(pointsToPlotX, pointsToPlotY, c=colours, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    fig = plt.figure(1, figsize=(8, 6))
    plt.show()
    end = time.time()
    print(f"Runtime of the program is {end - start}")       

if __name__ == "__main__":
    main()