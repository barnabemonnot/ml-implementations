import numpy as np

def kmeans(X, k, eps = 0.001):
    # x is a n x p 2D array
    centroids = X[initialiseCentroids(X, k),:]
    prev = 1.0
    while prev > eps:
        newCentroids = chooseCentroids(X, centroids, k)
        prev = np.linalg.norm(newCentroids-centroids)
        centroids = newCentroids
    return centroids

def initialiseCentroids(X, k):
    # returns centroids, a k x p 2D array using the kmeans++ initialisation procedure
    # https://en.wikipedia.org/wiki/K-means%2B%2B
    n = X.shape[0]
    p = X.shape[1]
    centroids = np.zeros((k, p))
    selectedCentroids = []
    distribution = [1.0/float(n)] * n
    while len(selectedCentroids) < k:
        s = np.random.choice(n, p=distribution)
        selectedCentroids += [s]
        # unselectedCentroids = [item for item in range(n) if item not in selectedCentroids]
        unselectedCentroids = list(set(range(n)) - set(selectedCentroids))
        distances = [min([np.linalg.norm(x - centroid) for centroid in X[selectedCentroids,:]])\
            for x in X[unselectedCentroids,:]]
        distribution = np.zeros((n,))
        distribution[unselectedCentroids] = np.power(distances, 2)
        distribution /= np.sum(distribution)
    return selectedCentroids

def chooseCentroids(X, centroids, k):
    n = X.shape[0]
    closestIdx = np.argmin([[np.linalg.norm(x-centroid) for centroid in centroids] for x in X], axis=1)
    return np.array([np.mean([x for x in X[[i for i in range(n) if closestIdx[i] == centroid]]], axis=0)\
        for centroid in range(k)])


A = np.random.normal(2*[0.1], 0.1, (10,2))
B = np.random.normal(2*[0.9], 0.1, (10,2))
X = np.concatenate((A, B), axis=0)
centroids = kmeans(X, 4)
