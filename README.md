# clustering
K-means clustering python implementation on Iris flower data set: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_ dataset.html

## Context
K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.

Typically, unsupervised algorithms make inferences from datasets using only input vectors without referring to known, or labelled, outcomes. A cluster refers to a collection of data points aggregated together because of certain similarities.

A target number  k is defined, which refers to the number of centroids you need in the dataset. A centroid is the imaginary or real location representing the center of the cluster. Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares. In other words, the K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible. The ‘means’ in the K-means refers to averaging of the data; that is, finding the centroid.

**Algorithm**
To process the learning data, the K-means algorithm in data mining starts with a first group of randomly selected centroids, which are used as the beginning points for every cluster, and then performs iterative (repetitive) calculations to optimize the positions of the centroids

It halts creating and optimizing clusters when either:
- The centroids have stabilized — there is no change in their values because the clustering has been successful.
- The defined number of iterations has been achieved.

## Program

The program that was written first reads in the 4 input features of the iris dataset and selects k random starting centroids from the dataset. Each centroid is stored in a dictionary object consisting of <label, centroid coordinates>. Each datapoint’s location is compared against each centroid to determine which one is closer. The datapoint is then appended to the cluster with the closest centroid. Each time this is performed on the dataset, the centroid will be updated. This process of clustering data is repeated until the centroids converge or the process is iterated 500 times.

To help assess the quality of the results, the variance of each cluster was calculated and summed. As presented in the figure below, there should be diminishing returns in how small the variance gets as k is increased. This should highlight the ideal value of k.

![image](https://github.com/haydensflee/clustering/assets/89950637/5f94c0b5-10ca-437a-b1b1-70fcaa795aec)

Since the centroid is randomly selected for each time the program is run, clustering was conducted multiple times for each value of k. However, the results were mostly the same for each k as presented below.
K=1. Variance = 3.227190599703595. Runtime = 1.1685631275177002 seconds

![image](https://github.com/haydensflee/clustering/assets/89950637/7cd72d97-6698-45ed-ba06-1fa11c8db5b1)

K=2. Variance = 1.0445678684621047. Runtime = 1.181516170501709 seconds

![image](https://github.com/haydensflee/clustering/assets/89950637/c65c0713-e774-4fe4-b6c0-7d67f790f61b)

K=3. Variance = 0.8942907432497242. Runtime = 1.1985220909118652 seconds

![image](https://github.com/haydensflee/clustering/assets/89950637/1de0925e-2a75-4341-85e1-3ffcdab2e06e)

K=4. Variance = 0.8860957921534436. Runtime = 1.1965293884277344 seconds

![image](https://github.com/haydensflee/clustering/assets/89950637/af3cb9fa-be02-4691-8eb8-5da4a8c7bb24)

K=4 (again). Variance = 0.883496902381032. Runtime = 1.1880218982696533 seconds

![image](https://github.com/haydensflee/clustering/assets/89950637/d5d5cdd9-25bf-424c-b6b4-f5cf7e3e29c0)

As shown in the results, the variance in the clusters grew smaller as k was increased, until k=4 was reached. At this point, the variance decreased a very small amount, and converged to different outcomes, of which two have been presented. Both qualitatively and quantitatively, it was observed that k=3 was the ideal choice and produced the best outputs, which was expected as the Iris dataset has three data labels.
Runtime is O(n^2).
