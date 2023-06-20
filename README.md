# Clustering_algorithm: K-means
Clustering algorithm: K-means from scratch.

## How to use
1. Create a `KMeans` object with these arguments:
    - num_clusters: number of clusters of the algorithm.
    - num_ejecuciones: number of runs of the algorithm.
    - num_iter: number of iterations for each run.
    ```
    model = KMeans(num_clusters, num_ejecuciones, num_iter)
    ```

2. call the `fit()` method of the model and pass it the `data`. `data` must be a 2-d `numpy` array with dimensions `(m,n)`, `m` is the number of examples and `n` is the number of features.
    ```
    history, centroids, index_centroids = model.fit(data)
    ```
    The `fit()` method will return: 
    - `history`: history of the cost got by the best run of the K-means algorithm.
    - `centroids`: 2-d `numpy` array with the locations of the cluster centroids of the best run of the algorithm. This array has dimensions `(k, n)`, where `k` is the number of clusters.
    - `index_centroids`: 1-d `numpy` array with the index of the cluster centroids assigned to each example of `data`. This array has a dimension of `(m,)`.

3. **Optional**. You can call the `compute_cost()` method to compute the cost of a set of clusters on a data.
    ```
    compute_cost(data, centroids, index_centroids)
    ``` 

## License
GPL-3.0