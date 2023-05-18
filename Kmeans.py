import numpy as np

def init_centroids(X, K):
    """
    Inicializa las ubicaciones de los K centroids.
    
    Args:
        - X (ndarray (m,n)): dataset
        - K (int): número de clusters
    
    Returns:
        - centroids (ndarray (K,n)): numpy array de los K centroids inicializados
    """
    m = X.shape[0]
    if K >= m:
        raise ValueError("El número de clusters debe ser menor al número de ejemplos")
        
    random_index = np.random.permutation(m) # revuelve los indices
    centroids = X[random_index[:K]] # elegimos los k primeros ejemplos aleatorios como ubicaciones iniciales de los centroids.
    
    return centroids


def find_closests_centroids(X, centroids):
    """
    Encuentra para cada ejemplo en data el centroid más cercano de acuerdo al cuadrado de la norma L2
    
    Args:
        - X (ndarray (m,n)): dataset
        - centroids (ndarray (K,n)): K centroids
        
    Returns:
        - index_cent (ndarray (m,)): numpy array con los índices de los centroids más cercanos para cada ejemplo. 
    """
    
    index_cent = np.zeros(X.shape[0], dtype=int) # vector de indices
    
    for i, ejemplo in enumerate(X):
        distancias = np.zeros(centroids.shape[0]) # K distancias, una para centroid
        
        for j, cent in enumerate(centroids):
            dist = np.linalg.norm(ejemplo - cent) ** 2 # distancia entre el ejemplo i y el centroid j
            distancias[j] = dist
        
        index_cent[i] = np.argmin(distancias) # se le asigna al ejemplo i el centroid más cercano
    
    return index_cent


def move_centroids(X, K, index_cent):
    """
    Mueve la ubicación de todos los centroids a las medias de sus respectivos clusters.
    
    Args:
        - X (ndarray (m,n)): dataset
        - K (int): número de centroids
        - index_cent (ndarray (m,)): centroids asignados para cada ejemplo dentro del dataset X.

    Returns:
        - centroids (ndarray (K, n)): K centroids con sus nuevas ubicaciones
    
    """
    
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        ejemplos_cluster = X[index_cent == k] # ejemplos asignados al centroid k
        centroids[k] = np.mean(ejemplos_cluster, axis=0) # media de los ejemplos del cluster k
    
    return centroids


def compute_cost(X, centroids, index_cent):
    """
    Retorna el costo calculado a partir de la función de costo o distortion que el algoritmo 
    K-means intenta minimizar.
    La función de costo o distortio utilizada es el promedio de las distancias al cuadrado
    entre cada ejemplo X[i] y la ubicación del centroid en su cluster.
    
    Args:
        - X (ndarray (m,n)): data
        - centroids (ndarray (K, n)): centroids de cada cluster
        - index_cent (ndarray (m,)): índice del centroid asignado a cada ejemplo X[i]
    
    Returns:
        - cost (float): costo o distancia promedio entre los ejemplos y sus centroids.
    """
    
    m = X.shape[0]
    cost = 0.0
    for k, cent in enumerate(centroids):
        ejemplos = X[index_cent == k] # todos los ejemplos en el cluster k
        cost = cost + np.linalg.norm(ejemplos - cent) ** 2 # costo del cluster k
    
    cost = cost / m
    return cost


def run_Kmeans(X, initial_centroids, num_iterations, verbose=False):
    """
    Ejecuta el algoritmo K-means sobre el dataset X.
    
    Args:
        - X (ndarray (m,n)): data
        - initial_centroids (ndarray (K,n)): centroids inicializados
        - num_iterations (int): número de veces que se ejecutará K-means
        - verbose (bool): True si se imprime el costo en cada iteración. False por
                        defecto.

    Returns:
        - cost (list): lista con los costos calculados en cada iteración.
        - centroids (ndarray (K,n)): numpy array con los centroids de cada cluster
                                   encontrados por K-means.
        - index_centroids (ndarray (m,)): numpy array con los índices de los centroids
                                        asignados a cada ejemplo X[i].
    """
    cost = []
    centroids = initial_centroids
    K = centroids.shape[0]
    index_centroids = np.zeros(X.shape[0], dtype=int)
    
    print("Running K-means algorithm ...")
    for i in range(num_iterations):
        index_centroids = find_closests_centroids(X, centroids)
        centroids = move_centroids(X, K, index_centroids)
        
        cost.append(compute_cost(X, centroids, index_centroids))
        
        if verbose:
            print(f"Costo en iteracion {i}: {cost[i]}") # costo
    
    print("Finished.")
    
    return cost, centroids, index_centroids

