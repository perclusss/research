## import modules here

import numpy as np
from collections import deque


def dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


class BiCluster(object):
    def __init__(self, vec, left=None, right=None, distance=-1, id=None):
        self.vec = np.array(vec)
        self.left = left
        self.right = right
        self.distance = distance
        self.id = id

    def __repr__(self):
        cn = self.__class__.__name__
        d = {
            'left': self.left,
            'right': self.right,
            'distance': self.distance,
            'id': self.id,
        }
        return '{}({}, **{})'.format(cn, repr(self.vec), repr(d))

    def get_leaves(self):
        leaves = []
        queue = deque([self])
        while queue:
            cluster = queue.popleft()
            if cluster.left is None:
                leaves.append(cluster)
            else:
                queue.append(cluster.left)
                queue.append(cluster.right)
        return leaves

    def get_leaves_ids(self):
        return [x.id for x in self.get_leaves()]


# This function performs the hierarchical algorithm for clustering the sets of data
def hcluster(points):
    distances = {}
    current_cluster_id = -1

    # Each cluster is a row of data
    clusters = [BiCluster(pt, id=ix) for ix, pt in enumerate(points)]

    while len(clusters) > 1:
        lowestpair = (0, 1)
        closest = dot_product(clusters[0].vec, clusters[1].vec)

        # loop through each cluster looking for the smallest distance between each
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                # the 'distances' array holds the result of the distance metric calculation between each cluster
                if (clusters[i].id, clusters[j].id) not in distances:
                    distances[(clusters[i].id, clusters[j].id)] = dot_product(clusters[i].vec, clusters[j].vec)

                dist = distances[(clusters[i].id, clusters[j].id)]

                # Find the closest pairs of clusters
                if dist < closest:
                    closest = dist
                    lowestpair = (i, j)

        # Create a new cluster from the two closest clusters using the average distance (merge_vector)
        new_cluster = BiCluster(
            (clusters[lowestpair[0]].vec + clusters[lowestpair[1]].vec) / 2.0,
            left=clusters[lowestpair[0]], right=clusters[lowestpair[1]],
            distance=closest, id=current_cluster_id,
        )

        # Set the cluster id to negative for clusters not contained in the original set
        current_cluster_id -= 1
        del clusters[lowestpair[1]]
        del clusters[lowestpair[0]]
        clusters.append(new_cluster)
    return clusters[0]


def divide(clusters, n):
    clusters = list(clusters)
    while len(clusters) < n:
        ix_max = np.argmax([c.distance for c in clusters])
        if clusters[ix_max].distance < 0:
            break
        clusters.append(clusters[ix_max].left)
        clusters.append(clusters[ix_max].right)
        del clusters[ix_max]
    return clusters


def hc(data, k):
    top_cluster = hcluster(data)
    clusters = divide([top_cluster], k)
    clusters_elements = [c.get_leaves_ids() for c in clusters]
    results = np.zeros(len(data), dtype=int)
    for cluster_num, leaves_ids in enumerate(clusters_elements):
        for i in leaves_ids:
            results[i] = cluster_num
    return list(results)


def compute_error(data, labels, k):
    n, d = data.shape
    centers = []
    for i in range(k):
        centers.append([0 for j in range(d)])

    for i in range(n):
        centers[labels[i]] = [x + y for x, y in zip(centers[labels[i]], data[i])]

    error = 0
    for i in range(n):
        error += dot_product(centers[labels[i]], data[i])
    return error


def plot_data(data):
    from matplotlib import pyplot as plt
    x = [r[0] for r in data]
    y = [r[1] for r in data]
    plt.plot(x, y, '.')
    plt.show()


def run():
    data = np.loadtxt('asset/data.txt', dtype=float)
    # plot_data(data)
    k = 3
    print(data)
    labels = hc(data, k)
    print(labels)
    print(compute_error(data, labels, k))

    return hcluster(data)


if __name__ == '__main__':
    tc = run()
