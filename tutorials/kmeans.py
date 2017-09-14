import tensorflow as tf
import math, numpy as np
import matplotlib.pyplot as plt


def plot_data(centroids, data, n_samples):
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
        samples = data[i*n_samples:(i+1)*n_samples]
        plt.scatter(samples[:,0], samples[:,1], c=colour[i], s=1)
        plt.plot(centroid[0], centroid[1], markersize=10, marker="x", color='k', mew=5)
        plt.plot(centroid[0], centroid[1], markersize=5, marker="x", color='m', mew=2)

        
def all_distances(a, b):
    diff = tf.squared_difference(tf.expand_dims(a, 0), tf.expand_dims(b,1))
    return tf.reduce_sum(diff, axis=2)
        
        
class Kmeans(object):

    def __init__(self, data, n_clusters):
        self.n_data, self.n_dim = data.shape
        self.n_clusters = n_clusters
        self.data = data
        self.v_data = tf.Variable(data)
        self.n_samples = self.n_data//self.n_clusters

    def run(self):
        tf.global_variables_initializer().run()
        initial_centroids = self.find_initial_centroids(self.n_clusters).eval()
        curr_centroids = tf.Variable(initial_centroids)
        nearest_indices = self.assign_to_nearest(curr_centroids)
        updated_centroids = self.update_centroids(nearest_indices)
        # Begin main algorithm
        tf.global_variables_initializer().run()
        c = initial_centroids
        for i in range(10):
            c2 = curr_centroids.assign(updated_centroids).eval()
            if np.allclose(c,c2): break
            c=c2
        return c2


    def find_initial_centroids(self, k):
        r_index = tf.random_uniform([1], 0, self.n_data, dtype=tf.int32)
        r = tf.expand_dims(self.v_data[tf.squeeze(r_index)], dim=1)
        initial_centroids = []
        for i in range(k):
            dist = all_distances(self.v_data, r)
            farthest_index = tf.argmax(tf.reduce_min(dist, axis=0), 0)
            farthest_point = self.v_data[tf.to_int32(farthest_index)]
            initial_centroids.append(farthest_point)
            r = tf.stack(initial_centroids)
        return r

    def choose_random_centroids(self):
        n_samples = tf.shape(v_data)[0]
        random_indices = tf.random_shuffle(tf.range(0, n_samples))
        centroid_indices = random_indices[:self.n_clusters]
        return tf.gather(self.v_data, centroid_indices)

    def assign_to_nearest(self, centroids):
        return tf.argmin(all_distances(self.v_data, centroids), 0)

    def update_centroids(self, nearest_indices):
        partitions = tf.dynamic_partition(self.v_data, tf.to_int32(nearest_indices), self.n_clusters)
        return tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0)
                                      for partition in partitions], 0)
   
        