import numpy as np
import math
import collections

num_d_squared_centers = 20
k = 200
coreset_size = 1000 - num_d_squared_centers
alpha = 16 * (math.log(k, 2) + 1)
iterations = 30


def mapper(key, value):
    # key: None
    # value: one line of input file
    d_squared_centers = list()
    d_squared_centers.append(value[np.random.randint(10000)])

    def squared_distance_to_nearest_center(vec):
        return distance_to_nearest_center(vec, d_squared_centers)**2

    #print("getting D^2 centers")
    for i in xrange(num_d_squared_centers - 1):
        #print("getting the " + str(i) + " center")
        distances = np.apply_along_axis(squared_distance_to_nearest_center, 1, value)
        next_center_idx = np.random.choice(10000, p=(distances / np.sum(distances)))
        d_squared_centers.append(value[next_center_idx])

    '''print(len(centers))
    print(str(centers))
    print(type(centers))'''

    # now we do significance sampling

    distances_sum = np.sum(np.apply_along_axis(squared_distance_to_nearest_center, 1, value))

    # get a map from (center -> list(vector))

    def vec_to_nearest_center_and_squared_distance_tuple(vec):
        min_dist = np.iinfo(np.int32).max
        ret_center = None
        for center in d_squared_centers:
            dist = np.linalg.norm(vec - center)
            if dist < min_dist:
                min_dist = dist
                ret_center = center

        return ret_center, min_dist**2

    def centers_to_nearest_point_dists():
        ret = collections.defaultdict(list)
        for j in range(value.shape[0]):
            #print("looking at " + str(j))
            center, dist = vec_to_nearest_center_and_squared_distance_tuple(value[j])
            ret[center.tostring()].append(dist)

        '''for center, dist in tuples:
            #print("looking at dist " + str(dist))
            ret[str(center)].append(dist)'''

        return ret

    #print("getting centers to dists")
    centers_to_dists = centers_to_nearest_point_dists()
    #print("size: " + str(len(centers_to_dists.keys())))

    def significance_sampling_prob(vec):
        c = (1.0 / 10000.0) * distances_sum
        nearest_center, squared_dist_to_nearest_center = vec_to_nearest_center_and_squared_distance_tuple(vec)
        dists_to_nearest_center = centers_to_dists[nearest_center.tostring()]

        first_term = alpha * squared_dist_to_nearest_center / c
        #print("first term: " + str(first_term))
        second_term = 2.0 * alpha * sum(dists_to_nearest_center) / (len(dists_to_nearest_center) * c)
        #print("second term: " + str(second_term))
        third_term = 4.0 * 10000.0 / len(dists_to_nearest_center)
        #print("third term: " + str(third_term))

        return first_term + second_term + third_term

    #print('getting significance sampling prob')
    #print(str(significance_sampling_prob(value[0])))

    #print("sampling for other vectors")
    probs = np.apply_along_axis(significance_sampling_prob, 1, value)
    sampled_indices = np.random.choice(10000, size=coreset_size, p=(probs / np.sum(probs)))

    #print("adding sampled vectors to return list")
    for index in sampled_indices:
        d_squared_centers.append(value[index])


    yield 0, d_squared_centers  # this is how you yield a key, value pair


def distance_to_nearest_center(vec, centers):
    min_dist = np.iinfo(np.int32).max
    for center in centers:
        dist = np.linalg.norm(vec - center)
    if dist < min_dist:
        min_dist = dist

    return min_dist


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    #print("in mapper")

    # time to do k-means

    centers = list()
    center_indices = np.random.choice(len(values), k)

    for idx in center_indices:
        centers.append(values[idx])

    #print("original centers length " + str(len(centers)))

    def vec_to_nearest_center(vec):
        min_dist = np.iinfo(np.int32).max
        ret_center = None
        for cntr in centers:
            dist = np.linalg.norm(vec - cntr)
            if dist < min_dist:
                min_dist = dist
                ret_center = cntr

        return ret_center


    def centers_to_nearest_point_index():
        ret = collections.defaultdict(list)
        for j in range(values.shape[0]):
            #print("looking at " + str(j))
            cntr = vec_to_nearest_center(values[j])
            ret[cntr.tostring()].append(j)

        '''for center, dist in tuples:
            #print("looking at dist " + str(dist))
            ret[str(center)].append(dist)'''

        return ret

    for i in range(1, iterations + 1):
        #print("on iteration " + str(i))
        #print("getting centers to dists")
        centers_to_indices = centers_to_nearest_point_index()
        #print("centers_to_indices length " + str(len(centers_to_indices.keys())))

        x = 0
        for value in centers_to_indices.itervalues():
            x = x + len(value)

        #print("num values " + str(x))


        new_centers = list()
        for center in centers_to_indices.keys():
            sum = np.zeros(250)
            for idx in centers_to_indices[center]:
                sum = sum + values[idx]

            mean_vec = sum / len(centers_to_indices[center])
            new_centers.append(mean_vec)

        if len(new_centers) < k:
            for _ in xrange(k - len(new_centers)):
                new_centers.append(values[np.random.randint(values.shape[0])])

        centers = new_centers

    #print(len(centers))
    yield centers
