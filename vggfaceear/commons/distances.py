import numpy as np
from scipy import spatial

def CosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def EuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def getChiSquaredDistance(a,b):    
    return np.sqrt(0.5*np.sum((a-b)**2/(a+b+1e-6)))

def getCosineDistance(a,b):
    return spatial.distance.cosine(a,b)

def getEuclideanDistance(a,b):
    return spatial.distance.euclidean(a,b)

def getDistance(distance, a, b):
    dist = 0.0

    if distance == 'chisquared':
        dist = np.sqrt(0.5*np.sum((a-b)**2/(a+b+0.000001)))
    elif distance == 'cosine':
        dist = spatial.distance.cosine(a,b)
    elif distance == 'euclidean':
        dist = spatial.distance.euclidean(a,b)
    elif distance == 'euclidean_l2':
        dist = spatial.distance.euclidean(l2_normalize(a),l2_normalize(b))
    else:
        print("---> Invalid distance name: {}. Available distances are: {}".format(distance, ['chisquared', 'cosine', 'euclidean', 'euclidean_l2']))

    return dist