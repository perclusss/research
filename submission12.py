## import modules here 
import numpy as np

def similarity(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return float(res)

def findDist(list1, list2, data):
    minsimi = float('inf')
    for i in range(len(list1)):
        a = list1[i]
        for j in range(len(list2)):
            b = list2[j]
            value = similarity(data[a], data[b])
            if (value < minsimi):
                minsimi = value
    return minsimi

def link(cluster, distance, k, data):
    number_of_clusters = len(cluster)
##    print(cluster)
##    for i in distance:
##        for j in i:
##            print(j, end = " ")
##        print('\n')
    if (number_of_clusters == k):
        result = []
        for i in range(len(data)):
            result.append(0)
        for i in range(number_of_clusters):
            for j in range(len(cluster[i])):
                result[cluster[i][j]] = i
        
        return result
    
    else:
        maximum = 0
        name1 = 0
        name2 = 0
##        print("number_of_clusters", number_of_clusters)
        for i in range(0, number_of_clusters):
            for j in range(i, number_of_clusters):
                if (distance[i][j] > maximum) and (distance[i][j] < 1.0):
                    name1 = i
                    name2 = j
##                    print(name1, name2)
##                    print(distance[i][j])
                    maximum = distance[i][j]


        new_cluster = cluster
        
        new_cluster[name2].append(name1)
        new_cluster.pop(name1)

        new_distance = []
        temp = []
        for i in range(0, len(new_cluster)):
            for j in range(0, len(new_cluster)):
                if (i == j): temp.append(float(1))
                else: temp.append(0)
            new_distance.append(temp)
            temp = []

        for i in range(0, len(new_cluster)):
            list1 = new_cluster[i]
            for j in range(i+1, len(new_cluster)):
                list2 = new_cluster[j]
                value = round(findDist(list1, list2, data), 5)
                new_distance[i][j] = value
                new_distance[j][i] = value
                
        return link(new_cluster, new_distance, k, data)
    
################# Question 1 #################

def hc(data, k):# do not change the heading of the function
    original_cluster = []
    for i in range(len(data)):
        temp = []
        temp.append(i)
        original_cluster.append(temp)

    number_of_vectors = len(data)
    original_distance = []
    temp = []
    for i in range(0, number_of_vectors):
        for j in range(0, number_of_vectors):
            temp.append(round(similarity(data[i], data[j]), 5))
        original_distance.append(temp)
        temp = []
##
##    for i in original_distance:
##        for j in i:
##            print(round(j,3), end = " ")
##        print('\n')
        

    return link(original_cluster, original_distance, k, data)
