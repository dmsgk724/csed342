#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, touching, quite, impressive, not, boring
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'so':1, 'touching':1, 'quite':0,'impressive':0,'not':-1,'boring':-1} # remove this line before writing code
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    texts = x.split();
    dict = {}
    for text in texts:
        if text in dict :
            dict[text]+=1
        else :
            dict[text]= 1
    return dict
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    def NLL_grad(phi, y):
        if y ==1:
            return -(1-sigmoid (dotProduct(weights,phi))) 
        elif y==-1:
            return sigmoid(dotProduct(weights,phi)) 
    

    for i in range(numIters):
        for x, y in trainExamples:
            feature = featureExtractor(x) # sparse vector
            increment(weights,-eta*NLL_grad(feature,y),feature)
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: ngram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER (our solution is 12 lines of code, but don't worry if you deviate from this)
    texts = x.split()
    phi={}
    for i in range (len(texts)-(n-1)):
        temp = ''
        for j,st in enumerate(texts[i:i+n]):
            if j==0 :temp+=st
            else : temp += ' '+st

        if(temp in phi) :
            phi[temp]+=1
        else :
            phi[temp]=1

    # END_YOUR_ANSWER
    return phi

############################################################
# Problem 3a: k-means exercise
############################################################

def problem_3a_1():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -2, 'mu_y': 0}, {'mu_x': 3, 'mu_y': 0})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ({'mu_x': -1/2, 'mu_y':3/2},{'mu_x': 3, 'mu_y':3/2})
    # END_YOUR_ANSWER

def problem_3a_2():
    """
    Return two centers which are 2-dimensional vectors whose keys are 'mu_x' and 'mu_y'.
    Assume the initial centers are
    ({'mu_x': -1, 'mu_y': -1}, {'mu_x': 2, 'mu_y': 3})
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ({'mu_x': -1/2, 'mu_y':3/2},{'mu_x': 3, 'mu_y':3/2})

    # END_YOUR_ANSWER

############################################################
# Problem 3: k-means implementation
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_ANSWER (our solution is 40 lines of code, but don't worry if you deviate from this)
    centroids = random.sample(examples,K) # centroids
    assignments = [None]*len(examples) # assignments list 
    cache_centroid={} # centroid feature square 
    cache_example={} # cache_example feature sqaure
    final_reconstruction_loss=-2
    prev_loss=-1
   
    def distance (center_idx, example_idx):
        # |X-Y|^2
        return abs(cache_centroid[center_idx]+cache_example[example_idx]-2*dotProduct(centroids[center_idx],examples[example_idx]))
    

    for i,e in enumerate(examples):
        cache_example[i]=dotProduct(e,e)
    

    for _ in range(maxIters):
        if(prev_loss==final_reconstruction_loss) :
            break # Complete convergence
        
        prev_loss = final_reconstruction_loss
        total_loss = 0

        for k in range(K):
            cache_centroid[k]=dotProduct(centroids[k],centroids[k])

        for j, e in enumerate(examples):
            loss_list = [distance(k,j) for k in range(K)]
            loss = min(loss_list)
            assignments[j] = loss_list.index(loss)
            total_loss+=loss # 각각의 example에 대한 loss를 더함
        
        final_reconstruction_loss=total_loss

        
        for k in range(K):
            sum = {}
            k_list = [e for i,e in enumerate(examples) if assignments[i]==k]
            for elem in k_list:
                increment(sum, 1, elem)
            centroids[k]={key:value/len(k_list) for key, value in sum.items()}
        


    return centroids, assignments,final_reconstruction_loss
    # END_YOUR_ANSWER

