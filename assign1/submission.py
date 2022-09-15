import collections
import math

############################################################
# Problem 1
def problem_1a():
    """
    return a number between 0 and 10
    """
    # BEGIN_YOUR_ANSWER
    return 1/2
    # END_YOUR_ANSWER

def problem_1b():
    """
    return a number between 0 and 1
    """
    # BEGIN_YOUR_ANSWER
    #5/p=3/1-p
    #5(1-p)=3p
    #5-5p=3p
    #8p=5
    return 5/8
    # END_YOUR_ANSWER

def problem_1c():
    """
    return one of [1, 2, 3, 4, 5]
    """
    # BEGIN_YOUR_ANSWER
    return 5
    # END_YOUR_ANSWER

############################################################
# Problem 2a

def getLongestWord(text):
    """
    Given a string |text|, return the longest word in |text|. 
    If there are ties, choose the word that comes first in the alphabet.
    
    For example:
    >>> text = "tiger cat dog horse panda"
    >>> getLongestWord(text) # 'horse'
    
    Note:
    - Assume there is no punctuation and no capital letters.
    
    Hint:
    - max/min function returns the maximum/minimum item with respect to the key argument.
    """

    # BEGIN_YOUR_ANSWER (our solution is 4 line of code, but don't worry if you deviate from this)
    ret=[]
    res=0
    for text_ in text.split():
        if(len(text_)>res):
            res=len(text_)
            ret.clear()
            ret.append(text_)
        elif(len(text_)==res):
            ret.append(text_)    
    return min(ret)
    # END_YOUR_ANSWER
    
############################################################
# Problem 2b

def manhattanDistance(loc1, loc2):
    """
    Return the generalized manhattan distance between two locations,
    where the locations are tuples of numbers.
    The distance is the sum of differences of all corresponding elements between two tuples.

    For exapmle:
    >>> loc1 = (2, 4, 5)
    >>> loc2 = (-1, 3, 6)
    >>> manhattanDistance(loc1, loc2)  # 5

    You can exploit sum, abs, zip functions and a generator to implement it as one line code!
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 line of code, but don't worry if you deviate from this)
    return  sum([abs(lo2-loc1[i]) for i,lo2 in enumerate(loc2)])
    # END_YOUR_ANSWER

############################################################
# Problem 2c

def euclideanDistance(loc1, loc2):
    """
    Return the generalized euclidean distance between two locations,
    where the locations are tuples of numbers.
    The distance is the length of a line segment between the two tuples on the euclidean space.

    For exapmle:
    >>> loc1 = (4, 4, 11)
    >>> loc2 = (1, -2, 5)
    >>> euclideanDistance(loc1, loc2)  # 9

    You can exploit math library to implement it as one line code without any other library!
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return math.sqrt(sum([math.pow(lo2-loc1[i],2) for i,lo2 in enumerate(loc2)]))
    # END_YOUR_ANSWER

############################################################
# Problem 2d

def countMutatedSentences(sentence):
    """
    Given a sentence (sequence of words), return the number of all possible
    mutated sentences of the same length, where each pair of adjacent words
    in the mutated sentences also occurs in the original sentence.

    For example:
    >>> countMutatedSentences('the cat and the mouse')  # 4

    where 4 possible mutated sentences exist:
    - 'and the cat and the'
    - 'the cat and the mouse'
    - 'the cat and the cat'
    - 'cat and the cat and'

    which consist of the following adjacent word pairs:
    - the cat
    - cat and
    - and the
    - the mouse

    Notes:
    - You don't need to generate actual mutated sentences.
    - You should apply dynamic programming for efficiency.
    """
    # BEGIN_YOUR_ANSWER (our solution is 17 lines of code, but don't worry if you deviate from this)
    words=sentence.split()
    word_set=set(words) #중복없는 word
    pairs={word:set() for word in word_set}
    #pairs[cur_word]-> prev,,
       
    dp=[{j:0 for j in word_set} for i in range(len(words))]
    # dp[cur_len][cur_word]
    
    for prev,cur in zip(words[:-1],words[1:]):
        pairs[cur].add(prev)

    for word in word_set:
        dp[0][word]=1

    for x in range(1, len(words)):
        for word in word_set:
            for e in pairs[word]: ## 현재 word앞에 있는 prev array
                dp[x][word]=dp[x][word]+dp[x-1][e]
   
    return sum(dp[len(words)-1][word] for word in word_set)
    



   
    # END_YOUR_ANSWER

############################################################
# Problem 2e

def sparseVectorDotProduct(v1, v2):
    """
    Given two sparse vectors |v1| and |v2|, each represented as collection.defaultdict(float), return
    return their dot product.
    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return sum([v1_*v2[k] for k,v1_ in v1.items()])
    # END_YOUR_ANSWER

############################################################
# Problem 2f

def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in set(list(v1.keys())+list(v2.keys())):
        v1[key]=v1[key]+scale*v2[key]
    # END_YOUR_ANSWER

############################################################
# Problem 2g

def computeMostFrequentWord(text):
    """
    Splits the string |text| by whitespace and returns two things as a pair: 
    the set of words that occur the maximum number of times, and their count
    i.e. (set of words that occur the most number of times, that maximum number/count)
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    myDict=collections.defaultdict(int)
    for text_ in text.split():
        myDict[text_]+=1  
    ret=[word for word,v in myDict.items() if max(myDict.values())==v]
    return set(ret), max(myDict.values())



    # END_YOUR_ANSWER
