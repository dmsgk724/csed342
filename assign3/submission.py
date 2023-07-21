import shell
import util
import wordsegUtil



############################################################
# Problem 1: Word Segmentation

# Problem 1a: Solve the word segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return ''  # remove this line before writing code
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.query  # remove this line before writing code
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        graph = []
        remain = len(self.query) - len(state)+1
        for i in range(1, remain):
            action = self.query[len(state):len(state)+i]
            next_state = self.query[:len(state)+i]
            graph.append((action,next_state,self.unigramCost(action))) 
        return graph
        # END_YOUR_ANSWER

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch()
    ucs.solve(WordSegmentationProblem(query, unigramCost))
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 1b: Solve the k-word segmentation problem under a unigram model

class KWordSegmentationProblem(util.SearchProblem):
    def __init__(self, k, query, unigramCost):
        self.k = k
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return ('',0)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[0] == self.query and state[1] ==self.k # remove this line before writing code
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        graph = []
        remain = len(self.query) - len(state[0])+1
        for i in range(1,remain):
            action = self.query[len(state[0]):len(state[0])+i]
            next_state = self.query[:len(state[0])+i]
            graph.append((action,(next_state,state[1]+1),self.unigramCost(action)))
        return graph
        # END_YOUR_ANSWER

def segmentKWords(k, query, unigramCost):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(KWordSegmentationProblem(k,query, unigramCost))
    return ' '.join(ucs.actions)  # remove this line before writing code
    # END_YOUR_ANSWER

############################################################
# Problem 2: Vowel Insertion

# Problem 2a: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (-1,wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        cur_idx, cur_word = state
        return cur_idx == len(self.queryWords)-1
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        cur_idx , cur_word = state
        nextFills = self.possibleFills(self.queryWords[cur_idx+1])
        graph = []
        if(len(nextFills)==0):
            only_next_candidate = self.queryWords[cur_idx+1]
            graph.append((only_next_candidate,(cur_idx+1,only_next_candidate),self.bigramCost(cur_word,only_next_candidate)))
        else :
            for e in nextFills:
                graph.append((e, (cur_idx+1,e),self.bigramCost(cur_word,e)))
        return graph
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(VowelInsertionProblem(queryWords,bigramCost,possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 2b: Solve the limited vowel insertion problem under a bigram cost

class LimitedVowelInsertionProblem(util.SearchProblem):
    def __init__(self, impossibleVowels, queryWords, bigramCost, possibleFills):
        self.impossibleVowels = impossibleVowels
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (-1,wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        cur_idx, cur_word = state
        return cur_idx == len(self.queryWords)-1
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 10 lines of code, but don't worry if you deviate from this)
        cur_idx , cur_word = state
        nextFills = self.possibleFills(self.queryWords[cur_idx+1])
        graph = []
        only_next_candidate = self.queryWords[cur_idx+1]
        check = True
        if(len(nextFills)==0):
            graph.append((only_next_candidate,(cur_idx+1,only_next_candidate),self.bigramCost(cur_word,only_next_candidate)))
        else :
            for e in nextFills:
                check = True
                for i in range(0,len(self.impossibleVowels)):
                    if(self.impossibleVowels[i] in e) :
                        check = False
                if(check==True):
                    graph.append((e, (cur_idx+1,e),self.bigramCost(cur_word,e)))
        if(len(graph)==0):
            graph.append((only_next_candidate,(cur_idx+1,only_next_candidate),self.bigramCost(cur_word,only_next_candidate)))
        return graph
        # END_YOUR_ANSWER

def insertLimitedVowels(impossibleVowels, queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(LimitedVowelInsertionProblem(impossibleVowels,queryWords,bigramCost,possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 3: Putting It Together

# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0,wordsegUtil.SENTENCE_BEGIN)
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        cur_idx, cur_word = state;
        return cur_idx==len(self.query);
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        cur_idx,cur_word = state
        graph = []
        remain = len(self.query) - cur_idx+1
        for i in range(1,remain):
            next_fills = self.possibleFills(self.query[cur_idx:cur_idx+i])
            #unlike problem 2( vowel free word can be a valid reconstruction) only includ output words that are
            # the reconstructions according to possiblefills
            if len(next_fills)>0:
                for w in next_fills:
                    graph.append((w,(cur_idx+i,w),self.bigramCost(cur_idx,w)))
        return graph
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 4: A* search

# Problem 4a: Define an admissible but not consistent heuristic function

class SimpleProblem(util.SearchProblem):
    def __init__(self):
        # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
        self.edgeCostpair = {'a':[('b',1),('c',1)],'b':[('d',2)],'c':[('d',3)],'d':[('e',2)]}
        # END_YOUR_ANSWER

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 'a'
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state=='e'
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
        graph = []
        for next_state,cost in self.edgeCostpair[state]:
            graph.append((next_state, next_state,cost))
            
        return graph
        # END_YOUR_ANSWER

def admissibleButInconsistentHeuristic(state):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    heuristic = {'a':3,'b':4,'c':0,'d':0,'e':0}
    return heuristic[state]
    # END_YOUR_ANSWER

# Problem 4b: Apply a heuristic function to the joint segmentation-and-insertion problem

def makeWordCost(bigramCost, wordPairs):
    """
    :param bigramCost: learned bigram cost from a training corpus
    :param wordPairs: all word pairs in the training corpus
    :returns: wordCost, which is a function from word to cost
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    cache = {}
    wordPairs=set(wordPairs)
    for a, b in wordPairs:
        cost = bigramCost(a,b)
        if b not in cache:
            cache [b] =cost
        else:
            cache[b] = min(cache[b],cost)
        

    def wordCost(x):
        return cache[x]
    
    return wordCost
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (0,)  # remove this line before writing code
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        cur_idx= state[0]
        return cur_idx==len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        cur_idx= state[0]
        graph = []
        remain = len(self.query) - cur_idx +1
        for i in range(1,remain):
            next_fills = self.possibleFills(self.query[cur_idx:cur_idx+i])
            if len(next_fills)>0:
                word,cost = min([(w,self.wordCost(w)) for w in next_fills],key = lambda x:x[1])
                graph.append((word,(cur_idx+i,),cost))
        return graph
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    dp = util.DynamicProgramming(RelaxedProblem(query,wordCost,possibleFills))
    cache = {}
    def heuristic(state):
        if state not in cache:
            cache [state] = dp(state)
        return cache[state]
    return heuristic

    # END_YOUR_ANSWER

def fastSegmentAndInsert(query, bigramCost, wordCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    problem=util.UniformCostSearch()
    problem.solve(JointSegmentationInsertionProblem(query,bigramCost,possibleFills),heuristic=makeHeuristic(query,wordCost,possibleFills))
    return ' '.join(problem.actions)
    # END_YOUR_ANSWER

############################################################

if __name__ == '__main__':
    shell.main()
