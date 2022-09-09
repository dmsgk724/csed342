def countMutatedSentences(sentence):
    # BEGIN_YOUR_ANSWER (our solution is 17 lines of code, but don't worry if you deviate from this)
    words=sentence.split()
    word_set=set(words) #중복없는 word

    pairs={word:set() for word in word_set}
    
    #pairs[cur_word]-> 뒤에 갈 수 있는 Pair
    
    dp=[{j:None for j in word_set} for i in range(len(words))]
    # dp[cur_len][cur_word]
    
    for prev,cur in zip(words[:-1],words[1:]):
        pairs[cur].add(prev)

    for word in word_set:
        dp[0][word]=1
    
    for x in range(1, len(words)):
        for word in word_set:
            for e in pairs[word]: ## 현재 word앞에 있는 prev array
                if(x!=0):
                    if(dp[x][word]==None) :
                        dp[x][word]=0
                    dp[x][word]=dp[x][word]+dp[x-1][e]
    res=0
    for word in word_set:
        if(dp[len(words)-1][word]==None) :
            continue
        else :
            res+=dp[len(words)-1][word]

    return res

#1 1
print(countMutatedSentences('a a a a a'))
print(countMutatedSentences('the cat'))
print(countMutatedSentences('the cat and the mouse'))