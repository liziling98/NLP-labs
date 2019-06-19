# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:26:45 2019

@author: acv18zl
"""
import os
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter

# get the direction of files from command line
dirname = os.getcwd()
train_name = sys.argv[1]
test_name = sys.argv[2]
text_dir = os.path.join(dirname, train_name)
q_dir = os.path.join(dirname, test_name)

# extract all sentences from text
def get_train_sen(text_dir):
    '''
    read files and all sentences, add <s> and <e> for every sentence
    '''
    texts = []
    with open(text_dir, 'r', encoding = 'utf-8') as file:
        line = file.readline()
        while line:
            # remove unmeaningful punctuation
            new_line = re.sub(r"[()\"#/@;:<>{}`+=~|.!?,]", "", line)
            new_line = new_line.replace(' \n', '')
            new_line = new_line.replace('--', '')
            # add start and ending
            words = new_line.lower().split()
            start = ['<s>']
            start.extend(words)
            start.extend(['<e>'])
            texts.append(start)
            line = file.readline()
    return texts
        
def gram_prob(texts, method):
    '''
    method: unigram, bigram, bigram-smoothing
    get related item:count for each method
    '''
    text_list = []
    for word_list in texts:
        if method == 'unigram':
            text_list.extend(word_list)
        if method == 'bigram':
            for i in range(len(word_list) - 1):
                key = word_list[i+1] + '|' + word_list[i]
                text_list.append(key)
    gram_counts = Counter(text_list)
    amount = sum(gram_counts.values())
    return gram_counts,amount

def get_q(q_dir):
    '''
    read questions.txt and split every sentence into question and candidates
    '''
    with open(q_dir, 'r', encoding = 'utf-8') as file:
        text = file.read()
        questions = re.findall("[A-Z].*?[\.!?]", text, re.MULTILINE | re.DOTALL )
        two_words = re.findall("[:].*?[\n]", text, re.MULTILINE | re.DOTALL )
    # extract the candidate words
    candidates = []
    for i in range(len(two_words)):
        remove_1 = two_words[i].replace('\n','')
        remove_2 = remove_1.replace(': ', '')
        candidates.append(remove_2.split('/'))
    return questions, candidates
    
def get_q_sen(questions, candidates):
    '''
    build complete sentences for testing
    '''
    test_sen = []
    for i in range(len(candidates)):
        # add start and ending for every sentence
        temp_list = [['<s>'],['<s>']]
        for j in range(len(temp_list)):
            choose = questions[i].replace('____',candidates[i][j]).lower().split()[:-1]
            temp_list[j].extend(choose)
            temp_list[j].extend(['<e>'])
        test_sen.append(temp_list)
    return test_sen

def compare(my_answer, method, candidates):
    '''
    compare my answer to right answer, and compute the accuracy
    '''
    true_answer = ['B','A','A','B','B','B','A','B','A','B']
    index = [i for i in range(10) if true_answer[i] == my_answer[i]]
    accuracy = len(index) / len(true_answer)
    wrong_index = [l for l in [0,1,2,3,4,5,6,7,8,9] if l not in index]
    wrong_word = [candidates[i] for i in wrong_index]
    print('\nwhen method is:', method,"wrong answer:",wrong_word)
    print('\nthe accuracy is:', accuracy)
    
# build bigram testing set
def get_test_set(test_sen, candidates):
    '''
    build bigram testing items
    '''
    test = []
    set1 = []
    set2 = []
    for q in range(10):
        key_1 = []
        key_2 = []
        # get the index of the candidate word
        index = test_sen[q][0].index(candidates[q][0])
        for c in range(1,index+2):
            # build new word which is combination of candidate and '|' and one of contexts
            key_1.append(test_sen[q][0][c] + '|' + test_sen[q][0][c-1])
            key_2.append(test_sen[q][1][c] + '|' + test_sen[q][1][c-1])
        # add the new word set into a list
        test.append([key_1, key_2])
        set1.append([key_1[index], key_1[index-1]])
        set2.append([key_2[index], key_2[index-1]])
    return test, set1, set2

def testing(bigram_counts,bi_amount,candidates,test,uni_counts,uni_amount,method):
    '''
    method: unigram, bigram, bigram-smoothing
    compute the probabilities of all testing items
    '''
    p1 = [1]*10
    p2 = [1]*10
    for i in range(10):
        if method == 'unigram':
            p1[i] = uni_counts[candidates[i][0]] / uni_amount
            p2[i] = uni_counts[candidates[i][1]] / uni_amount
        if method == 'bigram':
            for item1 in test[i][0]:
                p1[i] *= bigram_counts[item1] / bi_amount
            for item2 in test[i][1]:
                p2[i] *= bigram_counts[item2] / bi_amount
        if method == 'bigram-smoothing':
            for item in test[i][0]:
                down = item.split('|')[1]
                p1[i] *= (1 + bigram_counts[item]) / (uni_counts[down]+bi_amount)
            for item in test[i][1]:
                down = item.split('|')[1]
                p2[i] *= (1 + bigram_counts[item]) / (uni_counts[down]+bi_amount)
    # evaluation
    my_answer = []
    for i in range(10):
        if p1[i] > p2[i]:
            my_answer.append('A')
        if p1[i] < p2[i]:
            my_answer.append('B')
        if p1[i] == p2[i]:
            my_answer.append('equal')
    return my_answer, p1, p2

# get text, questions and candidate words
texts = get_train_sen(text_dir)
questions, candidates = get_q(q_dir)
test_sen = get_q_sen(questions, candidates)
# get parametres
uni_counts, uni_amount = gram_prob(texts,'unigram')
bigram_counts, bi_amount = gram_prob(texts,'bigram')
# build testing set
test, set1, set2 = get_test_set(test_sen, candidates)

# unigram answer
uni_answer, p1, p2 = testing(bigram_counts,bi_amount,candidates,test,uni_counts,uni_amount,'unigram')
candidates1 = [candidates[i][0] for i in range(10)]
candidates2 = [candidates[i][1] for i in range(10)]
uni_arr = np.array([candidates1,p1,candidates2,p2,uni_answer]).T
uni_df = pd.DataFrame(uni_arr)
# bigram answer
bi_answer, p1, p2 = testing(bigram_counts,bi_amount,candidates,test,uni_counts,uni_amount,'bigram')
bi_arr = np.array([set1,p1,set2,p2,bi_answer]).T
bi_df = pd.DataFrame(bi_arr)
# bigram smoothing answer
bi_s_answer, p1, p2 = testing(bigram_counts,bi_amount,candidates,test,uni_counts,uni_amount,'bigram-smoothing')
bi_s_arr = np.array([set1,p1,set2,p2,bi_s_answer]).T
bi_s_df = pd.DataFrame(bi_s_arr)

# compare with the right answer and show
pd.set_option('display.width', 100)
print('for unigram:\n',uni_df.to_string(),'\n')
compare(uni_answer, 'unigram',candidates)
print('for bigram:\n',bi_df.to_string(), '\n')
compare(bi_answer, 'bigram',candidates)
print('for bigram smoothing:\n',bi_s_df.to_string())
compare(bi_s_answer, 'bigram-smoothing',candidates)






