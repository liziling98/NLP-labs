# -*- coding: utf-8 -*-
from collections import Counter
import random
from itertools import product
import sys
from sklearn.metrics import f1_score

def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)

def cw_cl_counts_wt(inFile):
    '''
    link word tag pair together and compute word-tag counts in the corpus
    '''
    word_tag = []
    for i in range(len(inFile)):
        for j in range(len(inFile[i])):
            link = inFile[i][j][0] + "_" + inFile[i][j][1]        
            word_tag.append(link)
    return Counter(word_tag)

def cw_cl_counts_tt (inFile):
    '''
    link word-tag/tag-tag pair together and compute counts in the corpus
    '''
    tag_tag = []
    for i in range(len(inFile)):
        # add start symbol
        link = '<s>' + '_' + inFile[i][0][1]
        tag_tag.append(link)
        for j in range(len(inFile[i])-1):
            link = inFile[i][j][1] + '_' + inFile[i][j+1][1]
            tag_tag.append(link)
        tag_tag.append(link)
    for i in range(len(inFile)):
        for j in range(len(inFile[i])):
            link = inFile[i][j][0] + "_" + inFile[i][j][1]        
            tag_tag.append(link)
    return Counter(tag_tag)

def add_(sentence):
    '''
    add _ between word and tag or tag and tag
    '''
    lis = []
    for item in sentence:
        lis.append(item[0] + '_' + item[1])
    return lis

def add_2(sentence):
    '''
    add _ between word and tag or tag and tag
    '''
    lis = []
    for item in sentence:
        lis.append(item[0] + '_' + item[1])
    lis.append('<s>' + '_' + sentence[0][1])
    for i in range(len(sentence)-1):
        lis.append(sentence[i][1]+'_'+sentence[i+1][1])
    return lis

def remove_(sentence):
    '''
    remove _ between word and tag or tag and tag
    '''
    lis = []
    for pair in sentence:
        split = pair[0].split('_')
        lis.append((split[0],split[1],pair[1]))
    return lis

def list_possible(length,labels):
    '''
    list all possible combination of elements in a list
    '''
    return list(product(labels, repeat=length))

def phi_1(train,dictionary):
    '''
    return all word-tag and their count in the corpus
    '''
    dic = {}
    for sen in train:
        temp_lis = add_(sen)
        # valid_word contains all word-tag pairs in corpus
        valid_word = [k for k,v in dictionary.items()]
        for item in temp_lis:
            if item in valid_word:
                if item in dic.keys():
                    dic[item] += 1
                else:
                    dic[item] = 1
    return dic

def phi_2(train,dictionary):
    '''
    return all word-tag/tag-tag and their count in the corpus
    '''
    dic = {}
    for sen in train:
        temp_lis = add_2(sen)
        # valid_word contains all word-tag and tag-tag pairs in corpus
        valid_word = [k for k,v in dictionary.items()]
        # compute word-tag count
        for item in temp_lis:
            if item in valid_word:
                if item in dic.keys():
                    dic[item] += 1
                else:
                    dic[item] = 1
        # add start symbol and compute count of tag-tag    
        head = '<s>' + '_' + sen[0][1]
        if head in dic.keys():
            dic[head] += 1
        else:
            dic[head] = 1
        # compute tag-tag count
        for i in range(len(sen)-1):
            tag_tag = sen[i][1] + sen[i+1][1]
            if tag_tag in dic.keys():
                dic[tag_tag] += 1
            else:
                dic[tag_tag] = 1
    return dic

def phi_2_modify(train):
    '''
    add start tag-tag pair of each sentence
    '''
    output = []
    for i in range(len(train)):
        temp = []
        head = '<s>' 
        temp.append(head)
        # add all tags together in a sentence
        for j in range(len(train[i])-1):
            link = train[i][j+1][1]
            temp.append(link)
        output.append(temp)
    return output

def predict_1(weight,train,count):
    '''
    predict most suitable tags for a sentence based on phi 1
    '''
    length = len(train)
    # list all possible combination of this prediction
    possibles = list_possible(length,labels)
    scores = []
    for possible in possibles:
        score = 0
        for i in range(len(possible)):
            temp = train[i][0].split('_')
            link = temp[0] + '_' + possible[i]
            # compute scores, add them together and store score in a list
            if link in weight.keys():
                # count is the term-count in the whole corpus
                if link in count.keys():
                    score += weight[link] * count[link]
        scores.append(score)
    # compute maximun score and trace back to choose the unique tag sequence
    max_value = max(scores)
    max_index = []
    for i in range(len(scores)):
        if scores[i] == max_value:
            max_index.append(i)
    random.seed(123)
    choice = random.sample(max_index,1)[0]
    y = possibles[choice]
    # link predictions and words together
    result = []
    for i in range(len(train)):
        result.append(train[i][0] + '_' + y[i])
    return result

def predict_2(weight,train,count):
    '''
    predict most suitable tags for a sentence based on phi 2
    '''
    length = len(train)
    # list all possible combination of this prediction
    possibles = list_possible(length,labels)
    scores = []
    for possible in possibles:
        score = 0
        for i in range(len(possible)):
            # compute scores by word-tag
            temp = train[i][0].split('_')
            link = temp[0] + '_' + possible[i]
            if link in weight.keys():
                # count is the term-count in the whole corpus
                if link in count.keys():
                    score += weight[link] * count[link]
        # compute score of start-tag
        head = '<s>' + '_' + possible[0]
        if head in weight.keys(): 
            if head in count.keys():
                score += weight[head] * count[head]/10000
        # compute the score of midlle tag-tag pair
        for i in range(len(possible)-1):
            link = possible[i] + '_' + possible[i+1]
            if link in weight.keys(): 
                if link in count.keys():
                    score += weight[link] * count[link]/10000
        scores.append(score)
    # compute maximun score and trace back to choose the unique tag sequence
    random.seed(122)
    max_value = max(scores)
    max_index = []
    for i in range(len(scores)):
        if scores[i] == max_value:
            max_index.append(i)
    choice = random.sample(max_index,1)[0]
    y = possibles[choice]
    result = []
    for i in range(len(train)):
        result.append(train[i][0] + '_' + y[i])
    return result

def perceptron_1(train_set,ori_label,weight,count):
    '''
    process structured perceptopn based on phi 1
    update weight for each prediction if the it is differ from corpus
    '''
    for i in range(len(train_set)):
        label = predict_1(weight,train_set[i],count)
        # update weight if prediction is differ to true label
        if label != ori_label[i]:
            # compare prediction with true label
            p = Counter(label)
            feat_diff = Counter(ori_label[i])
            feat_diff.subtract(p)
            # update weight
            for k,v in feat_diff.items():
                if k in weight.keys():
                    weight[k] += v
    return weight

def perceptron_2(train_set,ori_label,weight,weight2,count):
    '''
    process structured perceptopn based on phi 2
    update weight for each prediction if the it is differ from modified corpus
    modify corpus by add <s> for each sentence
    '''
    for i in range(len(train_set)):
        label = predict_2(weight,train_set[i],count)
        # update weight if prediction is differ to true label
        if label != ori_label[i]:
            # compare prediction with true label
            p = Counter(label)
            feat_diff = Counter(ori_label[i])
            feat_diff.subtract(p)
            # update weight
            for k,v in feat_diff.items():
                # change weight only if differ value is positive
                if k in weight.keys():
                    weight[k] += v
    # reduce the weight value of tag-tag by dividing with 100
    for k,v in weight.items():
        if k in weight2:
            weight[k] = weight[k]/10000
    return weight

def train1(training,iteration):
    '''
    it is based on phi 1
    build reproducible iteration to test the accuracy of perceptron
    return an average weight for multiple interation
    '''
    # build weight for phi 1
    cw_cl_wt_dic = cw_cl_counts_wt(training)
    weight = {k:0 for k,v in cw_cl_wt_dic.items()}
    # compute term frequency of corpus
    count = phi_1(training,cw_cl_wt_dic)
    # reproducible iteration
    random.seed(11)
    for ite in range(iteration):
        # shuffle the trainin set
        random.shuffle(training)
        true_label = []
        # build true label in the corpus
        for sentence in training:
            temp = []
            for pair in sentence:
                temp.append(pair[0] + '_' + pair[1])
            true_label.append(temp)
        # update weight
        weight = perceptron_1(training,true_label,weight,count)
    # return average weight of iteration
    average = {}
    for k,v in weight.items():
        average[k] = v / iteration
    return average

def train2(training,iteration):
    '''
    it is based on phi 2
    build reproducible iteration to test the accuracy of perceptron
    return an average weight for multiple interation
    '''
    # build weight for phi 2
    train_phi_2 = phi_2_modify(training)
    cw_cl_tt_dic = cw_cl_counts_tt(training)
    # compute term frequency of corpus
    count = phi_2(training,cw_cl_tt_dic)
    # use weight2 to contain all tag-tag pairs
    weight = {k:0 for k,v in cw_cl_tt_dic.items()}
    tag_tag = []
    for sen in training:
        tag_tag.append("<s>"+"_"+sen[0][1])
        for i in range(len(sen)-1):
            tag_tag.append(sen[i][1]+'_'+sen[i+1][1])
    weight2 = [k for k,v in Counter(tag_tag).items()]
    # reproducible iteration
    random.seed(22)
    for ite in range(iteration):
        random.shuffle(train_phi_2)
        true_label = []
        # build true label in the corpus
        for sentence in training:
            temp = []
            # contains start-tag
            temp.append('<s>'+'_'+sentence[0][0])
            # contains tag-tag and word-tag in this sentence
            for i in range(len(sentence)-1):
                temp.append(sentence[i][0] + '_' + sentence[i][1])
                temp.append(sentence[i][1] + '_' + sentence[i+1][1])
            # comtains last word-tag pair and tag-end pair
            temp.append(sentence[-1][0] + '_' + sentence[-1][1])
            true_label.append(temp)
        # update weight
        weight = perceptron_2(train_data,true_label,weight,weight2,count)
    average = {}
    for k,v in weight.items():
        average[k] = v / iteration
    return average,count

def generate_answer_1(weight,test,count):
    '''
    prediction by using weight upon testing corpus based on phi 1
    '''
    labels = []
    for i in range(len(test)):
        # predict by using weight we generated above
        label = predict_1(weight,test[i],count)
        pair = []
        for link in label:
            split = link.split('_')
            pair.append((split[0],split[1]))
        labels.append(pair)
    return labels

def generate_answer_2(weight,test,count):
    '''
    prediction by using weight upon testing corpus based on phi 2
    '''
    labels = []
    for i in range(len(test)):
        # predict by using weight we generated above
        label = predict_2(weight,test[i],count)
        pair = []
        for link in label:
            split = link.split('_')
            pair.append((split[0],split[1]))
        labels.append(pair)
    return labels

def test_1(weight,testing,count):
    '''
    compute f1 score for the prediction based on phi 1
    '''
    # get the prediction
    my_answer = generate_answer_1(weight,test_data,count)
    # change prediction and test data into array
    y_pred = []
    y_test = []
    for i in range(len(my_answer)):
        for j in range(len(my_answer[i])):
            y_pred.append(my_answer[i][j][1])
            y_test.append(test_data[i][j][1])
    # compute f1 score
    f1_micro = f1_score(y_test, y_pred, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    return f1_micro,my_answer

def test_2(weight,testing,count):
    '''
    compute f1 score for the prediction based on phi 2
    '''
    # get the prediction
    my_answer = generate_answer_2(weight,testing,count)
    # change prediction and test data into array
    y_pred = []
    y_test = []
    for i in range(len(my_answer)):
        for j in range(len(my_answer[i])):
            y_pred.append(my_answer[i][j][1])
            y_test.append(test_data[i][j][1])
    # compute f1 score
    f1_micro = f1_score(y_test, y_pred, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    return f1_micro,my_answer

# generate lable here
labels = ['O','PER','LOC','ORG','MISC']
# load data
train_file = sys.argv[1]
test_file = sys.argv[2]
train_data = load_dataset_sents(train_file)
test_data = load_dataset_sents(test_file)
# generate term-count
cw_cl_wt_dic = cw_cl_counts_wt(train_data)
cw_cl_tt_dic = cw_cl_counts_tt(train_data)
# perceptron with consideration of word-tag feature
weight1 = train1(train_data,5)
f1_1,answer1 = test_1(weight1,test_data,cw_cl_wt_dic)
# perceptron with consideration of combining word-tag and tag-tag feature
weight2,phi2_count = train2(train_data,5)
f1_2,answer2 = test_2(weight2,test_data,phi2_count)
# return top10 most fetured pair
phi_1_top_10 = sorted(weight1.items(), key = lambda item:item[1], reverse = True)[:10]
phi_2_top_10 = sorted(weight2.items(), key = lambda item:item[1], reverse = True)[:10]
# output
print("f-score of phi_1 is ",f1_1)
print("f-score of phi_2 is ",f1_2)
print("top10 of phi_1",remove_(phi_1_top_10))
print("top10 of phi_2",remove_(phi_2_top_10))

