import os
import re
import collections
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# read files
# =============================================================================
def get_file_dir(rootDir): 
    '''
    get the dir of files and return path
    '''
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        if os.path.isdir(path): 
            get_file_dir(path)
    return path

# get the dir of data
dirname, py_name = os.path.split(os.path.abspath(__file__))
data_name = sys.argv[1]
file_dir =str(dirname) + '\\' + str(data_name)
data_dir = get_file_dir(file_dir)

# read all documents and get what the algorithm need
txt_path_list = []
doc_list = []
for path,dir_list,file_list in os.walk(data_dir):  
    for file_name in file_list:  
        txt_path = os.path.join(path, file_name)
        # get all higher path of .txt and them get unique pathby using set()
        txt_path_list.append(txt_path.split('\\cv')[0])
        # get all text filenames
        doc_list.append(txt_path.split('\\')[-2:])
# data_dir are a list contains pos file path and neg file path
data_dir = sorted(list(set(txt_path_list)))
neg_dir = data_dir[0]
pos_dir = data_dir[1]

# =============================================================================
# function module
# =============================================================================
def get_file_name(doc_list):  
    '''
    store document names seperately in pos and neg lists
    '''
    pos_list = [item[1] for item in doc_list if item[0] == 'pos']
    neg_list = [item[1] for item in doc_list if item[0] == 'neg']
    return pos_list, neg_list

def split_dataset(doc_list):
    '''
    get the training files and test files
    '''
    train_list = doc_list[:int(0.8*len(doc_list))]
    test_list = doc_list[-int(0.2*len(doc_list)):]
    return train_list, test_list

def get_text_from_txt(file_dir, name):
    '''
    get text of single .txt and store all strings in a long string
    '''
    word_str = ''
    with open(file_dir + '\\'+ name, 'r') as file:
        line = file.readline()
        while line:
            word_str += line.lower()
            line = file.readline()
    return word_str

def get_text_from_set(file_list,file_dir):
    '''
    get the text from all .txt in the file_list, file_dir path contains .txt files
    '''
    set_text = ''
    for name in file_list:
        text = get_text_from_txt(file_dir,name)
        set_text += text
    return set_text

def get_bow(file_set,file_dir,gram):
    '''
    get the bag of words of files in file_set, if gram = 1, we use unigram, etc...
    '''
    import numpy as np
    bow_dic = {}
    for name in file_set:
        text = get_text_from_txt(file_dir, name)
        text = re.sub("[^\w']"," ",text).split()
        bigram = []
        for i in range(len(text) - (gram - 1)):
            temp_txt = ''
            # build bag of words
            for j in np.arange(0,gram):
                temp_txt = temp_txt + text[i+j] + ' '
            bigram.append(temp_txt)
        bow_dic[name] = collections.Counter(bigram)
    return bow_dic
    
def generate_label(shuffle_list, pos_train):
    '''
    judge if the file is contained by posfiles, if yes label = 1, or label = -1
    '''
    return [1 if docid in pos_train else -1 for docid in shuffle_list]

def my_proceptron(epoch,train_bow,text_tf,pos_train,train_set):
    '''
    train_bow contains filenames in training set and their bag of words
    text_tf.keys() contains all featrued word
    train_set is a list contains all training filenames
    '''
    # initialize average weight with 0
    text_weight = {k:[0]*epoch for k,v in text_tf.items()}
    for i in range(epoch):
        # shuffle training set
        random.seed(482)
        random.shuffle(train_set)
        right_list = []
        # geneate label for this iteration
        true_label = generate_label(train_set, pos_train)
        # initialize weight used for this itration and pre process
        j = 0
        for docid in train_set:
            score = 0.0
            for word, count in train_bow[docid].items():
                if word in text_weight.keys():
                    score += count * text_weight[word][i]
            if score * true_label[j] <= 0:
                right_list.append(0)
                for word, count in train_bow[docid].items():
                    if word in text_weight.keys():
                        text_weight[word][i] += true_label[j] * count
            else:
                right_list.append(1)
            j += 1
    # get the average weight
    text_weight = {k:sum(v)/epoch for k,v in text_weight.items()}
    return text_weight, right_list

def testing(test_bow, text_weight, pos_test):
    '''
    ust the weight we got to predict whether the file is positive or not
    '''
    # generate testing labels
    test_label = generate_label(test_bow, pos_test)
    g = 0
    right = 0
    TP = 0
    for docid,bow in test_bow.items():
        score = 0.0
        for word, counts in bow.items():
            if word in text_weight.keys():
                score += counts * text_weight[word]
        if score * test_label[g] > 0:
            right += 1
            if score > 0:
                TP += 1
        g += 1
    # TP + FP = len(test_bow) = 400
    precision = right / len(test_bow) 
    # there are actually 200 pos files
    recall = TP / 200 
    F1 = 2 * precision * recall / (precision + recall)
    return precision, round(F1,4)
    
def plot_learningprogress(right_list):
    '''
    plot the learning progress in the last iteration
    '''
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111) 
    x= [i for i in range(21,1601)]
    # generate cumulative correct rate
    y = [np.mean(right_list[:i+1]) for i in range(len(right_list))][20:]
    ax.plot(x, y, 'k--')
    ax.set_xlabel('training set')
    ax.set_ylabel('cumulative accuracy')
    plt.title('bigram')
    
# =============================================================================
# process module
# =============================================================================
# get both pos and neg document names and split into training and testing set
pos_doc_list, neg_doc_list = get_file_name(doc_list)
pos_train, pos_test = split_dataset(pos_doc_list)
neg_train, neg_test = split_dataset(neg_doc_list)

# main process
grams = [1,2,3]
epoch = 20
for gram in grams:
    # build a list to contain all filename of training set
    train_set = sorted(list(set(pos_train + neg_train)))
    # get the bag of words of training set and testing set
    pos_train_bow = get_bow(pos_train, pos_dir, gram)
    neg_train_bow = get_bow(neg_train, neg_dir, gram)
    pos_test_bow = get_bow(pos_test, pos_dir, gram)
    neg_test_bow = get_bow(neg_test, neg_dir, gram)
    train_bow = dict(pos_train_bow, **neg_train_bow)
    test_bow = dict(pos_test_bow, **neg_test_bow)
    # get bow of features from whole training texts and words' df value(tfidf)
    temp_dic = {}
    df = {}
    for docid,bow in train_bow.items():
        for word,count in bow.items():
            if word in temp_dic.keys():
                df[word] += 1
                temp_dic[word] += count
            else:
                df[word] = 1
                temp_dic[word] = count
    # extract inportant features from whole words, remove word when df >= 800
    text_tf = {k:v for k,v in temp_dic.items() if v>1 and df[k] < 800}
    # do algorithm process
    print('when gram = ', gram)
    text_weight, right_list = my_proceptron(epoch,train_bow,text_tf,pos_train,train_set)
    # testing
    accuracy, F1 = testing(test_bow, text_weight, pos_test)
    print('the testing accuracy = ',accuracy)
    print('F-meature:', F1)
    # most positively featured feature and their weight
    text_weight = sorted(text_weight.items(), key=lambda x: x[1], reverse=True)
    print('the most positive featured features:', text_weight[:10])
    print('the most negative featured features:', text_weight[-10:])
    # plot the learning progress when using bigram features
    if gram == 2:
        plot_learningprogress(right_list)
plt.show()
