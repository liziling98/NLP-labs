from collections import Counter
import sys
import itertools
import numpy as np
import time, random
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


depochs = 5
feat_red = 0

### Load the dataset
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

#load data
train_data = load_dataset_sents(sys.argv[2])
test_data = load_dataset_sents(sys.argv[3])
## unique tags
all_tags = ["O", "PER", "LOC", "ORG", "MISC"]

# feature space of cw_ct
def cw_ct_counts(data, freq_thresh = 5): # data inputted as (cur_word, cur_tag)
    cw_c1_c = Counter()
    for doc in data:
        cw_c1_c.update(Counter(doc))
    return Counter({k:v for k,v in cw_c1_c.items() if v > freq_thresh})

cw_ct_count = cw_ct_counts(train_data, freq_thresh = feat_red)

# feature representation of a sentence cw-ct
def phi_1(sent, cw_ct_counts): # sent as (cur_word, cur_tag)
    phi_1 = Counter()
    # include features only if found in feature space
    phi_1.update([item for item in sent if item in cw_ct_count.keys()])
    return phi_1

# feature space of pt-ct
def pt_ct_counts(data, freq_thresh = 5): # input (cur_word, cur_tag)
    tagtag = Counter()
    for doc in data:
        tags = list(zip(*doc))[1]
        for i in range(len(tags)):
            if i == 0:
                tagtag.update([("<s>", tags[i])])
            else:
                tagtag.update([(tags[i-1], tags[i])])
            tagtag.update([(tags[i],"</s>")])
    # return feature space with features with counts above freq_thresh
    return Counter({k:v for k,v in tagtag.items() if v > freq_thresh})

pt_ct_count = pt_ct_counts(train_data, freq_thresh = feat_red)

# combining feature spaces
comb_featspaces = pt_ct_count + cw_ct_count

# creating our sentence features
def phi_2(sent, pt_ct_count):
    sentence, tags = zip(*sent)
    tags = ["<s>"] + list(tags) +["</s>"]
    # returning features if found in the feature space
    tags = [(tags[i], tags[i+1]) for i in range(len(tags)-1) if (tags[i], tags[i+1]) in pt_ct_count]
    return Counter(tags)

class Perceptron():
    def __init__(self,all_tags):
        super(Perceptron, self).__init__()
        self.all_tags = all_tags

    # creating all possible combinaions of
    def pos_combos(self,sentence):
        combos = [list(zip(sentence, p)) for p in itertools.product(self.all_tags,repeat=len(sentence))]
        return combos

    def scoring(self,doc, weights, extra_feat = False):
        # unzippin them
        sentence, tags = list(zip(*doc))
        # all possible combos of sequences
        combos = list(enumerate(self.pos_combos(sentence)))
        # our score matrix
        scores = np.zeros(len(combos))
        # looping through all possible combos
        for index, sent_tag in combos:
            if extra_feat is False:
                # retrieving the counter if its in our feature space
                phi = phi_1(sent_tag, cw_ct_count)
            else:
                phi1 = phi_1(sent_tag, cw_ct_count)
                phi2 = phi_2(sent_tag, pt_ct_count)
                phi = phi1 + phi2
            # if its not then the score is 0
            if len(phi) == 0:
                    scores[index] = 0
            else:
                temp_score = 0
                # otherwise do the w*local_phi
                for pair in phi:
                    if pair in weights:
                        temp_score += weights[pair]*phi[pair]
                    else:
                        temp_score += 0
                # store the score with the index
                scores[index] = temp_score
        # retrieve the index of the highest scoring sequence
        max_scoring_position = np.argmax(scores)
        # retrieve the highest scoring sequence
        max_scoring_seq = combos[max_scoring_position][1]
        return max_scoring_seq

    def veterbi(self, doc, weights, extra_feat= False):
        words, tags = list(zip(*doc))
        phi = phi_1(doc,cw_ct_count)
        #build array to store scores and indexes
        scores = np.zeros(len(doc)*5).reshape(len(doc),5)
        indexs = np.zeros((len(doc)-1)*5).reshape((len(doc)-1),5).astype(int)
        #handle the first word, only concern about scores
        for i in range(5):
            pair = (words[0],all_tags[i])
            if pair in weights:
                scores[0][i] = weights[pair]# * phi[pair]
            else:
                scores[0][i] = 0
        #handle every other word
        for i in range(1,len(doc)):
            #for each word, concern about 5 scores
            for j in range(5):
                pair = (words[i],all_tags[j])
                word_tag = weights[pair]
                lis = []
                for k in range(5):
                    if pair in weights:
                        score = scores[i-1][k] + word_tag
                    else:
                        score = scores[i-1][k]
                    lis.append(score)
                #compare the 5 scores of one pair
                scores[i][j] = max(lis)
                #store the index of max into arr
                indexs[i-1][j] = np.argmax(lis)
        reverse = []
        #get the index of maximum of last word
        back_pointer = np.argmax(scores[-1])
        reverse.append(all_tags[back_pointer])
        #trace back
        for i in range(len(doc)-1):
            back_pointer = indexs[-i-1][back_pointer]
            reverse.append(all_tags[back_pointer])
        reverse.reverse()
        prediction = [(words[i],reverse[i]) for i in range(len(doc))]
        return prediction

    def beam_search(self,doc,weights,beam,extra_feat=False):
        words, tags = list(zip(*doc))
        phi = phi_1(doc,cw_ct_count)
        record_lis = []
        #handle the first word
        scores = np.zeros(beam).reshape(1,beam)
        indexs = np.zeros(beam).reshape(1,beam).astype(int)
        #compute 5 scores
        temp = np.zeros(5).reshape(1,5)
        for i in range(5):
            pair = (words[0],all_tags[i])
            if pair in weights:
                temp[0][i] = weights[pair]
            else:
                temp[0][i] = 0
        #get the index of top values
        top = (-temp[0]).argsort()[:beam]
        #choose the best choices based on beam size
        for b in range(beam):
            idx = top[b]
            scores[0][b] = temp[0][idx]
            indexs[0][b] = idx
            lis = [all_tags[idx]]
            record_lis.append(lis)
        #handle every other word
        for i in range(1,len(doc)):
            temp_score = np.zeros(5*beam).reshape(5,beam)
            for j in range(5):
                pair = (words[i],all_tags[j])
                word_tag = weights[pair]
                #compute scores based on beam size
                for k in range(beam):
                    if pair in weights:
                        score = scores[0][k] + word_tag
                    else:
                        score = scores[0][k]
                    temp_score[j][k] = score
                top = (-temp_score.ravel()).argsort()[:beam]
                #get the index of 2-d array
                i2d = np.unravel_index(top, temp_score.shape)
            temp_lis = [i for i in record_lis]
            #choose the best choices based on beam size
            for b in range(beam):
                x = i2d[1][b]
                y = i2d[0][b]
                scores[0][b] + temp_score[y][x]
                record = temp_lis[x]
                attach = [all_tags[y]]
                record_lis[b] = record + attach
        #get the index of maximum of last word
        idx = np.argmax(scores[0])
        #get the related sequence of tags
        pred_tags = record_lis[idx]
        prediction = [(words[i],pred_tags[i]) for i in range(len(doc))]
        return prediction

    def train_perceptron(self, data, epochs, shuffle = True, extra_feat = False, new_mode = False, beam_mode = False, b = 0):
        # variables used as metrics for performance and accuracy
        iterations = range(len(data)*epochs)
        false_prediction = 0
        false_predictions = []
        # initialising our weights dictionary as a counter
        # counter.update allows addition of relevant values for keys
        # a normal dictionary replaces the key-value pair
        weights = Counter()
        start = time.time()
        # multiple passes
        for epoch in range(epochs):
            false = 0
            now = time.time()
            whole_time = 0
            # going through each sentence-tag_seq pair in training_data
            # shuffling if necessary
            if shuffle == True:
                random.seed(11242)
                random.shuffle(data)
            for doc in data:
                # retrieve the highest scoring sequence
                if new_mode == False:
                    max_scoring_seq = self.scoring(doc, weights, extra_feat = extra_feat)
                else:
                    if beam_mode == False:
                        max_scoring_seq = self.veterbi(doc, weights, extra_feat = extra_feat)
                    else:
                        max_scoring_seq = self.beam_search(doc, weights, beam = b,extra_feat = extra_feat)
                        
                # if the prediction is wrong
                if max_scoring_seq != doc:
                    correct = Counter(doc)
                    # negate the sign of predicted wrong
                    predicted = Counter({k:-v for k,v in Counter(max_scoring_seq).items()})
                    # add correct
                    weights.update(correct)
                    # negate false
                    weights.update(predicted)
                    """Recording false predictions"""
                    false += 1
                    false_prediction += 1
                false_predictions.append(false_prediction)
            time_record = round(time.time() - now,2)
            whole_time += time_record
            print("Epoch: ", epoch+1, 
                  " / Time for epoch: ", time_record,
                 " / No. of false predictions: ", false)
        return weights, false_predictions, iterations,whole_time

    # testing the learned weights
    def test_perceptron(self,data, weights, extra_feat = False):
        correct_tags = []
        predicted_tags = []
        i = 0
        for doc in data:
            _, tags = list(zip(*doc))
            correct_tags.extend(tags)
            max_scoring_seq = self.scoring(doc, weights, extra_feat = extra_feat)
            _, pred_tags = list(zip(*max_scoring_seq))
            predicted_tags.extend(pred_tags)
        return correct_tags, predicted_tags

    def evaluate(self,correct_tags, predicted_tags):
        f1 = f1_score(correct_tags, predicted_tags, average='micro', labels=self.all_tags)
        print("F1 Score: ", round(f1, 5))
        return f1


# =============================================================================
# test
# =============================================================================
arg_mode = sys.argv[1]
#viterbi
if arg_mode == '-v':
    print('For Viterbi:\n')
    train_data = load_dataset_sents(sys.argv[2])
    test_data = load_dataset_sents(sys.argv[3])
    perceptron_v = Perceptron(all_tags)
    weights_v, false_predictions_v, iterations_v,time_v = perceptron_v.train_perceptron(train_data, epochs = depochs, extra_feat=False, new_mode = True,beam_mode = False)
    correct_tags, predicted_tags_v = perceptron_v.test_perceptron(test_data, weights_v, extra_feat=False)
    print("the f1 score of Viterbi:")
    f1_v = perceptron_v.evaluate(correct_tags, predicted_tags_v)

    #normal
    train_data = load_dataset_sents(sys.argv[2])
    test_data = load_dataset_sents(sys.argv[3])
    perceptron = Perceptron(all_tags)
    weights, false_predictions, iterations, time_n = perceptron.train_perceptron(train_data, epochs = depochs, extra_feat=False,new_mode = False,beam_mode = False)
    correct_tags, predicted_tags = perceptron.test_perceptron(test_data, weights, extra_feat=False)
    print("the f1 score of Lab3:")
    f1 = perceptron.evaluate(correct_tags, predicted_tags)
    #compare their f1

    if f1_v == f1:
        print('they are exactly same!')
        speed = time_n/time_v
        print('speed up',round(speed,2),'times\n')
    else:
        print('the outputs are different!\n')
elif arg_mode == '-b':
    print('For beam_search:\n')
    for i in [1,2,3]:
        print('when beam =',i,'\n')
        train_data = load_dataset_sents(sys.argv[2])
        test_data = load_dataset_sents(sys.argv[3])
        perceptron_b = Perceptron(all_tags)
        weights_b, false_predictions_b, iterations_b,time_b = perceptron_b.train_perceptron(train_data, epochs = 5, extra_feat=False,new_mode = True,beam_mode = True, b =i)
        correct_tags, predicted_tags_b = perceptron_b.test_perceptron(test_data, weights_b, extra_feat=False)
        f1_b = perceptron_b.evaluate(correct_tags, predicted_tags_b)
else:
    print('invalid mode, please try -v or -b!')
