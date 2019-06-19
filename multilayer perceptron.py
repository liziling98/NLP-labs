# -*- coding: utf-8 -*-
# Author: Robert Guthrie
# implementation: ziling li

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#build the training set
#"-" is the start symbol
test_sentence = '''
- The mathematician ran .
- The mathematician ran to the store .
- The physicist ran to the store .
- The philosopher thought about it .
- The mathematician solved the open problem .
'''.split()

# build a list of tuples for trigram
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
#build the word and index pair in a dictionary
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size,dim):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, dim,bias=False)
        self.linear2 = nn.Linear(dim,vocab_size,bias = False)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def change_param(hidden,learning_rate,epoch_time,embedding_dim,gram):
    #initialize
    CONTEXT_SIZE = gram
    EMBEDDING_DIM = embedding_dim
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE,dim=hidden)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #train the model
    for epoch in range(epoch_time):
        total_loss = torch.Tensor([0])
        for context, target in trigrams:
            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))
            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            model.zero_grad()
            # Step 3. Run the forward pass, getting log probabilities over next words
            log_probs = model(context_var)
            # Step 4. Compute your loss function. (Again, Torch wants the target word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])))
            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        losses.append(total_loss)
    return model

def predict(model,test_tri):
    '''
    compute probabilities of all choices and get the maximum value
    return the most possible word behind the trigram
    '''
    #get the output log probs of this input pair by using trained model
    context_idxs = [word_to_ix[w] for w in test_tri]
    context_var = autograd.Variable(torch.LongTensor(context_idxs))
    log_probs = model(context_var)
    #get the most possible next word
    arr = log_probs.detach().numpy()
    idx = arr.argmax()
    for k,v in word_to_ix.items():
        if idx == v:
            prediction = k
    return prediction

def test_sen(model):
    '''
    for this sentence, build trigrams and feed them into model
    for each trigram pair, predict the most possible candidate words and return it
    '''
    #build test set
    test_sen = "- The mathematician ran to the store .".split()
    test_tri = [([test_sen[i], test_sen[i + 1]], test_sen[i + 2])
                for i in range(len(test_sen) - 2)]
    #test for this sentence
    right = 0
    pred_list = []
    for context, target in test_tri:
        prediction = predict(model,context)
        pred_list.append(prediction)
        if target == prediction:
            right += 1
    #return the prediction and accuracy
    print("the prediction of the sentence in question1--'Run a Sanity check' is:",pred_list)
    print("the accuracy is:",right/len(test_tri))
    #predict for the context "START The", and return the answer
    pred_answer = predict(model,["-","The"])
    print('''prediction for the context "START The" is:''', pred_answer)

def compute_prob(model,test,word):
    '''
    test is a list contains two pairs of trigram
    and get their probability of combined with "solved"
    compare the probabilities and return the bigger one
    '''
    prob_total = 1
    target_lis = [word,"solved","the"]
    for i in range(3):
        context_idxs = [word_to_ix[w] for w in test[i]]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        log_probs = model(context_var)
        prob = -log_probs[0][word_to_ix[target_lis[i]]].item()
        prob_total *= prob
    return prob_total

#change hyper parameters and make sure the predictions are right
#build parameters list
params = [[80,0.001,1500,5,2],
          [96,0.001,1500,5,2],
          [80,0.001,1200,3,2],
          [96,0.01,500,5,2],
          [96,0.001,2000,6,2]]

print("start to change hyper-parameters...\n")

torch.manual_seed(1)
for param in params:
    #train the model
    print("when hidden_dim =",param[0], "learning_rate=",param[1],
          "epoch_time=",param[2],"embedding_dim=",param[3])
    model = change_param(hidden=param[0],learning_rate=param[1],
                         epoch_time=param[2],embedding_dim=param[3],
                         gram = param[4])

    #use the trained model to test this specific sentence
    test_sen(model)

    #test the result of the gap
    test1 = [["-","The"],["The","physicist"],["physicist","solved"]]
    test2 = [["-","The"],["The","philosopher"],["philosopher","solved"]]
    log1 = compute_prob(model,test1,"physicist")
    log2 = compute_prob(model,test2,"philosopher")
    prediction = ["physicist" if log1 < log2 else "philosopher"]
    print("the answer for question 2--'Test',the prediction of this gap is:",prediction[0])

    #compute the cosine between these role
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    w1 = model.embeddings(torch.tensor(word_to_ix["mathematician"],dtype=torch.long))
    w2 = model.embeddings(torch.tensor(word_to_ix["philosopher"],dtype=torch.long))
    w3 = model.embeddings(torch.tensor(word_to_ix["physicist"],dtype=torch.long))
    if cos(w1,w3) > cos(w1, w2):
        print("the similarity of mathematician and physicist is higher than of mathematician and philosopher\n")
    else:
        print("the similarity of mathematician and philosopher is higher than of mathematician and physicist\n")
    model.apply(weight_reset)
