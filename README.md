# NLP-labs

several lab assignments in NLP class

1. Text Classification with the perceptron

使用binary perceptron和bag of words representation对电影评论文本进行情感分析，只针对正面情绪和负面情绪，数据来源地址为http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz. 

多次训练感知机，每次训练前打乱训练集，最终返回平均值，对比了bigram与trigram在这次实验中对最终结果的影响。

2. Language modelling

完成三种语言模型(bigram, trigram, bigram with add-1 smoothing)并用来填句，比如，给出一个缺失一个单词的句子，模型从两个候选单词中选择出其中最可能的一个：

I don't know \____ to go out or not . : weather/whether

数据来源地址为https://drive.google.com/file/d/1eum-7rIjNlrOo-PTaA3FxQFd9Egfs-Lm/view.

3. Named Entity Recognition with the Structured Perceptron

使用结构化感知机训练出named entity recogniser (NER)，对于句子中的每一个单词，NER都需要预测出一个标签，标签如下：

• O: not part of a named entity
• PER: part of a person’s name
• LOC: part of a location’s name
• ORG: part of an organisation’s name
• MISC: part of a name of a different type (miscellaneous, e.g. not person, location or organisation)

本次实验使用了两种特征提取方法，Current word-current label和Previous label-current label，结构化感知机算法如下：

4. Viterbi and Beam search

基于第3次实验，使用Viterbi和Beam search去加速结构化感知机的argmax部分，

5.  Neural Language Modelling

基于pytorch的词向量训练和填词练习
