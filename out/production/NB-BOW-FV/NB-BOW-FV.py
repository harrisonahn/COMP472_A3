# Naive Bayes Bag of Words Filtered Vocabulary Model
import csv
import numpy
import math
import decimal
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, f1_score, precision_score


def filtered_vocabulary():
    # input input list of words in training/test set
    with open('covid_training.tsv', encoding="utf8") as f:
        content = f.readlines()

    sentences = []
    results = []
    vocabulary = []
    for i in range(len(content)):
        temp = content[i].split("\t")
        sentences.append(temp[1])
        results.append(temp[2])

    f.close()

    # removing first results since it's an index title
    results.remove(results[0])

    # change result list to binary, no = 0, yes = 1
    for i in range(len(results)):
        if results[i] == 'yes':
            results[i] = 1
        if results[i] == 'no':
            results[i] = 0

    # making all words in sentences lower case
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()

    # removing first sentence since it's an index title
    sentences.remove(sentences[0])

    joinedsentences = ' '.join(map(str, sentences))
    # adding every word to a list
    vocabulary = joinedsentences.split()
    print("length before removing duplicates: " + str(len(vocabulary)))
    # removing duplicates from vocabulary
    vocabulary = list(dict.fromkeys(vocabulary))
    print("length after removing duplicates: " + str(len(vocabulary)))

    # create training training matrix
    trainingmatrix = [[0] * len(vocabulary)] * len(sentences)

    for i in range(len(sentences)):
        wordsinsentence = sentences[i].split()
        trainingmatrix[i] = [0] * len(vocabulary)
        for j in range(len(vocabulary)):
            for k in range(len(wordsinsentence)):
                if vocabulary[j] == wordsinsentence[k]:
                    # print("Found match")
                    # print("sentence index")
                    # print(i)
                    # print("vocabulary index")
                    # print(j)
                    # print(vocabulary[j])
                    # print(wordsinsentence[k])
                    trainingmatrix[i][j] += 1

    # add duplicate words to new list and add new list to dictionary
    temp_list = trainingmatrix
    for i in range(len(sentences)):
        for j in range(len(vocabulary)):
            if trainingmatrix[i][j] == 1:
                print(len(trainingmatrix[i][j]))
                temp_list.remove(trainingmatrix[i][j])
    print(temp_list)
    print(len(temp_list))

    # train NB
    clf = MultinomialNB()
    clf.class_log_prior_ = 10
    clf.fit(trainingmatrix, results)


    # open test file
    with open('covid_test_public.tsv', encoding="utf8") as t:
        content = t.readlines()
    ids = []
    sentences = []
    results = []
    for i in range(len(content)):
        temp = content[i].split("\t")
        ids.append(temp[0])
        sentences.append(temp[1])
        results.append(temp[2])

    t.close()

    # removing first result since it's an index title
    ids.remove(ids[0])
    results.remove(results[0])

    # change result list to binary , yes = 1 and no = 0
    for i in range(len(results)):
        if results[i] == 'yes':
            results[i] = 1
        if results[i] == 'no':
            results[i] = 0

    # making all words in sentences lower case
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()

    # removing the first sentence since its an index title
    sentences.remove(sentences[0])

    # create testing matrix
    testingmatrix = [[0] * len(vocabulary)] * len(sentences)

    for i in range(len(sentences)):
        wordsinsentence = sentences[i].split()
        testingmatrix[i] = [0] * len(vocabulary)
        for j in range(len(vocabulary)):
            for k in range(len(wordsinsentence)):
                if vocabulary[j] == wordsinsentence[k]:
                    # print("Found match")
                    # print("sentence index")
                    # print(i)
                    # print("vocabulary index")
                    # print(j)
                    # print(vocabulary[j])
                    # print(wordsinsentence[k])
                    testingmatrix[i][j] += 1

    prediction = clf.predict(testingmatrix)
    predictprob = clf.predict_proba(testingmatrix)
    correct_or_wrong = []

    for i in range(len(prediction)):
        if prediction[i] == results[i]:
            correct_or_wrong.append('correct')
        else:
            correct_or_wrong.append('wrong')

    length = len(prediction)
    word_prediction = [[0]] * length
    for i in range(len(prediction)):
        if prediction[i] == int(1):
            word_prediction[i] = 'yes'
        if prediction[i] == int(0):
            word_prediction[i] = 'no'

    for i in range(len(results)):
        if results[i] == 1:
            results[i] = str('yes')
        if results[i] == 0:
            results[i] = str('no')

    print("predictions " + str(word_prediction))
    print("results " + str(results))

    accuracy = clf.score(testingmatrix, results)
    per_class_precision = precision_score(results, word_prediction)
    per_class_recall = recall_score(results, word_prediction)
    per_class_f1 = f1_score(results, accuracy)
    print("score = " + str(accuracy))
    print("per_class_precision = " + str(per_class_precision))
    print("per_class_recall = " + str(per_class_recall))
    print("per_class_f1 = " + str(per_class_f1))

    f = open("trace_NB-BOW-FV.txt", "a")
    for i in range(len(prediction)):
        outputsentence = str(ids[i]) + "  " + str(word_prediction[i]) + "  " + str(predictprob[i]) + "  " + str(
            results[i]) + "  " + str(correct_or_wrong[i]) + "\n"
        f.write(outputsentence)
    f.close()

    f = open("eval_NB-BOW-FV.txt", "a")



def main():
    filtered_vocabulary()


if __name__ == "__main__":
    main()
