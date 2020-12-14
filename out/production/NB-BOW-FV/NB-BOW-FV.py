# Naive Bayes Bag of Words Filtered Vocabulary Model
import csv
import numpy
import math
import decimal
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
    temp_list = []
    temp_list2 = []
    vocabulary = joinedsentences.split()
    print("length before removing words that appear once: " + str(len(vocabulary)))
    # adding duplicates from vocabulary to new list

    vocabulary = list(dict.fromkeys(vocabulary))
    print(len(vocabulary))

    for words in vocabulary:
        if vocabulary.count(words) > 1:
            # print(str(words) + " " + str(vocabulary.count(words)))
            temp_list.append(words)
        elif vocabulary.count(words) == 1:
            temp_list2.append(words)
    print(len(temp_list2))
    vocabulary = temp_list
    print("length after removing words that appear once: " + str(len(vocabulary)))

    # create training training matrix
    training_matrix = [[0] * len(vocabulary)] * len(sentences)

    for i in range(len(sentences)):
        words_in_sentence = sentences[i].split()
        training_matrix[i] = [0] * len(vocabulary)
        for j in range(len(vocabulary)):
            for k in range(len(words_in_sentence)):
                if vocabulary[j] == words_in_sentence[k]:
                    training_matrix[i][j] += 1

    # train NB
    clf = MultinomialNB()
    clf.class_log_prior_ = 10
    clf.fit(training_matrix, results)

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
    testing_matrix = [[0] * len(vocabulary)] * len(sentences)

    for i in range(len(sentences)):
        words_in_sentence = sentences[i].split()
        testing_matrix[i] = [0] * len(vocabulary)
        for j in range(len(vocabulary)):
            for k in range(len(words_in_sentence)):
                if vocabulary[j] == words_in_sentence[k]:
                    testing_matrix[i][j] += 1

    prediction = clf.predict(testing_matrix)

    prediction_no = []
    true_no = []

    # inverse arrays manually for every item in array that is == 1 then append a 0 and viceversa for both prediction
    # and result
    for i in range(len(prediction)):
        if prediction[i] == 1:
            prediction_no.append(0)
        else:
            prediction_no.append(1)

    for i in range(len(results)):
        if results[i] == 1:
            true_no.append(0)
        else:
            true_no.append(1)

    # find precision, recall, and f1 for yes and no
    precision_yes = precision_score(results, prediction)
    precision_no = precision_score(true_no, prediction_no)
    recall_yes = recall_score(results, prediction)
    recall_no = recall_score(true_no, prediction_no)
    f1_yes = f1_score(results, prediction)
    f1_no = f1_score(true_no, prediction_no)

    prediction_prob = clf.predict_proba(testing_matrix)
    score = clf.score(testing_matrix, results)
    accuracy = accuracy_score(prediction, results)

    print("\n-- Score --")
    print(score)
    print("\n-- Accuracy --")
    print(accuracy)

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

    f = open("trace_NB-BOW-FV.txt", "a")
    for i in range(len(prediction)):
        predict_prob = prediction_prob[i]
        output_sentence = str(ids[i]) + "  " + str(word_prediction[i]) + "  " + str(predict_prob[0]) + "  " + str(
            results[i]) + "  " + str(correct_or_wrong[i]) + "\n"
        f.write(output_sentence)
    f.close()

    # Opening output file and printing evaluation file
    f = open("eval_NB-BOW-FV.txt", "a")
    f.write(str(accuracy) + "\r")
    f.write(str(precision_yes) + "  " + str(precision_no) + "\r")
    f.write(str(recall_yes) + "  " + str(recall_no) + "\r")
    f.write(str(f1_yes) + "  " + str(f1_no) + "\r")
    f.close()

def main():
    filtered_vocabulary()


if __name__ == "__main__":
    main()
