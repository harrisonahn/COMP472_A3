import csv
import numpy
import math
import decimal
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score





with open('covid_training.tsv',encoding="utf8") as f:
    content = f.readlines()

sentences = []
results = []
vocabulary = []
for i in range(len(content)):
    temp = content[i].split("\t")
    sentences.append(temp[1])
    results.append(temp[2])

f.close()

# removing first result since its an index title
results.remove(results[0])

# change result list to binary , yes = 1 and no = 0
for i in range(len(results)):
    if results[i] == 'yes':
        results[i] = 1
    if results[i] == 'no':
        results[i] =0

# making all words in sentences lower case
for i in range(len(sentences)):
    sentences[i] = sentences[i].lower()

# removing the first sentence since its an index title
sentences.remove(sentences[0])

joinedsentences = ' '.join(map(str,sentences))
# adding every word to a list
vocabulary = joinedsentences.split()
print("length before removing duplicates")
print(len(vocabulary))
# removing duplicates from vocabulary 
vocabulary = list(dict.fromkeys(vocabulary))
print("length after removing duplicates")
print(len(vocabulary))


#create training training matrix
trainingmatrix = [[0]*len(vocabulary)]*len(sentences)


for i in range(len(sentences)):
    wordsinsentence = sentences[i].split()
    trainingmatrix[i] = [0]*len(vocabulary)
    for j in range(len(vocabulary)):
        for k in range(len(wordsinsentence)):
            if vocabulary[j] == wordsinsentence[k]:
                #print("Found match")
                #print("sentence index")
                #print(i)
                #print("vocabulary index")
                #print(j)
                #print(vocabulary[j])
                #print(wordsinsentence[k])
                trainingmatrix[i][j]+=1

# train NB

clf = MultinomialNB(0.01,True,None)
clf.class_log_prior_ = 10

clf.fit(trainingmatrix,results)


# open test file
with open('covid_test_public.tsv',encoding="utf8") as t:
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

# removing first result since its an index title
ids.remove(ids[0])
results.remove(results[0])

# change result list to binary , yes = 1 and no = 0
for i in range(len(results)):
    if results[i] == 'yes':
        results[i] = 1
    if results[i] == 'no':
        results[i] =0

# making all words in sentences lower case
for i in range(len(sentences)):
    sentences[i] = sentences[i].lower()

# removing the first sentence since its an index title
sentences.remove(sentences[0])
        
#create testing matrix
testingmatrix = [[0]*len(vocabulary)]*len(sentences)


for i in range(len(sentences)):
    wordsinsentence = sentences[i].split()
    testingmatrix[i] = [0]*len(vocabulary)
    for j in range(len(vocabulary)):
        for k in range(len(wordsinsentence)):
            if vocabulary[j] == wordsinsentence[k]:
                #print("Found match")
                #print("sentence index")
                #print(i)
                #print("vocabulary index")
                #print(j)
                #print(vocabulary[j])
                #print(wordsinsentence[k])
                testingmatrix[i][j]+=1



#Do all calculationshere before moving on to string transform



#Create yes and no arrays for both the prediction and the true
prediction = clf.predict(testingmatrix)

predno = []
trueno = []

#inverse arrays manually for every item in array that is == 1 then append a 0 and viceversa for both prediction and result
for i in range(len(prediction)):
    if prediction[i]==1:
        predno.append(0)
    else:
        predno.append(1)

for i in range(len(results)):
    if results[i] == 1:
        trueno.append(0)
    else:
        trueno.append(1)


# finding precision , recall and f1 for yes and no
yesprecision = precision_score(results,prediction)

noprecision = precision_score(trueno,predno)

yesrecall = recall_score(results,prediction)

norecall = recall_score(trueno,predno)

yesf1 = f1_score(results,prediction)

nof1 = f1_score(trueno,predno)

#creating string to contain both the yes and no in one line

stringprecision = str(yesprecision) + "  " + str(noprecision)

stringrecall = str(yesrecall) + "  " + str(norecall)

stringf1 = str(yesf1) + "  " + str(nof1)




preditprob = clf.predict_proba(testingmatrix)
score = clf.score(testingmatrix,results)
accuracy = accuracy_score(prediction,results)
print("--prediction_prob--")
print(preditprob)
print("--Score--")
print(score)
print("--Accuracy--")
print(accuracy)

rightorwrong = []


#creating string variables for Trace File

for i in range(len(prediction)):
    if prediction[i] == results[i]:
        rightorwrong.append('correct')
    else:
        rightorwrong.append('wrong')

length = len(prediction)
wordprediction = [[0]]*length
for i in range(len(prediction)):
    if prediction[i] == int(1):
        wordprediction[i] = 'yes'
    if prediction[i] == int(0):
        wordprediction[i] = 'no'

for i in range(len(results)):
    if results[i] == 1:
        results[i] = str('yes')
    if results[i] == 0:
        results[i] = str('no')
#Opening output file and printing Trace
f = open("trace_NB-BOW-0.txt", "a")
for i in range(len(prediction)):
    outputsentence = str(ids[i]) + "  "+ str(wordprediction[i]) + "  "+ str(preditprob[i][0]) +"  "+ str(results[i]) +"  "+ str(rightorwrong[i])+"\r"
    f.write(outputsentence)

f.close()

#Opening output file and printing eval
f = open("eval_NB-BOW-0.txt", "a")
f.write(str(accuracy)+"\r")
f.write(stringprecision+"\r")
f.write(stringrecall+"\r")
f.write(stringf1+"\r")
f.close()

