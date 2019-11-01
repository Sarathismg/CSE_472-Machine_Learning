import csv
from collections import Counter
import math
import random

import numpy as np
import pandas
from sklearn import preprocessing

values_in_attributes = []
number_of_classes = 2
number_of_attributes = 0
rand1 = random.Random()
rand2 = random.Random()
rand2.seed(123)

class Tree:

    def __init__(self, attributeno, label):
        self.array = []
        self.attribute = attributeno
        self.label = label

    def push_node(self, t):
        self.array.append(t)


def LabelEncode(a,no):
    enc = preprocessing.LabelEncoder()
    enc.fit(a[:,no])
    b = enc.transform(a[:,no])
    a[:,no] = b

def cleanUnLabelledData(a):
    for k in range(0,len(a)):
        if a[k][-1] == " " or a[k][-1] == " ?":
            np.delete(a,k,0);
            k = k - 1;

def fillMissingData(a, ind, isDiscrete):
    #islabel true if its not some kind of attributes (where mean not possible)
    if isDiscrete == True:
        #z = np.ndarray.tolist(a[:,ind])
        #dat = max(z,key=z.count)
        f = Counter(a[:,ind])
        maxx = 0
        maxxind = 0
        for j in f:
            if f[j] > maxx:
                maxx = f[j]
                maxxind = j

        l = np.where(a[:,ind] == ' ')
        for k in range(0,len(l[0])):
            a[l[0][k]][ind] = maxxind

    else:
        tmp = a[:,ind]
        tmp1 = np.delete(tmp,np.where(tmp == ' ')).astype(float)
        mean = np.mean(tmp1)
        l = np.where(a[:, ind] == ' ')
        for k in range(0, len(l[0])):
            a[l[0][k]][ind] = mean

        a[:,ind] = a[:,ind].astype(np.float)
#dataset = []

def preprocessed_ds1():
    dataframe = pandas.read_csv("Churn.csv")
    dataset = dataframe.values
    dataset = np.array(dataset)

    discrete = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]
    continuous = [5, 18, 19]

    #handling missing values
    cleanUnLabelledData(dataset)
    for i in discrete:
        fillMissingData(dataset, i, True)

    for i in continuous:
        fillMissingData(dataset, i, False)

    #encoding and binarization
    for i in discrete:
        LabelEncode(dataset, i)

    for i in continuous:
        binarize(dataset,i)


    return dataset

def preprocessed_ds2():
    dataframe1 = pandas.read_csv("adult_data.txt")
    dataset1 = dataframe1.values
    dataset1 = np.array(dataset1)

    dataframe2 = pandas.read_csv("adult_test.txt")
    dataset2 = dataframe2.values
    dataset2 = np.array(dataset2)

    dataset = np.concatenate((dataset1,dataset2))


    discrete = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    continuous = [0, 2, 4, 10, 11, 12]

    # handling missing values
    cleanUnLabelledData(dataset)
    print("Cleaning done")
    for i in discrete:
        fillMissingData(dataset, i, True)

    for i in continuous:
        fillMissingData(dataset, i, False)

    # encoding and binarization
    for i in discrete:
        LabelEncode(dataset, i)

    for i in continuous:
        binarize(dataset, i)

    return dataset

def preprocessed_ds3():
    dataframe = pandas.read_csv("creditcard.csv")
    dataset = dataframe.values
    dataset = np.array(dataset)

    continuous = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,22,23,24,25,26,27,28,29]
    discrete = [30]

    # handling missing values
    cleanUnLabelledData(dataset)
    for i in discrete:
        fillMissingData(dataset, i, True)

    for i in continuous:
        fillMissingData(dataset, i, False)

    # encoding and binarization
    for i in discrete:
        LabelEncode(dataset, i)

    for i in continuous:
        binarize(dataset, i)

    return dataset

'''
def preprocessed_ds4():
    dataframe = pandas.read_csv("online1_data.csv")
    dataset = dataframe.values
    dataset = np.array(dataset)

    continuous = []
    discrete = [0,1,2,3,4,5,6,7,8]

    # handling missing values
    cleanUnLabelledData(dataset)
    for i in discrete:
        fillMissingData(dataset, i, True)

    for i in continuous:
        fillMissingData(dataset, i, False)

    # encoding and binarization
    for i in discrete:
        LabelEncode(dataset, i)

    for i in continuous:
        binarize(dataset, i)

    return dataset
    '''

def PLURALITY_VAL(examples):
    '''u, indices = np.unique(examples[:, -1], return_inverse=True)
    return u[np.argmax(np.bincount(indices))]'''
    no_of_ones = np.sum(examples[:, -1])
    no_of_zeros = len(examples) - no_of_ones
    if no_of_ones > no_of_zeros:
        return 1
    else:
        return 0


def get_subex(examples, attribute_no, attribute_val):
    '''dat = []
    for k in range(0, len(examples)):
        if examples[k][attribute_no] == attribute_val:
            dat.append(examples[k])
    dat = np.array(dat)
    return dat'''

    dat = examples[np.where(examples[:,attribute_no] == attribute_val)]
    if len(dat) == 0:
        return []
    return dat


def all_same_class(examples):

    sub = examples[:, -1]
    total_sum = np.sum(sub)
    if total_sum == 0:
        return 0
    elif total_sum == len(examples):
        return 1
    else:
        return -1



def get_IG(attributes, examples):
    min_ind = -1
    min_info_total = 1;

    for i in range(0, len(attributes)):
        info_total = 0
        newarr = examples[:, attributes[i]]
        cc = Counter(newarr)

        for value in range(0, values_in_attributes[attributes[i]]):
            mul = cc[value] / len(newarr)
            temp_ds = get_subex(examples, attributes[i], value)
            if len(temp_ds) == 0:
                continue
            #print(temp_ds)
            ccc = Counter(temp_ds[:, -1])
            sum = 0

            for class_no in range(0, number_of_classes):
                # handle cases
                if (ccc[class_no] == 0):
                    continue
                sum += -(ccc[class_no] / len(temp_ds)) * math.log2((ccc[class_no] / len(temp_ds)))

            mul *= sum
            info_total += mul

        if info_total < min_info_total:
            min_info_total = info_total
            min_ind = attributes[i]

    return min_ind

def parse_tree(t):
    a = []
    print(str(t.attribute)+"  "+str(t.label) )
    a.append(t)
    while(len(a) != 0):
        s = a.pop(0)
        for i in range(0,len(s.array)):
            a.append(s.array[i])
            print(str(s.array[i].attribute)+" "+str(s.array[i].label),end=" ||")
        print("")

def get_binarized_boundary(a, ind):
    #s = a[np.argsort(a,ind)]
    s = a[a[:,ind].argsort()]
    s1 = s[:, ind]
    s2 = s[:, -1]
    gini_arr = []
    cc = Counter(s2)
    lessno = 0
    lessyes = 0
    greaterno = cc[0]
    greateryes = cc[1]


    total = greaterno + greateryes

    for i in range(0, len(s)):
        greater = greateryes + greaterno
        less = lessyes + lessno
        gini = 0

        if greater == 0:
            gini = 1 - (lessyes/less)**2 - (lessno/less)**2

        elif less == 0:
            gini = 1 - (greateryes / greater) ** 2 - (greaterno/greater)**2

        else:
            gini = (1 - (lessyes/less)**2 - (lessno/less)**2) * (less / total) + (1 - (greateryes / greater) ** 2 - (greaterno/greater)**2)* (greater/total)

        gini_arr.append(gini)


        #update

        if s2[i] == 0:
            lessno+=1
            greaterno-=1

        else:
            lessyes+=1
            greateryes-=1


    gini_arr.append(1 - (lessyes/less)**2 - (lessno/less)**2)
    lowest_gini = 1
    lowest_gini_bound = -1
    for k in range(0,len(s1)):
        if k==len(s1)-1 or s1[k]!=s1[k+1]:
            if gini_arr[k] < lowest_gini:
                lowest_gini = gini_arr[k]
                lowest_gini_bound = s1[k]

    return lowest_gini_bound

def binarize(a, ind):
    lim = get_binarized_boundary(a, ind)
    for m in range(0, len(a)):
        if a[m][ind] < lim:
            a[m][ind] = 0

        else:
            a[m][ind] = 1


def DTL(examples, attributes, parent_exmaples, depth):
    print("Depth left: "+str(depth))
    if len(examples) == 0:
        return Tree(-1, PLURALITY_VAL(parent_exmaples))

    elif all_same_class(examples) != -1:
        return Tree(-1, all_same_class(examples))

    elif len(attributes) < 1 or depth <= 0:
        return Tree(-1, PLURALITY_VAL(examples))

    else:
        A = get_IG(attributes, examples)
        t = Tree(A, -1)
        for k in range(0, values_in_attributes[A]):
            sub_ex = get_subex(examples, A, k)
            subtree = DTL(sub_ex, np.delete(attributes, np.where(attributes == A)), examples, depth - 1)
            t.push_node(subtree)

        return t

def Adaboost(examples, K):
    global number_of_attributes, values_in_attributes

    w = []
    h = []
    Z = []

    for i in range(0,len(examples)):
        w.append(1/len(examples))

    w = np.array(w)

    #assert sum(w) == 1, str(sum(w))

    for i in range(0, K):
        data = examples[np.random.choice(examples.shape[0], len(examples),replace=True, p=w), :]
        dataset = np.array(data)

        number_of_classes = len(set(dataset[:, -1]))
        print("no of classes: " + str(number_of_classes))

        number_of_attributes = len(dataset[0]) - 1
        print("no of attr: " + str(number_of_attributes))

        left_attribute = []

        for kk in range(0, number_of_attributes):
            #values_in_attributes.append(len(set(dataset[:, kk])))
            left_attribute.append(kk)

        left_attribute = np.array(left_attribute)
        #values_in_attributes = np.array(values_in_attributes)

        m = DTL(data, left_attribute, dataset, 1)

        error = 0

        for j in range(0, len(examples)):
            if predict(m, examples[j]) != examples[j][-1]:
                error = error + w[j]
        if error > 0.5:
            continue

        h.append(m)

        for j in range(0, len(examples)):
            if predict(m, examples[j]) == examples[j][-1]:
                w[j] = w[j] * (error / (1 - error))

        ssum = np.sum(w)
        for j in range(0, len(w)):
            w[j] /= ssum

        Z.append(math.log2((1 - error) / error))

    return h, Z


def predict(T, val):
    while(T.label == -1):
        s = T.attribute
        #print(val[s])
        T = T.array[int(val[s])]

    return T.label

def predict_Adaboost(h, Z, val):
    sum = 0
    for i in range(0, len(h)):
        a = predict(h[i], val)
        if( a == 0):
            sum -= Z[i]
        else:
            sum +=Z[i]

    if sum > 0:
        return 1;
    else:
        return 0;

def dataset1():
    global number_of_attributes, values_in_attributes
    dataset = preprocessed_ds1()
    dataset = dataset[:, 1:]
    train = []
    test = []
    random.seed(8354)
    # print(ds[4])
    for m in range(0, len(dataset)):
        ran = random.uniform(0, 1)
        if ran < 0.8:
            train.append(dataset[m])
        else:
            test.append(dataset[m])

    # placing some global values
    dataset = np.array(dataset)

    number_of_classes = len(set(dataset[:, -1]))
    print("no of classes: " + str(number_of_classes))

    number_of_attributes = len(dataset[0]) - 1
    print("no of attr: " + str(number_of_attributes))

    left_attribute = []

    for i in range(0, number_of_attributes):
        values_in_attributes.append(len(set(dataset[:, i])))
        left_attribute.append(i)

    left_attribute = np.array(left_attribute)
    values_in_attributes = np.array(values_in_attributes)
    print("Completed Preprocessing....Trainig>>>>")
    train = np.array(train)
    test = np.array(test)
    # call train
    tree = DTL(train, left_attribute, dataset, 20)
    print("Completed Training")

    # evaluate
    tot = len(test)
    corr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for k in range(0, len(test)):
        b = predict(tree, test[k])
        print(str(b) + "  ->  " + str(test[k][-1]))
        if b == test[k][-1]:
            corr += 1
        if b == 0:
            if test[k][-1] == 0:
                tn += 1
            elif test[k][-1] == 1:
                fn += 1
        elif b == 1:
            if test[k][-1] == 1:
                tp += 1
            elif test[k][-1] == 0:
                fp += 1

    print("Accuracy: " + str(corr / tot))
    print("Recall(TPR): " + str(tp / (tp + fn)))
    print("Specificity(TNR): " + str(tn / (tn + fp)))
    print("Precision(PPV): " + str(tp / (tp + fp)))
    print("False Discovery Rate(FDR): " + str(fp / (fp + tp)))
    print("F1 Score: " + str((2 * tp) / (2 * tp + fp + fn)))

def dataset2():
    global number_of_attributes, values_in_attributes
    dataset = preprocessed_ds2()

    train = dataset[0:32561, :]
    test = dataset[32561: ,:]
    dataset = np.array(dataset)

    with open("new_file.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(dataset)

    number_of_classes = len(set(dataset[:, -1]))
    print("no of classes: " + str(number_of_classes))

    number_of_attributes = len(dataset[0]) - 1
    print("no of attr: " + str(number_of_attributes))

    left_attribute = []

    for i in range(0, number_of_attributes):
        values_in_attributes.append(len(set(dataset[:, i])))
        left_attribute.append(i)

    left_attribute = np.array(left_attribute)
    values_in_attributes = np.array(values_in_attributes)
    print("Completed Preprocessing....Trainig>>>>")
    train = np.array(train)
    test = np.array(test)
    # call train
    tree = DTL(train, left_attribute, dataset, 20)
    print("Completed Training")

    tot = len(test)
    corr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for k in range(0, len(test)):
        b = predict(tree, test[k])
        print(str(b) + "  ->  " + str(test[k][-1]))
        if b == test[k][-1]:
            corr += 1
        if b == 0:
            if test[k][-1] == 0:
                tn += 1
            elif test[k][-1] == 1:
                fn += 1
        elif b == 1:
            if test[k][-1] == 1:
                tp += 1
            elif test[k][-1] == 0:
                fp += 1
    print("tp : "+str(tp)+" fn "+str(fn))
    print("Accuracy: " + str(corr / tot))
    print("Recall(TPR): " + str(tp / (tp + fn)))
    print("Specificity(TNR): " + str(tn / (tn + fp)))
    print("Precision(PPV): " + str(tp / (tp + fp)))
    print("False Discovery Rate(FDR): " + str(fp / (fp + tp)))
    print("F1 Score: " + str((2 * tp) / (2 * tp + fp + fn)))


def dataset3():
    global number_of_attributes, values_in_attributes
    dataset = preprocessed_ds3()

    with open("new_file.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(dataset)

    train = []
    test = []
    random.seed(16)
    # print(ds[4])
    for m in range(0, len(dataset)):
        ran = random.uniform(0, 1)
        ran2 = rand2.uniform(0, 1)
        if ran2 > 0.90 or dataset[m][-1] == 1:
            if ran > 0.2:
                train.append(dataset[m])
            else:
                test.append(dataset[m])

    # placing some global values
    dataset = np.array(dataset).astype(np.int)

    number_of_classes = len(set(dataset[:, -1]))
    print("no of classes: " + str(number_of_classes))

    number_of_attributes = len(dataset[0]) - 1
    print("no of attr: " + str(number_of_attributes))

    left_attribute = []

    for i in range(0, number_of_attributes):
        values_in_attributes.append(len(set(dataset[:, i])))
        left_attribute.append(i)

    left_attribute = np.array(left_attribute)
    values_in_attributes = np.array(values_in_attributes)
    print("Completed Preprocessing....Trainig>>>>")
    train = np.array(train)
    test = np.array(test)
    print("Train size: "+str(len(train)))
    print("Test size: "+str(len(test)))
    # call train
    tree = DTL(train, left_attribute, train, 30)
    print("Completed Training")

    # evaluate
    tot = len(test)
    corr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for k in range(0, len(test)):
        b = predict(tree, test[k])
        print(str(b) + "  ->  " + str(test[k][-1]))
        if b == test[k][-1]:
            corr += 1
        if b == 0:
            if test[k][-1] == 0:
                tn += 1
            elif test[k][-1] == 1:
                fn += 1
        elif b == 1:
            if test[k][-1] == 1:
                tp += 1
            elif test[k][-1] == 0:
                fp += 1
    print("tp : "+str(tp)+" fn "+str(fn))
    print("Accuracy: " + str(corr / tot))
    print("Recall(TPR): " + str(tp / (tp + fn)))
    print("Specificity(TNR): " + str(tn / (tn + fp)))
    print("Precision(PPV): " + str(tp / (tp + fp)))
    print("False Discovery Rate(FDR): " + str(fp / (fp + tp)))
    print("F1 Score: " + str((2 * tp) / (2 * tp + fp + fn)))

'''
def dataset4(tN, tP, fN, fP):
    global number_of_attributes, values_in_attributes
    dataset = preprocessed_ds4()

    with open("new_file.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(dataset)

    train = []
    test = []
    random.seed(16)
    # print(ds[4])
    for m in range(0, len(dataset)):
        ran = random.uniform(0, 1)
        ran2 = rand2.uniform(0, 1)
        if ran > 0.2:
            train.append(dataset[m])
        else:
            test.append(dataset[m])

    # placing some global values
    dataset = np.array(dataset).astype(np.int)

    number_of_classes = len(set(dataset[:, -1]))
    print("no of classes: " + str(number_of_classes))

    number_of_attributes = len(dataset[0]) - 1
    print("no of attr: " + str(number_of_attributes))

    left_attribute = []

    for i in range(0, number_of_attributes):
        values_in_attributes.append(len(set(dataset[:, i])))
        left_attribute.append(i)

    left_attribute = np.array(left_attribute)
    values_in_attributes = np.array(values_in_attributes)
    print("Completed Preprocessing....Trainig>>>>")
    train = np.array(train)
    test = np.array(test)
    print("Train size: " + str(len(train)))
    print("Test size: " + str(len(test)))
    # call train
    tree = DTL(train, left_attribute, train, 30)
    print("Completed Training")

    # evaluate
    tot = len(test)
    corr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for k in range(0, len(test)):
        b = predict(tree, test[k])
        print(str(b) + "  ->  " + str(test[k][-1]))
        if b == test[k][-1]:
            corr += 1
        if b == 0:
            if test[k][-1] == 0:
                tn += 1
            elif test[k][-1] == 1:
                fn += 1
        elif b == 1:
            if test[k][-1] == 1:
                tp += 1
            elif test[k][-1] == 0:
                fp += 1

    tn = tn * tN
    tp = tp * tP
    fn = fn * fN
    fp = fp * fP
    print("tp : " + str(tp) + " fn " + str(fn))
    print("Accuracy: " + str(corr / tot))
    print("Recall(TPR): " + str(tp / (tp + fn)))
    print("Specificity(TNR): " + str(tn / (tn + fp)))
    print("Precision(PPV): " + str(tp / (tp + fp)))
    print("False Discovery Rate(FDR): " + str(fp / (fp + tp)))
    print("F1 Score: " + str((2 * tp) / (2 * tp + fp + fn)))
    '''





def dataset1_adaboost():
        global number_of_attributes, values_in_attributes
        dataset = preprocessed_ds1()
        dataset = dataset[:, 1:]
        train = []
        test = []
        random.seed(8354)
        # print(ds[4])
        for m in range(0, len(dataset)):
            ran = random.uniform(0, 1)
            if ran < 0.8:
                train.append(dataset[m])
            else:
                test.append(dataset[m])

        number_of_classes = len(set(dataset[:, -1]))
        print("no of classes: " + str(number_of_classes))

        number_of_attributes = len(dataset[0]) - 1
        print("no of attr: " + str(number_of_attributes))

        for i in range(0, number_of_attributes):
            values_in_attributes.append(len(set(dataset[:, i])))

        values_in_attributes = np.array(values_in_attributes)

        # placing some global values

        print("Completed Preprocessing....Trainig>>>>")
        train = np.array(train)
        test = np.array(test)
        # call train
        trees, weights = Adaboost(train, 20)
        print("Completed Training")

        # evaluate
        tot = len(test)
        corr = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for k in range(0, len(test)):
            b = predict_Adaboost(trees, weights, test[k])
            print(str(b) + "  ->  " + str(test[k][-1]))
            if b == test[k][-1]:
                corr += 1
            if b == 0:
                if test[k][-1] == 0:
                    tn += 1
                elif test[k][-1] == 1:
                    fn += 1
            elif b == 1:
                if test[k][-1] == 1:
                    tp += 1
                elif test[k][-1] == 0:
                    fp += 1

        print("Accuracy: " + str(corr / tot))
        '''print("Recall(TPR): " + str(tp / (tp + fn)))
        print("Specificity(TNR): " + str(tn / (tn + fp)))
        print("Precision(PPV): " + str(tp / (tp + fp)))
        print("False Discovery Rate(FDR): " + str(fp / (fp + tp)))
        print("F1 Score: " + str((2 * tp) / (2 * tp + fp + fn)))'''

def dataset2_adaboost():
    global number_of_attributes, values_in_attributes
    dataset = preprocessed_ds2()

    train = dataset[0:32561, :]
    test = dataset[32561:, :]
    dataset = np.array(dataset)

    number_of_classes = len(set(dataset[:, -1]))
    print("no of classes: " + str(number_of_classes))

    number_of_attributes = len(dataset[0]) - 1
    print("no of attr: " + str(number_of_attributes))

    for i in range(0, number_of_attributes):
        values_in_attributes.append(len(set(dataset[:, i])))

    values_in_attributes = np.array(values_in_attributes)

    print("Completed Preprocessing....Trainig>>>>")
    train = np.array(train)
    test = np.array(test)
    # call train
    trees, weights = Adaboost(train, 15)
    print("Completed Training")

    '''test performance'''
    tot = len(test)
    corr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for k in range(0, len(test)):
        b = predict_Adaboost(trees, weights, test[k])
        print(str(b) + "  ->  " + str(test[k][-1]))
        if b == test[k][-1]:
            corr += 1
        if b == 0:
            if test[k][-1] == 0:
                tn += 1
            elif test[k][-1] == 1:
                fn += 1
        elif b == 1:
            if test[k][-1] == 1:
                tp += 1
            elif test[k][-1] == 0:
                fp += 1

    print("Accuracy: " + str(corr / tot))
    '''print("Recall(TPR): "+str(tp/(tp + fn)))
    print("Specificity(TNR): "+str(tn/(tn + fp)))
    print("Precision(PPV): "+str(tp / (tp + fp)))
    print("False Discovery Rate(FDR): "+str(fp / (fp + tp)))
    print("F1 Score: "+ str((2 * tp) / (2*tp + fp + fn)))'''

def dataset3_adaboost():
    global number_of_attributes, values_in_attributes
    dataset = preprocessed_ds3()

    with open("new_file.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(dataset)

    train = []
    test = []
    #random.seed(84)
    # print(ds[4])
    for m in range(0, len(dataset)):
        ran = random.uniform(0, 1)
        ran2 = rand2.uniform(0, 1)
        if ran2 < 0.10 or dataset[m][-1] == 1:
            if ran < 0.8:
                train.append(dataset[m])
            else:
                test.append(dataset[m])

    number_of_classes = len(set(dataset[:, -1]))
    print("no of classes: " + str(number_of_classes))

    number_of_attributes = len(dataset[0]) - 1
    print("no of attr: " + str(number_of_attributes))

    for i in range(0, number_of_attributes):
        values_in_attributes.append(len(set(dataset[:, i])))

    values_in_attributes = np.array(values_in_attributes)

    # placing some global values
    dataset = np.array(dataset).astype(np.int)

    print("Completed Preprocessing....Trainig>>>>")
    train = np.array(train)
    test = np.array(test)
    # call train
    trees, weights = Adaboost(train, 20)
    print("Completed Training")

    tot = len(test)
    corr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for k in range(0, len(test)):
        b = predict_Adaboost(trees, weights, test[k])
        print(str(b) + "  ->  " + str(test[k][-1]))
        if b == test[k][-1]:
            corr += 1
        if b == 0:
            if test[k][-1] == 0:
                tn += 1
            elif test[k][-1] == 1:
                fn += 1
        elif b == 1:
            if test[k][-1] == 1:
                tp += 1
            elif test[k][-1] == 0:
                fp += 1

    print("Accuracy: " + str(corr / tot))
    #print("Recall(TPR): "+str(tp/(tp + fn)))
    #print("Specificity(TNR): "+str(tn/(tn + fp)))
    #print("Precision(PPV): "+str(tp / (tp + fp)))
    #print("False Discovery Rate(FDR): "+str(fp / (fp + tp)))
    #print("F1 Score: "+ str((2 * tp) / (2*tp + fp + fn)))


'''
def dataset4_adaboost(tN, tP, fN, fP):
    global number_of_attributes, values_in_attributes
    dataset = preprocessed_ds4()

    with open("new_file.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(dataset)

    train = []
    test = []
    #random.seed(84)
    # print(ds[4])
    for m in range(0, len(dataset)):
        ran = random.uniform(0, 1)
        ran2 = rand2.uniform(0, 1)
        if ran < 0.8:
            train.append(dataset[m])
        else:
            test.append(dataset[m])

    number_of_classes = len(set(dataset[:, -1]))
    print("no of classes: " + str(number_of_classes))

    number_of_attributes = len(dataset[0]) - 1
    print("no of attr: " + str(number_of_attributes))

    for i in range(0, number_of_attributes):
        values_in_attributes.append(len(set(dataset[:, i])))

    values_in_attributes = np.array(values_in_attributes)

    # placing some global values
    dataset = np.array(dataset).astype(np.int)

    print("Completed Preprocessing....Trainig>>>>")
    train = np.array(train)
    test = np.array(test)
    # call train
    trees, weights = Adaboost(train, 20)
    print("Completed Training")

    tot = len(test)
    corr = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for k in range(0, len(test)):
        b = predict_Adaboost(trees, weights, test[k])
        print(str(b) + "  ->  " + str(test[k][-1]))
        if b == test[k][-1]:
            corr += 1
        if b == 0:
            if test[k][-1] == 0:
                tn += 1
            elif test[k][-1] == 1:
                fn += 1
        elif b == 1:
            if test[k][-1] == 1:
                tp += 1
            elif test[k][-1] == 0:
                fp += 1

    tn = tn * tN
    tp = tp * tP
    fn = fn * fN
    fp = fp * fP

    print("Accuracy: " + str(corr / tot))
    print("Recall(TPR): "+str(tp/(tp + fn)))
    print("Specificity(TNR): "+str(tn/(tn + fp)))
    print("Precision(PPV): "+str(tp / (tp + fp)))
    print("False Discovery Rate(FDR): "+str(fp / (fp + tp)))
    print("F1 Score: "+ str((2 * tp) / (2*tp + fp + fn)))
    '''


#dataset4_adaboost(1, 1, 1, 1)
dataset2_adaboost()
