#name : muhammad Arham
#roll : 231453689
#lab 7 AI

import pandas as pd

def main():
    ####################
    testingLabels = []
    p_of_0 = 0
    p_of_1 = 0
    df = pd.read_csv('trainNaiveLabels.csv')
    testingLabels = df['label1'].tolist()
    number1 = 0
    number0 = 0
    for i,n in enumerate(testingLabels):
        if n == 0:
            number0 += 1
        else:
            number1 += 1
    p_of_0 = number0 / len(testingLabels)
    p_of_1 = number1 / len(testingLabels)
    #######################
    labels = []
    #df = pd.read_csv('trainNaiveLabels.csv')
    labels = df['label1'].tolist()
    indexes_0 = []
    indexes_1 = []
    for i,n in enumerate(labels):
        if n == 0:
            indexes_0 += [i]
        else:
            indexes_1 += [i]
    #print(indexes_0)
    data = pd.read_csv('trainNaive.csv')
    data_list = data.values.tolist()
    #print(data_list)
    data_list_0 = []
    data_list_1 = []
    for i in indexes_0:
        data_list_0.append(data_list[i])
    for i in indexes_1:
        data_list_1.append(data_list[i])
    #print(data_list_0[0])
    #number of 0s
    number_0_in_0 = 0
    number_1_in_0 = 0
    for i in data_list_0: #counting number of 0s and 1s in 0 catagory
        for j in i:
            if j == 0:
                number_0_in_0 += 1
            elif j == 1:
                number_1_in_0 += 1
    #saving probabilties for 0 case
    p_case0_0 = []# p saved from 0 to 9 cases from feat 1 to feat 10
    for i in range(1,11):
        n = 0
        for j in data_list_0:
            if j[i] == 0:
                n += 1
        #print(n)
        p_case0_0.append(n/number_0_in_0)
    #print(p_case0_0,number_0_in_0)
    p_case0_1 = []
    for i in range(1,11):
        n = 0
        for j in data_list_0:
            if j[i] == 1:
                n += 1
        #print(n)
        p_case0_1.append(n/number_1_in_0)
    #print(p_case0_1)
    #saving probabilites for 1 case
    #0 p for case 1 
    number_0_in_1 = 0
    number_1_in_1 = 0
    for i in data_list_1: #counting number of 0s and 1s in 1 catagory
        for j in i:
            if j == 0:
                number_0_in_1 += 1
            elif j == 1:
                number_1_in_1 += 1
#saving probabilties for 0 case
    p_case1_0 = []# p saved from 0 to 9 cases from feat 1 to feat 10
    for i in range(1,11):
        n = 0
        for j in data_list_1:
            if j[i] == 0:
                n += 1
        #print(n)
        p_case1_0.append(n/number_0_in_1)
    #print(p_case0_0,number_0_in_0)
    p_case1_1 = []
    for i in range(1,11):
        n = 0
        for j in data_list_1:
            if j[i] == 1:
                n += 1
        #print(n)
        p_case1_1.append(n/number_1_in_1)
    #print(p_case0_0)
    print(len(p_case0_1))
    #print(p_case1_0)
    #print(p_case1_1)
    ans = Naive_Bayes_classifier(p_case0_0,p_case0_1,p_case1_0,p_case1_1,p_of_0,p_of_1,data_list[3])
    #print(ans)  
    #################################
    #       TESTING
    #################################
    ans = pd.read_csv('trainNaiveLabels.csv')
    answers = ans['label1'].tolist()
    #print(testingLabels)
    testingData = pd.read_csv('trainNaive.csv')
    testingValues = testingData.values.tolist()
    ##print(testingValues)
    #weeding out some data to get around 300 values
    #print(len(testingValues))
    newFoundLabels = []
    for i in range(0,len(testingValues),5):
        n = [0,0]
        n[0] = labels[i]
        n[1] = Naive_Bayes_classifier(p_case0_0,p_case0_1,p_case1_0,p_case1_1,p_of_0,p_of_1,testingValues[i])
        newFoundLabels.append(n)
    #print(newFoundLabels)
    n = 0
    for i in newFoundLabels:
        if i[0] == i[1]:
            n += 1
    accuracy = n/len(newFoundLabels)
    print("accuray of the classifier : "+str((accuracy*100)))




def Naive_Bayes_classifier(p_case0_0,p_case0_1,p_case1_0,p_case1_1,p0,p1,testingRow):
    ans_0 = p0
    ans_1 = p1
    for i,n in enumerate(testingRow):
        if i != 0:
            if n == 0:
                ans_0 *= p_case0_0[i-1]
                ans_1 *= p_case1_0[i-1]
            elif n == 1:
                ans_0 *= p_case0_1[i-1]
                ans_1 *= p_case1_1[i-1]
    #print(ans_0,ans_1)
    if ans_0 < ans_1:
        return 1
    else:
        return 0


    
main()
