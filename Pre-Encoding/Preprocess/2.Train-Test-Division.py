import math
import random
import csv

import numpy as np
fix_seed = 2023
random.seed(fix_seed)
np.random.seed(fix_seed)

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])  # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# 读取一个txt文件，并将其转为fasta文件
def txt_to_fasta(input_file):
    sequences = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sequences.append(line)

    output_file = input_file.split('txt')[0] + 'fasta'
    with open(output_file, 'w') as f:
        for i, sequence in enumerate(sequences):
            f.write(f'>Sequence{i + 1}\n')
            f.write(sequence + '\n')


def partition(ls, size):
    return [ls[i:i + size] for i in range(0, len(ls), size)]

# seq_types = ['Human_Brain', 'Human_Kidney', 'Human_Liver', 'Mouse_brain', 'Mouse_heart',
#              'Mouse_kidney', 'Mouse_liver', 'Mouse_test', 'rat_brain', 'rat_kidney','rat_liver']

seq_types = ['rat_liver']

P_file_name1 = '../Dataset/Tissue Specific/'
P_file_tail = '/PositiveOnehot.npy'
P_file_tail1 = '/PositiveGraphs.npy'
P_file_tail1Loop = '/PositiveLoopGraphs.npy'
P_file_tail2 = '/PositiveFeature.npy'

# P_file_name1 = '../Dataset/Tissue Specific/'
# P_file_tail = '/PositiveOnehot_PrimaryStructure.npy'
# P_file_tail1 = '/PositiveGraphs_PrimaryStructure.npy'
# P_file_tail1Loop = '/PositiveLoopGraphs_PrimaryStructure.npy'
# P_file_tail2 = '/PositiveFeature.npy'
P_file_tail3 = '/PositiveRegionIndex.npy'

N_file_name1 = '../Dataset/Tissue Specific/'
N_file_tail = '/NegativeOnehot.npy'
N_file_tail1 = '/NegativeGraphs.npy'
N_file_tail1Loop = '/NegativeLoopGraphs.npy'
N_file_tail2 = '/NegativeFeature.npy'
N_file_tail3 = '/NegativeRegionIndex.npy'

i = 0
while i < len(seq_types):
    print('i='+str(i))
    Positive_sample = []
    Negative_sample = []

    input_P_file_name = P_file_name1 + seq_types[i] + P_file_tail  # 正onehot
    input_P_file_name1 = P_file_name1 + seq_types[i] + P_file_tail1  # 正图
    input_P_Loop_file_name1 = P_file_name1 + seq_types[i] + P_file_tail1Loop  # 正图Loop
    input_P_file_name2 = P_file_name1 + seq_types[i] + P_file_tail2  # 正特征
    input_P_file_name3 = P_file_name1 + seq_types[i] + P_file_tail3  # 正区域


    input_N_file_name = N_file_name1 + seq_types[i] + N_file_tail  # 负onehot
    input_N_file_name1 = N_file_name1 + seq_types[i] + N_file_tail1  # 负图
    input_N_Loop_file_name1 = N_file_name1 + seq_types[i] + N_file_tail1Loop  # 负图Loop
    input_N_file_name2 = N_file_name1 + seq_types[i] + N_file_tail2  # 负特征
    input_N_file_name3 = N_file_name1 + seq_types[i] + N_file_tail3  # 负区域

    P_onehot = np.load(input_P_file_name).tolist()
    P_graph = np.load(input_P_file_name1).tolist()
    P_Loop_graph = np.load(input_P_Loop_file_name1).tolist()
    P_feature = np.load(input_P_file_name2).tolist()
    P_region = np.load(input_P_file_name3).tolist()

    N_onehot = np.load(input_N_file_name).tolist()
    N_graph = np.load(input_N_file_name1).tolist()
    N_Loop_graph = np.load(input_N_Loop_file_name1).tolist()
    N_feature = np.load(input_N_file_name2).tolist()
    N_region = np.load(input_N_file_name3).tolist()

    print('len(P_onehot)'+str(seq_types[i])+' '+str(len(P_onehot)))
    # print(P_graph[0])
    # print(P_onehot[0])

    # 由AllEdge产生RandomList
    # RandomList = random.sample(range(0, len(P_onehot)), len(P_onehot))  # 打乱顺序，返回打乱顺序后的列表（LM） 长度为行数
    # # 从0到len(NewAllEdgeNum)这个数列中随机取len(NewAllEdgeNum)个数
    # print('len(RandomList)', len(RandomList))
    # print(RandomList)
    #
    # NewRandomList = partition(RandomList, math.ceil(len(RandomList) / 5))  # 列表之列表
    # print('len(NewRandomList[0])', len(NewRandomList[0]))
    # StorFile(NewRandomList, '../data/' + seq_types[i] + '/NewRandomList.csv')  # 将数据分为5行，每行数据为int类型，表示要取哪条数据

    # StorFile(NewRandomList, 'data_kmer/'+seq_types[xx]+'/NewRandomList.csv') # 将数据分为5行，每行数据为int类型，表示要取哪条数据
    # StorFile(NewRandomList, 'data_oneHot/'+seq_types[xx]+'/NewRandomList.csv') # 将数据分为5行，每行数据为int类型，表示要取哪条数据
    NewRandomList=[]
    ReadMyCsv2(NewRandomList,'../data/' + seq_types[i] + '/NewRandomList.csv')

    counter = 0
    while counter < len(NewRandomList):  # 5
        print(counter)
        Num = 0
        TestListPair = []
        TrainListPair = []

        TestListLoopPair = []
        TrainListLoopPair = []

        TestListOneHot = []
        TrainListOneHot = []

        TestListFeature = []
        TrainListFeature = []

        TestRegion = []
        TrainRegion = []

        counter2 = 0
        while counter2 < len(NewRandomList):
            if counter2 == counter:  # 四行训练，另外一行测试
                TestListPair.extend(P_graph[Num:Num + len(NewRandomList[counter2])])
                TestListLoopPair.extend(P_Loop_graph[Num:Num + len(NewRandomList[counter2])])
                TestListOneHot.extend(P_onehot[Num:Num + len(NewRandomList[counter2])])
                TestListFeature.extend(P_feature[Num:Num + len(NewRandomList[counter2])])
                TestRegion.extend(P_region[Num:Num + len(NewRandomList[counter2])])

            if counter2 != counter:
                TrainListPair.extend(P_graph[Num:Num + len(NewRandomList[counter2])])
                TrainListLoopPair.extend(P_Loop_graph[Num:Num + len(NewRandomList[counter2])])
                TrainListOneHot.extend(P_onehot[Num:Num + len(NewRandomList[counter2])])
                TrainListFeature.extend(P_feature[Num:Num + len(NewRandomList[counter2])])
                TrainRegion.extend(P_region[Num:Num + len(NewRandomList[counter2])])
            Num = Num + len(NewRandomList[counter2])
            counter2 = counter2 + 1

        Num = 0
        counter2 = 0
        while counter2 < len(NewRandomList):
            if counter2 == counter:  # 四行训练，另外一行测试
                TestListPair.extend(N_graph[Num:Num + len(NewRandomList[counter2])])
                TestListLoopPair.extend(N_Loop_graph[Num:Num + len(NewRandomList[counter2])])
                TestListOneHot.extend(N_onehot[Num:Num + len(NewRandomList[counter2])])
                TestListFeature.extend(N_feature[Num:Num + len(NewRandomList[counter2])])
                TestRegion.extend(N_region[Num:Num + len(NewRandomList[counter2])])
            if counter2 != counter:
                TrainListPair.extend(N_graph[Num:Num + len(NewRandomList[counter2])])
                TrainListLoopPair.extend(N_Loop_graph[Num:Num + len(NewRandomList[counter2])])
                TrainListOneHot.extend(N_onehot[Num:Num + len(NewRandomList[counter2])])
                TrainListFeature.extend(N_feature[Num:Num + len(NewRandomList[counter2])])
                TrainRegion.extend(N_region[Num:Num + len(NewRandomList[counter2])])

            Num = Num + len(NewRandomList[counter2])
            counter2 = counter2 + 1

        # 再增加一组随机下标，对每一折进行调整
        RandomList_Train = random.sample(range(0, len(TrainListFeature)), len(TrainListFeature))
        # print(RandomList_Train)
        RandomList_Test = random.sample(range(0, len(TestListFeature)), len(TestListFeature))
        print(RandomList_Test)

        TrainListPairNew=[] # 训练的二级结构图
        TestListPairNew=[] # 测试的二级结构图

        TrainListLoopPairNew=[] # 训练的二级结构图
        TestListLoopPairNew=[] # 测试的二级结构图

        TrainListOneHotNew=[] # 训练的标签图
        TestListOneHotNew=[] # 测试的标签图

        TrainListFeatureNew=[] #训练的节点属性
        TestListFeatureNew=[] # 测试的节点属性

        TrainListRegionNew = []  # 训练的节点属性
        TestListRegionNew = []  # 测试的节点属性

        k=0
        while k<len(RandomList_Train):
            TrainListPairNew.append(TrainListPair[RandomList_Train[k]])
            TrainListLoopPairNew.append(TrainListLoopPair[RandomList_Train[k]])
            TrainListOneHotNew.append(TrainListOneHot[RandomList_Train[k]])
            TrainListFeatureNew.append(TrainListFeature[RandomList_Train[k]])
            TrainListRegionNew.append(TrainRegion[RandomList_Train[k]])
            k+=1
        k=0
        while k<len(RandomList_Test):
            TestListPairNew.append(TestListPair[RandomList_Test[k]])
            TestListLoopPairNew.append(TestListLoopPair[RandomList_Test[k]])
            TestListOneHotNew.append(TestListOneHot[RandomList_Test[k]])
            TestListFeatureNew.append(TestListFeature[RandomList_Test[k]])
            TestListRegionNew.append(TestRegion[RandomList_Test[k]])
            k+=1

        # 保存平衡样本的随机列表
        RandomList_TrainSave=[[a] for a in RandomList_Train]
        RandomList_TestSave=[[a] for a in RandomList_Test]
        StorFile(RandomList_TrainSave, '../data/' + seq_types[i] + '/Train_Test/Random/NewRandomList_Train' + str(counter) + '.csv')
        StorFile(RandomList_TestSave, '../data/' + seq_types[i] + '/Train_Test/Random/NewRandomList_Test' + str(counter) + '.csv')

        TestNameGraph = '../data/' + seq_types[i] + '/Train_Test/Test_graph' + str(counter) + '.npy'
        np.save(TestNameGraph, np.array(TestListPairNew))

        TrainNameGraph = '../data/' + seq_types[i] + '/Train_Test/Train_graph' + str(counter) + '.npy'
        np.save(TrainNameGraph, np.array(TrainListPairNew))

        TestNameLoopGraph = '../data/' + seq_types[i] + '/Train_Test/Test_Loop_graph' + str(counter) + '.npy'
        np.save(TestNameLoopGraph, np.array(TestListLoopPairNew))

        TrainNameLoopGraph = '../data/' + seq_types[i] + '/Train_Test/Train_Loop_graph' + str(counter) + '.npy'
        np.save(TrainNameLoopGraph, np.array(TrainListLoopPairNew))

        TestNameOnehot = '../data/' + seq_types[i] + '/Train_Test/Test_OnehotLabel' + str(counter) + '.npy'
        # StorFile(TestListOneHot,'TestListOneHot'+str(counter)+'.csv')
        np.save(TestNameOnehot, np.array(TestListOneHotNew))

        TrainNameOnehot = '../data/' + seq_types[i] + '/Train_Test/Train_OnehotLabel' + str(counter) + '.npy'
        np.save(TrainNameOnehot, np.array(TrainListOneHotNew))

        TrainNameFeature = '../data/' + seq_types[i] + '/Train_Test/Train_Feature' + str(counter) + '.npy'
        np.save(TrainNameFeature, np.array(TrainListFeatureNew))

        TestNameFeature = '../data/' + seq_types[i] + '/Train_Test/Test_Feature' + str(counter) + '.npy'
        np.save(TestNameFeature, np.array(TestListFeatureNew))

        TrainNameRegion = '../data/' + seq_types[i] + '/Train_Test/Train_Region' + str(counter) + '.npy'
        np.save(TrainNameRegion, np.array(TrainListRegionNew))

        TestNameRegion = '../data/' + seq_types[i] + '/Train_Test/Test_Region' + str(counter) + '.npy'
        np.save(TestNameRegion, np.array(TestListRegionNew))

        counter = counter + 1
    print(str(seq_types[i])+' 执行完毕')
    i += 1
