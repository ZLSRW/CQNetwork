import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd
import csv
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:          # 注意表头
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return

def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            # print(counter)
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, seq_size=41, seq_size1=64, interval=1):
        self.interval = interval

        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        self.seq_size=seq_size
        self.seq_size1=seq_size1
        # if normalize_method:
        #     self.data, _ = normalized(self.data, normalize_method, norm_statistic)
    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - 1
        train_data = self.data[lo: hi]
        # print('df.shape '+str(self.data.shape))

        #结合位点、时序训练数据
        train_seq_CGR,train_seq_Kmer,train_seq_Onehot,train_label=self.get_seq_label(train_data)
        # print(type(train_seq[0]))
        # print(type(train_label[0]))
        train_seq_CGR = torch.from_numpy(train_seq_CGR).type(torch.float)
        train_seq_Kmer = torch.from_numpy(train_seq_Kmer).type(torch.float)
        train_label = torch.from_numpy(train_label).type(torch.float)

        return train_seq_CGR,train_seq_Kmer,train_seq_Onehot,train_label  #train+target

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(1, self.df_length)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

    def get_seq_label(self,seq):
        seqs_CGR=[]
        seqs_Kmer=[]
        seqs_Onehot=[]
        labels=[]
        counter=0
        for ele in seq:
            i=0
            while i<len(ele):
                ele[i]=float(ele[i])
                i+=1
            # print('ele.length'+str(len(ele)))
            temp_seq_CGR=ele[:self.seq_size]
            # print(temp_seq_CGR)
            # print(len(temp_seq_CGR))

            temp_seq_Kmer=ele[self.seq_size:self.seq_size+self.seq_size1]

            temp_seq_Onehot=ele[(self.seq_size+self.seq_size1):-1]
            temp_label=ele[-1]
            seqs_CGR.append(temp_seq_CGR)
            seqs_Kmer.append(temp_seq_Kmer)
            seqs_Onehot.append(temp_seq_Onehot)
            labels.append(temp_label)
        return np.array(seqs_CGR,dtype='float64'),np.array(seqs_Kmer,dtype='float64'),np.array(temp_seq_Onehot,dtype='float64'),np.array(labels)

#数据读取测试
if __name__ == '__main__':
   print("done!")
