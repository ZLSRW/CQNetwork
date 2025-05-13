import os
from datetime import datetime

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from models.main.handler import train, inverse_validate_process
from models.main.configure import *
import argparse
from data_loader.SiteBinding_dataloader1 import *
import numpy as np
# from .models.Utils import *
# from models.Utils import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--inverse', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='consensus')

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=8e-3)

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--validate_freq', type=int, default=5)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--batch_size1', type=int)  # 对最后一折的批次单独定义
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--optimizer', type=str, default='RMSProp')  # RMSProp
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=20)
parser.add_argument('--decay_rate', type=float, default=0.1)  # 0.5
parser.add_argument('--dropout_rate', type=float, default=0.2)  # 0.5
parser.add_argument('--leakyrelu_rate', type=int, default=0.5)  # 0.2

parser.add_argument('--size', type=int, default=41)
parser.add_argument('--num', type=int, default=1)

torch.cuda.set_device(0)

args = parser.parse_args()
print(f'Training configs: {args}')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 'Mouse_brain',
    # seq_types = ['Human_Brain', 'Human_Kidney', 'Human_Liver',  'Mouse_heart',
    #              'Mouse_kidney', 'Mouse_liver', 'Mouse_test', 'rat_brain', 'rat_kidney','rat_liver']
    # seq_types =['Mouse_heart','Mouse_kidney', 'Mouse_liver']
    # seq_types =['Mouse_heart','rat_kidney']
    seq_types = ['rat_liver']
    graph_types = ['consensus_motif_Graph','consensusGraph','motifGraph']
    if args.train:
        j = 0
        while j < len(seq_types):
            print(str(seq_types[j]) + '_train_validation beging!')

            result_train_file = os.path.join('output', args.dataset, seq_types[j])
            result_test_file = os.path.join('output', args.dataset, 'test')
            if not os.path.exists(result_train_file):
                os.makedirs(result_train_file)
            if not os.path.exists(result_test_file):
                os.makedirs(result_test_file)

            if args.train:  # 训练加验证
                try:
                    before_train = datetime.now().timestamp()
                    i = 1
                    all_result = []
                    while i < 5:
                        # if i!=1 and i!=3:
                        # if 1!=2 and i!=4:
                        if 1 == 1:
                            print('fold ' + str(i) + ' ')
                            print('-' * 99)
                            trainData = np.load(
                                './Pre-Encoding/data/' + str(seq_types[j]) + '/Train_Test/all/TrainData' + str(
                                    i) + '.npy', allow_pickle=True).tolist() 
                            testData = np.load(
                                './Pre-Encoding/data/' + str(seq_types[j]) + '/Train_Test/all/TestData' + str(
                                    i) + '.npy', allow_pickle=True).tolist()

                            # print(trainData[2][0])

                            # args.batch_size=len(testData[0])
                            temp1 = int(len(trainData[0]) / 4)
                            temp2 = len(testData[0])
                            if temp1 > temp2:
                                args.batch_size = temp1
                            else:
                                args.batch_size = temp2

                            args.batch_size1 = int(len(trainData[0]) - 3 * args.batch_size - 1)

                            print('Train begining!')
                            forecast_feature, result = train(trainData, testData, args, result_train_file, i,
                                                             seq_types[j])
                            all_result.append(result)
                            StorFile(all_result,
                                     './Pre-Encoding/data/' + str(seq_types[j]) + '/Result/result_supp' + str(
                                         i) + '.csv')
                            print(i)
                        i += 1
                    # StorFile(all_result, './Pre-Encoding/data/Human_Brain/Result/result.csv')
                    after_train = datetime.now().timestamp()
                    print(f'Training took {(after_train - before_train) / 60} minutes')

                except KeyboardInterrupt:
                    print('-' * 99)
                    print('Exiting from training early')
            # if args.evaluate:
            #     before_evaluation = datetime.now().timestamp()
            #     test(test_data, args, result_train_file, result_test_file)
            #     after_evaluation = datetime.now().timestamp()
            #     print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')

            print(str(seq_types[j]) + '_train_validation done!')
            j += 1


