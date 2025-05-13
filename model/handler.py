import json
from datetime import datetime
import warnings

from data_loader.SiteBinding_dataloader1 import ForecastDataset
from .ConsensusNetwork import *
# from models.seq_graph import Model

import torch.utils.data as torch_data
import time
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
from .Utils import *

from .configure import *

warnings.filterwarnings("ignore")


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def save_model(model, model_dir, epoch, fold):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, str(fold) + '_' + epoch + '_PepBindA.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def save_model1(model, model_dir, epoch, fold):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, str(fold) + '_' + 'best' + '_IBGGN.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


# def load_model(model_dir):
#     if not model_dir:
#         return
#     file_name = os.path.join(model_dir, '1_best__PepBindA.pt')
#     print(file_name)
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     if not os.path.exists(file_name):
#         return
#     with open(file_name, 'rb') as f:
#         model = torch.load(f)
#     return model

def load_model(model_dir, fold):
    if not model_dir:
        return
    file_name = os.path.join(model_dir, str(fold) + '_best_IBGGN.pt')
    print(file_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def validate_inference_binding_site(model,dataloader):
    model.eval()
    with torch.no_grad():

        All_test_feature=[]

        All_confuse_matrix=[0,0,0,0]
        All_labels=[]
        All__labels_pred=[]

        All_result=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        cnt=0

        for i, (graph,Loop_graph, labels, onehots, onehot_feature,RegionIndex) in enumerate(dataloader):

            graph = graph.to('cuda:0')
            Loop_graph = Loop_graph.to('cuda:0')
            labels = labels.to('cuda:0')
            onehot_x = onehots.to('cuda:0')  # RNA_OP
            onehot_x = onehot_x.reshape(-1, 41, 15)
            onehot_feature = onehot_feature.to('cuda:0')



            final_representation, labels_pred, _,_ = model(graph,Loop_graph,onehot_x, onehot_feature,RegionIndex)

            labels_pred.squeeze()

            TN,FP,FN,TP,y_real,y_predict = Confuse(labels, labels_pred)
            All_confuse_matrix[0]+=TN
            All_confuse_matrix[1]+=FP
            All_confuse_matrix[2]+=FN
            All_confuse_matrix[3]+=TP

            All_labels.extend(y_real)
            All__labels_pred.extend(y_predict)


            cnt+=1

            labels_real = list(labels.contiguous().view(-1).cpu().detach().numpy())

            forecast_features = list(final_representation.cpu().detach().numpy())  # 全局特征
            xx = 0
            while xx < len(forecast_features):
                forecast_features[xx] = list(forecast_features[xx])
                forecast_features[xx].append(int(labels_real[xx]))
                xx += 1
            All_test_feature.extend(forecast_features)

        result,Real_Prediction, Real_Prediction_Prob=Confuse_Indicator(All_confuse_matrix,All_labels,All__labels_pred)
        # result=[x / cnt for x in All_result]

    return result, All_test_feature, Real_Prediction, Real_Prediction_Prob, onehot_feature


def train(train_data, valid_data, args, result_file, fold, species):

    ISGNN = ConsensusMotifNetwork(batch=args.batch_size, batch1=args.batch_size1, size=args.size,
                                           num=args.num)

    ISGNN.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=ISGNN.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=ISGNN.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(train_data)
    valid_set = ForecastDataset(valid_data)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = torch.nn.BCELoss(reduction='mean')

    total_params = 0
    for name, parameter in ISGNN.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_Acc = 0.0
    best_result = []
    best_Real_Predition = []
    best_Real_Predition_Prob = []

    best_train_feature = []
    best_x1_feature = []
    best_x2_feature = []

    best_validate_feature = []
    best_initial_feature = []
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        ISGNN.train()
        loss_total = 0
        cnt = 0
        auc_total = 0
        aupr_total = 0

        Temp_train_feature = []
        Temp_x1_feature = []
        Temp_x2_feature = []

        All_confuse_matrix = [0, 0, 0, 0]
        All_labels = []
        All__labels_pred = []

        for i, (graph, Loop_graph, labels, onehots, onehot_feature,RegionIndex) in enumerate(train_loader):
            graph = graph.to(args.device)
            Loop_graph = Loop_graph.to(args.device)
            labels = labels.to(args.device)
            onehot_x = onehots.to(args.device)  #
            onehot_x = onehot_x.reshape(-1, 41, 15)
            Elom4_feature = onehot_feature.to(args.device)
            RegionIndex = RegionIndex.to(args.device)


            final_representation, labels_pred, con_loss, distance_Loss = ISGNN(graph,Loop_graph,onehot_x, Elom4_feature, RegionIndex)

            labels_pred.squeeze()


            labels_real = list(labels.contiguous().view(-1).cpu().detach().numpy())
            forecast_feature = list(final_representation.cpu().detach().numpy())
            xx = 0
            while xx < len(forecast_feature):
                forecast_feature[xx] = list(forecast_feature[xx])
                forecast_feature[xx].append(int(labels_real[xx]))
                xx += 1
            Temp_train_feature.extend(forecast_feature)

            train_auc, _, _ = auroc(labels_pred.squeeze(), labels)
            train_aupr, _, _ = auprc(labels_pred.squeeze(), labels)

            binding_loss = criterion(labels_pred.squeeze(), labels.float())
            all_loss = binding_loss+0.1*con_loss+0.1*distance_Loss


            auc_total += train_auc
            aupr_total += train_aupr

            # 训练过程中
            """
            loss需要进行修改，不仅要考虑forecast和target，还要考虑预测结合位点和实际结合位点的关系（结合位点的损失不区分输入和目标，而是一起考虑）；
            """

            print(
                'epoch %d, all_loss %.4f, binding_loss %.4f, con_loss %.4f, distance_Loss %.4f,train_auc %.4f, train_aupr %.4f  '
                % (epoch + 1, all_loss, binding_loss, con_loss, distance_Loss, train_auc, train_aupr))
            cnt += 1

            # loss.backward()
            ISGNN.zero_grad()

            all_loss.backward()

            my_optim.step()

            loss_total += float(all_loss)

        # result, _, _ = Confuse_Indicator(All_confuse_matrix, All_labels,All__labels_pred)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | train_auc {:5.4f}| train_aupr {:5.4f}'.format(
                epoch + 1, (
                        time.time() - epoch_start_time), loss_total / cnt, auc_total / cnt, aupr_total / cnt))

        if 1 == 1:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')

            result, validate_features, Real_prediction, Real_prediction_prob, initial_feature = validate_inference_binding_site(ISGNN, valid_loader)

            MCC = result[0]
            auc = result[1]
            aupr = result[2]
            F1 = result[3]
            Acc = result[4]
            Sen = result[5]
            Spec = result[6]
            Prec = result[7]

            print('validate_MCC: ' + str(round(MCC, 4)) + ' ' + ' validate_auc: ' + str(
                round(auc, 4)) + ' validate_aupr: ' + str(round(aupr, 4)) + ' ' + ' validate_F1: ' + str(
                round(F1, 4)) + ' ' +
                  ' validate_Acc: ' + str(round(Acc, 4)) + ' ' + ' validate_Sen: ' + str(
                round(Sen, 4)) + ' ' + ' validate_Spec: ' + str(round(Spec, 4)) + ' '
                  + ' validate_Prec: ' + str(round(Prec, 4)))

            if Acc >= best_validate_Acc:
                best_validate_Acc = Acc
                best_result = result


                save_model1(ISGNN, result_file, epoch, fold)



    return forecast_feature, best_result


def inverse_validate_process(args, result_train_file, x1, x2, pre1, pre2):  # 五个参数分别为：模型、输出的两种表征、输入的两种表征
    model = load_model(result_train_file)
    x1 = np.array(x1, dtype='float64')
    x1 = torch.from_numpy(x1).type(torch.float).to(args.device)

    x2 = np.array(x2, dtype='float64')
    x2 = torch.from_numpy(x2).type(torch.float).to(args.device)

    pre1 = np.array(pre1, dtype='float64')
    pre1 = torch.from_numpy(pre1).type(torch.float).to(args.device)

    pre2 = np.array(pre2, dtype='float64')
    pre2 = torch.from_numpy(pre2).type(torch.float).to(args.device)

    model.eval()
    G1 = None
    G2 = None
    G1, G2 = model(G1, G2, x1, x2, pre1, pre2, True)
    return G1, G2


def test(test_data, args, result_train_file, save_dict, fold):  #

    print(result_train_file)

    model = load_model(result_train_file, fold)

    test_set = ForecastDataset(test_data)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)

    result, validate_features, Real_prediction, Real_prediction_prob, initial_feature = validate_inference_binding_site(model, test_loader)

    MCC = result[0]
    auc = result[1]
    aupr = result[2]
    F1 = result[3]
    Acc = result[4]
    Sen = result[5]
    Spec = result[6]
    Prec = result[7]

    print(
        'validate_MCC: ' + str(round(MCC, 4)) + ' ' + ' validate_auc: ' + str(round(auc, 4)) + ' validate_aupr: ' + str(
            round(aupr, 4)) + ' ' + ' validate_F1: ' + str(round(F1, 4)) + ' ' +
        ' validate_Acc: ' + str(round(Acc, 4)) + ' ' + ' validate_Sen: ' + str(
            round(Sen, 4)) + ' ' + ' validate_Spec: ' + str(round(Spec, 4)) + ' '
        + ' validate_Prec: ' + str(round(Prec, 4)))


    target_folder1 = './Case_CS_CT/' + str(save_dict) + '/Result'
    target_folder2 = './Case_CS_CT/' + str(save_dict) + '/Features'

    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(target_folder1):
        os.makedirs(target_folder1)

    if not os.path.exists(target_folder2):
        os.makedirs(target_folder2)




    return result
