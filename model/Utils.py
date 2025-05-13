import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import random
#
# fix_seed = 2023
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

# 元素级矩阵除法
def safe_elementwise_divide(mat1, mat2):
    # 将除数中元素值为0的位置置为1，以避免除0错误
    mat2_nonzero = mat2.clone()
    mat2_nonzero[mat2_nonzero == 0] = 1.0

    # 执行元素级除法
    result = mat1 / mat2_nonzero

    mat2=mat2.repeat(mat1.size(0),1,1)
    # 将除数元素值为0的位置的结果置为0
    result[mat2 == 0] = 0.0

    return result

# 矩阵求解
def solve_for_A(B, x):
    # 假设有 batch_size 个批次
    batch_size = B.size(0)

    # 创建一个空的张量来存储结果
    A = torch.empty(batch_size, x.size(-2), x.size(-2))

    for i in range(batch_size):
        A[i] = torch.abs(torch.matmul(B[i],torch.pinverse(x[i])))

    return A


def remove_last_element(input_list):
    # 创建一个新的列表，用于存储处理后的结果
    new_list = []

    # 遍历输入的三维列表
    for i in range(len(input_list)):
        new_2d_list = []
        for j in range(len(input_list[i])):
            new_2d_list.append(input_list[i][j][:-1])
        new_list.append(new_2d_list)

    return new_list

#钩子mask
def get_zero_grad_hook(mask, device="cuda:0"):
    """zero out gradients"""
    def hook(grad):
        return grad * mask.to(device)
    return hook

#将预测概率转换为预测标签。
def Prediction_label(pred):
    a,b,c=pred.size()
    pred_arry=pred.detach().numpy()
    # print(str(a)+' '+str(b)+' '+str(c))
    predlabels=np.zeros((a,b,c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                if pred_arry[i][j][k]>0.5:
                    predlabels[i][j][k]=1.0
                else:
                    predlabels[i][j][k]=0.0
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(predlabels)

#attention mask
def Zero_One_Mask(pred):
    a,b,c=pred.size()
    pred_arry=pred.detach().numpy()
    # print(str(a)+' '+str(b)+' '+str(c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                if pred_arry[i][j][k]>0.5:
                    pred_arry[i][j][k]=1.0
                else:
                    pred_arry[i][j][k]=0.0
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(pred_arry)

#index mask
def Index_Mask(pred,inputs_site,scale=0.5):
    a, b, c = pred.size()
    # print(type(scale))
    d=math.ceil(c*scale)

    a1,b1,c1 = inputs_site.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    # 获得mask后的训练概率和训练标签
    pred_mask = np.zeros((a, b, d))
    pred_mask_index = np.zeros((a, b, d))
    inputs_site_mask = np.zeros((a1, b1, d))

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0
            counter=0
            index=0
            while k < c:
                #根据scale对预测序列和标签序列进行裁切
                if inputs_site[i][j][k] == 1.0 and counter<=3:
                    counter+=1
                    pred_mask_index[i][j][index] = k #记录正样本位置
                    pred_mask[i][j][index]=pred_arry[i][j][k]
                    inputs_site_mask[i][j][index]=inputs_site[i][j][k]
                    index+=1
                k += 1
            # print(str(inputs_site[i][j])+' '+str(counter))
            while index < d:
                n=0
                while n < len(inputs_site[i][j]):
                    if inputs_site[i][j][n]==0.0:
                        pred_mask_index[i][j][index] = n  # 记录正样本位置
                        pred_mask[i][j][index] = pred_arry[i][j][n]
                        inputs_site_mask[i][j][index] = inputs_site[i][j][n]
                        index += 1
                        break
                    n+=1



            j += 1
        i += 1

    return torch.from_numpy(pred_mask_index).float(),torch.from_numpy(pred_mask).float(),torch.from_numpy(inputs_site_mask) .float()#下标，概率，位点与否

def Sequence_reduction(pred,inputs_site,ID_len): #将窗口中的转态全部转换为二维列表，左边为状态，右边为标签
    a, b, c = pred.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    ID_len_array=ID_len.detach().numpy()
    # 获得mask后的训练概率和训练标签

    #用于指标计算
    prediction=[]
    label=[]
    #用于序列还原
    ID_prediction=[]
    ID_label=[]

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0

            while k < int(ID_len_array[i][j][1]):
                temp_id_pre = []
                temp_id_lab = []
                #根据scale对预测序列和标签序列进行裁切
                prediction.append(float(pred_arry[i][j][k]))
                label.append(float(inputs_site[i][j][k]))

                temp_id_pre.append(ID_len_array[i][j][0])
                temp_id_pre.append(float(pred_arry[i][j][k]))

                temp_id_lab.append(ID_len_array[i][j][0])
                temp_id_lab.append(float(inputs_site[i][j][k]))

                ID_prediction.append(temp_id_pre)
                ID_label.append(temp_id_lab)

                k += 1
            j += 1
        i += 1

    return prediction,label,ID_prediction,ID_label

def Sequence_reduction1(pred,inputs_site,ID_len): #将窗口中的转态全部转换为二维列表，左边为状态，右边为标签
    a, b, c = pred.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    ID_len_array=ID_len.detach().numpy()
    # 获得mask后的训练概率和训练标签

    #用于指标计算
    prediction=[]
    label=[]
    #用于序列还原
    ID_prediction=[]
    ID_label=[]

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0
            while k < c:
                temp_id_pre = []
                temp_id_lab = []
                #根据scale对预测序列和标签序列进行裁切
                prediction.append(float(pred_arry[i][j][k]))
                label.append(float(inputs_site[i][j][k]))

                temp_id_pre.append(ID_len_array[i][j][0])
                temp_id_pre.append(float(pred_arry[i][j][k]))

                temp_id_lab.append(ID_len_array[i][j][0])
                temp_id_lab.append(float(inputs_site[i][j][k]))

                ID_prediction.append(temp_id_pre)
                ID_label.append(temp_id_lab)

                k += 1
            j += 1
        i += 1

    return prediction,label,ID_prediction,ID_label


# def Indicator(y_real,y_predict):
#     from sklearn.metrics import confusion_matrix
#     x=[]
#     y=[]
#     for ele in y_predict:
#         ele=float(ele)
#         if ele>0.5:
#             x.append(1)
#         else:
#             x.append(0)
#     for ele in y_real:
#         ele=int(ele)
#         y.append(ele)
#
#     np.array(x)
#
#     CM = confusion_matrix(x, y)
#     print(CM)
#     CM = CM.tolist()
#     TN = CM[0][0]
#     FP = CM[0][1]
#     FN = CM[1][0]
#     TP = CM[1][1]
#     print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
#     Acc = (TN + TP) / (TN + TP + FN + FP)
#     Sen = TP / (TP + FN)
#     Spec = TN / (TN + FP)
#     Prec = TP / (TP + FP)
#     MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
#     F1 = f1_score(x, y)
#     auc=AUC(torch.tensor(x),torch.tensor(y))
#
#     # 分母可能出现0，需要讨论待续
#     print('MCC:', round(MCC, 4))
#     print('AUC:', round(AUC, 4))
#     print('F1:', round(F1, 4))
#     print('Acc:', round(Acc, 4))
#     print('Sen:', round(Sen, 4))
#     print('Spec:', round(Spec, 4))
#     print('Prec:', round(Prec, 4))
#
#     Result = []
#     Result.append(round(MCC, 4))
#     Result.append(round(AUC, 4))
#     Result.append(round(F1, 4))
#     Result.append(round(Acc, 4))
#     Result.append(round(Sen, 4))
#     Result.append(round(Spec, 4))
#     Result.append(round(Prec, 4))
#     return Result


# #多维张量指定维度对角化 2D->3D
# def tensor_diag(input):
#     device = torch.device("cuda:0")
#     a,b=input.size()
#     input=input.cpu()
#     input_arry=input.detach().numpy()
#     input_diag_array=np.zeros((a,b,b))
#     # print(str(a)+' '+str(b)+' '+str(c))
#     i=0
#     while i<a:
#         j=0
#         while j<b:
#             input_diag_array[i][j][j]=input_arry[i][j]
#             j+=1
#         i+=1
#     return torch.from_numpy(input_diag_array).float().to(device)

def tensor_diag(A):
    # print(A.shape)
    device = torch.device("cuda:0")
    # 获取A的形状
    rows, cols = A.size()
    # 创建一个全零的三维张量B，维度为(rows, cols, cols)
    B = torch.zeros((rows, cols, cols))

    # 使用for循环填充张量B
    for i in range(rows):
        # 将A的每一行作为对角线元素，并赋值给B的相应位置
        B[i] = torch.diag(A[i])

    return B.to(device)


#最大最小标准化，mask最小值且进行归一化
def normalize_and_symmetrize_tensor(input_tensor, threshold=0.5):
    # Calculate row-wise minimum and maximum
    min_values, _ = input_tensor.min(dim=-1, keepdim=True)
    max_values, _ = input_tensor.max(dim=-1, keepdim=True)

    # Perform row-wise min-max normalization
    normalized_tensor = (input_tensor - min_values) / (max_values - min_values)

    # Set values less than the threshold to 0 暂时去掉阈值筛选
    # normalized_tensor[normalized_tensor < threshold] = 0

    # Calculate row-wise sums
    row_sums = normalized_tensor.sum(dim=-1, keepdim=True)

    # Add a small epsilon to avoid division by zero for rows with all zeros
    epsilon = 1e-8
    row_sums = row_sums + epsilon

    # Perform row-wise normalization
    final_normalized_tensor = normalized_tensor / row_sums

    # Symmetrize the normalized tensor
    final_normalized_tensor = (final_normalized_tensor + final_normalized_tensor.transpose(-1, -2)) / 2

    return final_normalized_tensor

#最大最小标准化，mask最小值且进行归一化
def max_min(input):
    a, b, c = input.size()
    threahold=0.6
    # input_array = input.detach().numpy()
    input_array = input.cpu().detach().numpy()
    i = 0
    while i < a:
        j = 0
        while j < b:
            k=0
            max_x=max(input_array[i][j])
            min_x=min(input_array[i][j])
            # print('最大值 '+str(max_x)+' 最小值 '+str(min_x))
            diff=max_x-min_x
            while k<c:
                input_array[i][j][k]=((input_array[i][j][k])-min_x)/diff
                # if input_array[i][j][k]<threahold:
                #     input_array[i][j][k]=0.0
                # else:
                #     input_array[i][j][k]=1.0

                k+=1
            # sumx = sum(input_array[i][j])
            # print(sumx)
            j += 1

        i += 1
    # print(input_array)
    return torch.from_numpy(input_array).float()

#最大最小归一化以及硬标签转换(非张量)
def max_min_2D(input):
    a=len(input)
    b = len(input[0])
    threahold=0.5
    input_array = input
    i = 0
    while i < a:
        j = 0
        max_x=max(input_array[i])
        min_x=min(input_array[i])
        # print('最大值 '+str(max_x)+' 最小值 '+str(min_x))
        diff=max_x-min_x
        while j<b:
            input_array[i][j]=((input_array[i][j])-min_x)/diff
            if input_array[i][j]<threahold:
                input_array[i][j]=0.0
            # else:
                #     input_array[i][j][k]=1.0
            j += 1
        i += 1
    # print(input_array)
    return input_array

#构建图上正样本的mask
def pos_neg_mask(input,pos_mask,neg_mask):
    # print(input)
    a, b = input.size()
    input_array = input.detach().numpy()
    pos_mask = pos_mask.detach().numpy()
    neg_mask = neg_mask.detach().numpy()
    i = 0
    while i < a:
        j = 0
        while j < b:
            if input_array[i][j]<0:
                neg_mask[i][j]=0 #负值表示存在关联，即使得负样本mask在该位置为0
            else:
                pos_mask[i][j]=0
            j += 1
        i += 1
    return torch.from_numpy(pos_mask).float(),torch.from_numpy(neg_mask).float()

#找出列表中元素值相同的元素下标
def find_duplicates(lst):
    indices = {}
    for i, x in enumerate(lst):
        if x not in indices:
            indices[x] = [i]
        else:
            indices[x].append(i)
    return [v for v in indices.values() if len(v) > 1]

def normalize_motif(ids, tensor):
    """
    对于给定的三维张量，按行对指定维度下标的切片求平均值并标准化。

    Args:
        ids (list[list[int]]): 一个二维列表，每行记录了具有相同性质的维度索引。
        tensor (torch.Tensor): 要修改的三维张量，形状为 (height, width, depth)。

    Returns:
        torch.Tensor: 修改后的张量，形状与输入张量相同。
    """
    # 按行迭代，对每行中的维度下标所对应的切片求平均值并标准化
    for row in ids:
        # 当前行对应的所有下标
        idxs = row
        # 对应下标的切片
        slices = [tensor[:, idx, :] for idx in idxs]
        # 计算所有切片的平均值
        mean_ = torch.stack(slices, axis=0).mean()
        # 对所有切片中的所有值除以平均值
        for slice_ in slices:
            if mean_ != 0:
                slice_ /= mean_
            else:
                slice_.fill_(0)  # 避免对 0 进行除法运算

    return tensor


#计算位点评价指标计算，包括AUC，AUPR，F1
# 1.AUC,fpr,tpr
def auroc(prob, label):
    y_true = label.data.cpu().numpy().flatten()
    y_scores = prob.data.cpu().numpy().flatten()
    y_scores=np.nan_to_num(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc_score = auc(fpr, tpr)
    return auroc_score, fpr, tpr


def AUC(prob, label):
    # y_true = label.data.numpy().flatten()
    # y_scores = prob.data.numpy().flatten()
    fpr, tpr, thresholds = roc_curve(label, prob)
    auroc_score = auc(fpr, tpr)
    return auroc_score

# def AUPR(prob,label):
#     precision,recall,thresholds=precision_recall_curve(prob,label)
#     auprc_score=auc(recall,precision)
#     return auprc_score

# 2.auprc_score,precision,recall
def auprc(prob,label):
    y_true=label.data.cpu().numpy().flatten()
    y_scores=prob.data.cpu().numpy().flatten()
    y_scores = np.nan_to_num(y_scores)
    precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
    auprc_score=auc(recall,precision)
    return auprc_score,precision,recall

def AUPR(prob,label):
    precision,recall,thresholds=precision_recall_curve(prob,label)
    auprc_score=auc(recall,precision)
    return auprc_score

def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction
def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb

def Confuse_Indicator(CM,y_real,y_predict):
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    RealAndPrediction = MyRealAndPrediction(x, y)
    RealAndPredictionProb = MyRealAndPredictionProb(x, z)

    TN = CM[0]
    FP = CM[1]
    FN = CM[2]
    TP = CM[3]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    if (TP+FN)!=0:
        Sen = TP / (TP + FN) #这个也是recall
    else:
        Sen=np.inf
    if (TN + FP)!=0:
        Spec = TN / (TN + FP)
    else:
        Spec=np.inf
    if (TP + FP)!=0:
        Prec = TP / (TP + FP)
    else:
        Prec=np.inf
    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))!=0:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC=np.inf
    if (Prec+Sen)==0:
        f1=0
    else:
        f1 = (2*Prec*Sen)/(Prec+Sen)
    Auc,_,_= auroc(torch.tensor(y_predict),torch.tensor(x))
    Aupr,_,_=auprc(torch.tensor(y_predict),torch.tensor(x))

    Result = []
    Result.append(round(MCC, 4))
    Result.append(round(Auc, 4))
    Result.append(round(Aupr, 4))
    Result.append(round(f1, 4))
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))

    print('MCC:%.4f, Acc:%.4f, Sen:%.4f, Sp:%.4f' % (MCC, Acc, Sen, Spec))

    return Result, RealAndPrediction,RealAndPredictionProb

def Confuse(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    y_real=list(y_real.data.cpu().numpy().flatten())
    y_predict=list(y_predict.data.cpu().numpy().flatten())
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    CM = confusion_matrix(x, y)

    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    return TN,FP,FN,TP,y_real,y_predict

def Indicator(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    y_real=y_real.data.cpu().numpy().flatten()
    y_predict=y_predict.data.cpu().numpy().flatten()
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    RealAndPrediction = MyRealAndPrediction(x, y)
    RealAndPredictionProb = MyRealAndPredictionProb(x, z)

    CM = confusion_matrix(x, y)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print(CM)
    Acc = (TN + TP) / (TN + TP + FN + FP)
    if (TP+FN)!=0:
        Sen = TP / (TP + FN)
    else:
        Sen=np.inf
    if (TN + FP)!=0:
        Spec = TN / (TN + FP)
    else:
        Spec=np.inf
    if (TP + FP)!=0:
        Prec = TP / (TP + FP)
    else:
        Prec=np.inf
    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))!=0:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC=np.inf
    f1 = f1_score(y, x)
    Auc,_,_= auroc(torch.tensor(y_predict),torch.tensor(x))
    Aupr,_,_=auprc(torch.tensor(y_predict),torch.tensor(x))

    Result = []
    Result.append(round(MCC, 4))
    Result.append(round(Auc, 4))
    Result.append(round(Aupr, 4))
    Result.append(round(f1, 4))
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))

    print('MCC:%.4f, Acc:%.4f, Sen:%.4f, Sp:%.4f' % (MCC, Acc, Sen, Spec))

    return Result,RealAndPrediction,RealAndPredictionProb

#标准化
def normalized_input(input):
    a,b,c=input.size()
    input=input.detach().numpy()
    # print(str(a)+' '+str(b)+' '+str(c))
    new_input=np.zeros((a,b,c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                new_input[i][j][k]=(input[i][j][k]-min(input[i][j]))/(max(input[i][j])-min(input[i][j]))
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(new_input).float()


#马氏距离计算

def mahalanobis_distance(X, regularization_param=1e-5):
    # 计算均值向量
    mean_vector = torch.mean(X, dim=1, keepdim=True)

    # 计算去中心化矩阵
    X_c = X - mean_vector

    # 计算协方差矩阵
    cov_matrix = torch.matmul(X_c.transpose(1, 2), X_c) / (X.size(1) - 1)

    # 添加正则化项
    cov_matrix += regularization_param * torch.eye(cov_matrix.size(-1), device=cov_matrix.device)

    # 计算协方差矩阵的逆矩阵
    inv_cov_matrix = torch.inverse(cov_matrix)
    #
    # print(X_c.shape)
    # print(inv_cov_matrix.shape)
    # print(X_c.transpose(1, 2).shape)

    # 计算马氏距离
    mahalanobis_dist = torch.bmm(torch.bmm(X_c, inv_cov_matrix), X_c.transpose(1, 2))

    return mahalanobis_dist.squeeze()

def gaussian_kernel_similarity(mahalanobis_dist, sigma=1.0):
    # 计算高斯核函数
    similarity = torch.exp(-0.5 * (mahalanobis_dist / sigma)**2)

    return similarity


# 阈值筛选
import torch

# 对一级结构构图进行按行的归一化
import torch


def row_normalize_softmax(input_tensor):
    # 获取张量的形状
    b, n, _ = input_tensor.shape

    # 对每个批次的每个矩阵的每行进行 softmax 归一化，不改变形状
    for i in range(b):
        for j in range(n):
            row_values = input_tensor[i, j, :]
            input_tensor[i, j, :] = torch.nn.functional.softmax(row_values, dim=-1)

    return input_tensor


# 示例
batch_size = 2
num_nodes = 3

# 针对一级结构图，只保留主对角线两侧的值作为效应蛋白的结合区域
def threshold_filter_motif(input_tensor, k):
    # 获取张量的形状
    b, n, _ = input_tensor.shape

    # 对每一行，只保留对角线上元素和其前后 k 个元素的值，其余元素置 0
    for i in range(b):
        for j in range(n):
            for l in range(n):
                input_tensor[i, j, j] = 1.0
                if abs(j - l) > k:
                    input_tensor[i, j, l] = 0.0
                elif input_tensor[i, j, l] < 0.5:
                    input_tensor[i, j, :] = 0.0

    return input_tensor

# 针对二级结构图，保留茎环结构特征和碱基配对特征。
def threshold_filter(adj_tensor):
    # 计算每个矩阵的行平均值
    row_means = adj_tensor.mean(dim=2, keepdim=True)

    # 将小于行平均值的元素置零
    adj_tensor = torch.where(torch.lt(adj_tensor, row_means), torch.tensor(0.0).to('cuda:0'), adj_tensor)

    return adj_tensor

def keep_top_k_values(adj_tensor, k):
    # 获取每行最大的K个值和对应的索引
    top_k_values, top_k_indices = adj_tensor.topk(k, dim=2)

    # 创建一个与原始张量相同形状的零张量
    mask = torch.zeros_like(adj_tensor)

    # 在每行的最大K个值的索引处设置为1
    mask.scatter_(2, top_k_indices, 1)

    # 通过乘法将不是最大K个值的元素置零
    adj_tensor = adj_tensor * mask

    return adj_tensor


from sklearn.preprocessing import normalize
import numpy as np
from scipy.special import iv
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.optimize import minimize
from scipy.integrate import quad
import sys
import math
import time

def threshold_to_zero(mx, threshold):
    """Set value in a sparse matrix lower than
     threshold to zero.

    Return the 'coo' format sparse matrix.

    Parameters
    ----------
    mx : array_like
        Sparse matrix.
    threshold : float
        Threshold parameter.
    """
    high_values_indexes = set(zip(*((np.abs(mx) >= threshold).nonzero())))
    nonzero_indexes = zip(*(mx.nonzero()))

    if not sp.isspmatrix_lil(mx):
        mx = mx.tolil()

    for s in nonzero_indexes:
        if s not in high_values_indexes:
            mx[s] = 0.0
    mx = mx.tocoo()
    mx.eliminate_zeros()
    return mx


