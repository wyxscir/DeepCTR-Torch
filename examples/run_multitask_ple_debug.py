# -*- coding: utf-8 -*-
# 试试deepctr-torch这个库
# 安装这个库的时候顺便安装了tf2.20 还有torch2.0

# 改ple试一试..
# debug看一看模型的结构是否对的上

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from ple import PLE

if __name__ == "__main__":
    # data description can be found in https://www.biendata.xyz/competition/icmechallenge2019/
    # 用200条数据做个例子 数据是字节短视频的一些特征
    data = pd.read_csv('/Users/wangyuxin102/Downloads/mycode/mmoe/examples/byterec_sample.txt', sep='\t',
                       names=["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like",
                              "music_id", "device", "time", "duration_time"])
    # data现在就是有表头的csv的表...

    sparse_features = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
    dense_features = ["duration_time"]  # 持续时间 上面的都叫稀疏特征 就这个持续时间叫密集特征

    target = ['finish', 'like']  # 目标应该是 是否完播 和 该用户是否喜欢本视频 数据可能不是很准啊就这个意思反正
    # 所以这俩就是所谓的多任务了

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # 对稀疏的特征进行标签编码，对密集的特征进行简单转换 
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])  # 编码都变成0-连续数字的形式..
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])  # 对这个不是编码的数值特征做一个类似归一化的处理，ok,看起来非常合理

    # 2.count #unique features for each sparse field,and record dense feature field name
    # 计算每个稀疏字段的唯一特征，并记录密集特征字段名称

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    split_boundary = int(data.shape[0] * 0.8)  # 划分训练集 测试集
    train, test = data[:split_boundary], data[split_boundary:]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    # 定义模型 里面两个多任务都是二分类问题 model.compile指定优化器等
    model = PLE(dnn_feature_columns, task_types=['binary', 'binary'],
                 l2_reg_embedding=1e-5, task_names=target, device=device)
    model.compile("adagrad", loss=["binary_crossentropy", "binary_crossentropy"],
                  metrics=['binary_crossentropy'], )
    # fit是训练器 200个样本很快就训练好了 送进来的数据还的具体看一下前面都做了什么处理
    history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2)
    pred_ans = model.predict(test_model_input, 256)  # 再在两个任务上进行测试
    print("")  # 他这个代码没有保存模型的实现
    for i, target_name in enumerate(target):
        print("%s test LogLoss" % target_name, round(log_loss(test[target[i]].values, pred_ans[:, i]), 4))
        print("%s test AUC" % target_name, round(roc_auc_score(test[target[i]].values, pred_ans[:, i]), 4))
