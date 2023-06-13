# -*- coding:utf-8 -*-
"""
Author:
    zanshuxun, zanshuxun@aliyun.com

Reference:
    [1] Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C] (https://dl.acm.org/doi/10.1145/3219819.3220007)
"""
import torch
import torch.nn as nn

from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import combined_dnn_input
from deepctr_torch.layers import DNN, PredictionLayer


class MMOE(BaseModel):
    """Instantiates the Multi-gate Mixture-of-Experts architecture.

    torch的模型贼简单 直接init和forward定义一下就可以

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model. 模型需要的所有深层的特征
    :param num_experts: integer, number of experts.  专家网络的数量
    :param expert_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param gate_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param tower_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param l2_reg_linear: float, L2 regularizer strength applied to linear part.
    :param l2_reg_embedding: float, L2 regularizer strength applied to embedding vector.
    :param l2_reg_dnn: float, L2 regularizer strength applied to DNN.
    :param init_std: float, to use as the initialize std of embedding vector.
    :param seed: integer, to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN.
    :param dnn_use_bn: bool, Whether use BatchNormalization before activation or not in DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression'].
    :param task_names: list of str, indicating the predict target of each tasks.
    :param device: str, ``"cpu"`` or ``"cuda:0"``.
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.

    :return: A PyTorch model instance.
    """

    def __init__(self, dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128),
                 gate_dnn_hidden_units=(64,), tower_dnn_hidden_units=(64,), l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False,
                 task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr'), device='cpu', gpus=None):
        super(MMOE, self).__init__(linear_feature_columns=[], dnn_feature_columns=dnn_feature_columns,
                                   l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std,
                                   seed=seed, device=device, gpus=gpus)  # 基类应该也有不少东西
        self.num_tasks = len(task_names)
        if self.num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than 1")
        if len(dnn_feature_columns) == 0:
            raise ValueError("dnn_feature_columns is null!")
        if len(task_types) != self.num_tasks:
            raise ValueError("num_tasks must be equal to the length of task_types")

        for task_type in task_types:
            if task_type not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

        self.num_experts = num_experts
        self.task_names = task_names
        self.input_dim = self.compute_input_dim(dnn_feature_columns)  # 33 输入维度
        self.expert_dnn_hidden_units = expert_dnn_hidden_units  # (256,128)
        self.gate_dnn_hidden_units = gate_dnn_hidden_units  # (64, )
        self.tower_dnn_hidden_units = tower_dnn_hidden_units  # (64, )

        # expert dnn  预先定义的DNN层作为专家网络
        self.expert_dnn = nn.ModuleList([DNN(self.input_dim, expert_dnn_hidden_units, activation=dnn_activation,
                                             l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                             init_std=init_std, device=device) for _ in range(self.num_experts)])

        # gate dnn 门网络，最后输出的是标量形式的权重
        if len(gate_dnn_hidden_units) > 0:
            self.gate_dnn = nn.ModuleList([DNN(self.input_dim, gate_dnn_hidden_units, activation=dnn_activation,
                                               l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                                               init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.gate_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.gate_dnn_final_layer = nn.ModuleList(
            [nn.Linear(gate_dnn_hidden_units[-1] if len(gate_dnn_hidden_units) > 0 else self.input_dim,
                       self.num_experts, bias=False) for _ in range(self.num_tasks)])

        # tower dnn (task-specific)
        if len(tower_dnn_hidden_units) > 0:
            self.tower_dnn = nn.ModuleList(
                [DNN(expert_dnn_hidden_units[-1], tower_dnn_hidden_units, activation=dnn_activation,
                     l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                     init_std=init_std, device=device) for _ in range(self.num_tasks)])
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.tower_dnn.named_parameters()),
                l2=l2_reg_dnn)
        self.tower_dnn_final_layer = nn.ModuleList([nn.Linear(
            tower_dnn_hidden_units[-1] if len(tower_dnn_hidden_units) > 0 else expert_dnn_hidden_units[-1], 1,
            bias=False)
                                                    for _ in range(self.num_tasks)])

        self.out = nn.ModuleList([PredictionLayer(task) for task in task_types])

        regularization_modules = [self.expert_dnn, self.gate_dnn_final_layer, self.tower_dnn_final_layer]
        for module in regularization_modules:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], module.named_parameters()), l2=l2_reg_dnn)
        self.to(device)
    # 他这个模型封装的比较规范 forward就是根据输入X 前向传播一下 loss计算等部分在fit的另外部分来做 所以传进来的参数没有标签啥的
    def forward(self, X):  # 前向传播的时候所有需要的特征，标签 都在这个X里面了，直接对其操作即可 对于编号特征他还传进来的真是编号 这个接触的不多 很神奇
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,  # X：torch.Size([32, 9])
                                                                                  self.embedding_dict)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # torch.Size([32, 33]) 把初始的特征转换为输入的维度 32是batchsize

        # expert dnn
        expert_outs = []  # 三个，里面每个专家是  torch.Size([32, 128]) （经过专家DNN的学习） 直接送进去input就行
        for i in range(self.num_experts):
            expert_out = self.expert_dnn[i](dnn_input)
            expert_outs.append(expert_out)
        expert_outs = torch.stack(expert_outs, 1)  # (bs, num_experts, dim)  # torch.Size([32, 3, 128])

        # gate dnn
        mmoe_outs = []
        for i in range(self.num_tasks):  # 对于每一个门来说，前面的专家网络已经传播完了，在这里直接用专家
            if len(self.gate_dnn_hidden_units) > 0:
                gate_dnn_out = self.gate_dnn[i](dnn_input)  # 门DNN这里同样也是 input直接进去
                gate_dnn_out = self.gate_dnn_final_layer[i](gate_dnn_out)  # torch.Size([32, 3])
            else:
                gate_dnn_out = self.gate_dnn_final_layer[i](dnn_input)
            gate_mul_expert = torch.matmul(gate_dnn_out.softmax(1).unsqueeze(1), expert_outs)  # (bs, 1, dim)  # torch.Size([32, 1, 128])
            mmoe_outs.append(gate_mul_expert.squeeze())  # 最后得到两个门的结果

        # tower dnn (task-specific)
        task_outs = []
        for i in range(self.num_tasks):
            if len(self.tower_dnn_hidden_units) > 0:
                tower_dnn_out = self.tower_dnn[i](mmoe_outs[i])  # 一个门的结果 对应一个塔的结果
                tower_dnn_logit = self.tower_dnn_final_layer[i](tower_dnn_out)
            else:
                tower_dnn_logit = self.tower_dnn_final_layer[i](mmoe_outs[i])
            output = self.out[i](tower_dnn_logit)  # logits最后还有一层级
            task_outs.append(output)
        task_outs = torch.cat(task_outs, -1)
        return task_outs # torch.Size([32, 2])
