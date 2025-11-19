"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue, MetricLogger_auc, SmoothedValue_v2
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from transformers import GenerationConfig
from sklearn.metrics import roc_auc_score, accuracy_score
from minigpt4.tasks.base_task import BaseTask
import time
import numpy as np




def u_dcg(predict, label):
    pos_num = label.sum()
    labels_unique = np.unique(label)
    if labels_unique.shape[0] < 2 or label.shape[-1] < 2:
        return -1
    ranked_id = np.argsort(-predict)
    ranked_label = label[ranked_id]
    flag = 1.0 / np.log2(np.arange(ranked_label.shape[-1]) + 2.0)
    dcg = (ranked_label * flag).sum()
    idcg = flag[:pos_num].sum()
    return dcg / idcg


# 将这个函数替换掉原来的 compute_dcg 函数
def compute_u_ndcg(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)  # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]

        k += 1
    print("only one interaction users (for nDCG):", only_one_interaction)
    ndcg_list = []  # <--- 修改了变量名
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true

        ui_ndcg = u_dcg(pre_i, label_i)  # <--- 修改了变量名 (u_dcg 是nDCG计算函数)

        if ui_ndcg >= 0:
            ndcg_list.append(ui_ndcg)  # <--- 修改了变量名
            computed_u.append(ui)
        else:
            only_one_class += 1
            # print("only one class")

    ndcg_for_user = np.array(ndcg_list)  # <--- 修改了变量名
    print("computed user (for nDCG):", ndcg_for_user.shape[0], "can not users:", only_one_class)
    u_ndcg = ndcg_for_user.mean()  # <--- 修改了变量名
    print("u-nDCG for validation Cost:", time.time() - start_time, 'u-nDCG:', u_ndcg)  # <--- 修改了日志
    return u_ndcg, computed_u, ndcg_for_user  # <--- 返回 u_ndcg


def uAUC_me(user, predict, label):
    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)  # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id, end_id = total_num, total_num + counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts == 1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]

        k += 1
    print("only one interaction users:", only_one_interaction)
    auc = []
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i, pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except ValueError:  # <--- [修改点] 收紧了 except
            only_one_class += 1
            # print("only one class")

    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time() - start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user


# Function to gather tensors across processes
def gather_tensor(tensor, dst=0):
    if dist.is_available():
        world_size = dist.get_world_size()
        if world_size > 1:
            if not isinstance(tensor, list):
                tensor = [tensor]

            gathered_tensors = [torch.empty_like(t) for t in tensor]
            dist.gather(tensor, gathered_tensors, dst=dst)

            return gathered_tensors
        else:
            return tensor
    else:
        return tensor


class RecBaseTask(BaseTask):
    def valid_step(self, model, samples):
        outputs = model.generate_for_samples(samples)
        return outputs
        # raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        pass
        # model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(selfF):
        raise NotImplementedError

    # ... (你注释掉的旧 evaluation 函数) ...

    def evaluation(self, model, data_loaders, cuda_enabled=True):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        auc_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        auc_logger.add_meter("auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        # TODO make it configurable
        print_freq = len(data_loaders.loaders[0]) // 5  # 10

        results = []
        results_loss = []

        k = 0
        use_auc = False
        for data_loader in data_loaders.loaders:
            results_logits = []
            labels = []
            users = []
            for samples in metric_logger.log_every(data_loader, print_freq, header):
                # samples = next(data_loader)
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                eval_output = self.valid_step(model=model, samples=samples)
                # results_loss.append(eval_output['loss'].item())
                if 'logits' in eval_output.keys():
                    use_auc = True
                    users.extend(samples['UserID'].detach().cpu().numpy())

                    # --- [修改点] 使用 .clone() 来确保安全 ---
                    # 1. 获取原始 logits
                    logits_for_metrics = eval_output['logits']

                    # 2. 存储原始概率用于 AUC/nDCG
                    results_logits.extend(logits_for_metrics.detach().cpu().numpy())
                    labels.extend(samples['label'].detach().cpu().numpy())

                    # 3. 创建一个 *副本* 来计算 acc
                    logits_for_acc = logits_for_metrics.clone()

                    # 4. 在 *副本* 上进行二值化操作
                    pred_labels = (logits_for_acc > 0.5).float()

                    # 5. 计算 acc
                    acc = (pred_labels == samples['label']).sum() / samples['label'].shape[0]
                    metric_logger.update(acc=acc.item())
                    # --- 修改结束 ---

                else:
                    metric_logger.update(acc=0)

                metric_logger.update(loss=eval_output['loss'].item())
                torch.cuda.empty_cache()

            results_logits_ = torch.tensor(results_logits).to(eval_output['logits'].device).contiguous()
            labels_ = torch.tensor(labels).to(eval_output['logits'].device).contiguous()
            users_ = torch.tensor(users).to(eval_output['logits'].device).contiguous()

            auc = 0
            uauc = 0  # <-- 初始化
            u_ndcg = 0  # <-- 初始化

            if is_dist_avail_and_initialized():
                print("wating comput metrics.....")  # <-- 修改日志
                rank = dist.get_rank()
                gathered_labels = [labels_.clone() for _ in range(dist.get_world_size())]
                gathered_logits = [results_logits_.clone() for _ in range(dist.get_world_size())]
                gathered_users = [users_.clone() for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_labels, labels_)
                dist.all_gather(gathered_logits, results_logits_)
                dist.all_gather(gathered_users, users_)

                labels_a = torch.cat(gathered_labels, dim=0).flatten().cpu().numpy()
                results_logits_a = torch.cat(gathered_logits, dim=0).flatten().cpu().numpy()
                users_a = torch.cat(gathered_users, dim=0).flatten().cpu().numpy()
                print("computing metrics....")  # <-- 修改日志

                # 1. 计算全局 AUC
                auc = roc_auc_score(labels_a, results_logits_a)

                # 2. 计算 user-averaged AUC (uAUC)
                uauc, _, _ = uAUC_me(users_a, results_logits_a, labels_a)

                # 3. 计算 user-averaged nDCG (u-nDCG)
                u_ndcg, _, _ = compute_u_ndcg(users_a, results_logits_a, labels_a)

                print("finished comput metrics.....")  # <-- 修改日志
            else:
                labels_np = labels_.cpu().numpy()
                logits_np = results_logits_.cpu().numpy()
                users_np = users_.cpu().numpy()

                # 1. 计算全局 AUC
                auc = roc_auc_score(labels_np, logits_np)

                # 2. 计算 user-averaged AUC (uAUC)
                uauc, _, _ = uAUC_me(users_np, logits_np, labels_np)

                # 3. 计算 user-averaged nDCG (u-nDCG)
                u_ndcg, _, _ = compute_u_ndcg(users_np, logits_np, labels_np)

            if is_dist_avail_and_initialized():
                dist.barrier()

            metric_logger.synchronize_between_processes()

            auc_rank0 = 0  # 初始化
            if use_auc:
                auc_rank0 = roc_auc_score(labels_.cpu().numpy(), results_logits_.cpu().numpy())

            # --- 修改日志输出 ---
            logging.info(
                "Averaged stats: " + str(metric_logger.global_avg()) +
                " ***auc: " + str(auc) +
                " ***uauc: " + str(uauc) +
                " ***u-nDCG: " + str(u_ndcg)
            )
            print("rank_0 auc:", str(auc_rank0))

            if use_auc:
                # --- 修改返回的字典 ---
                results = {
                    'agg_metrics': auc,  # 保持 agg_metrics 为 global auc (或者你可以改成uauc)
                    'acc': metric_logger.meters['acc'].global_avg,
                    'loss': metric_logger.meters['loss'].global_avg,
                    'auc': auc,  # 明确添加 global auc
                    'uauc': uauc,  # 明确添加 uauc
                    'u_ndcg': u_ndcg  # 明确添加 u_ndcg
                }
            else:  # only loss usable
                results = {
                    'agg_metrics': -metric_logger.meters['loss'].global_avg,
                }

        return results