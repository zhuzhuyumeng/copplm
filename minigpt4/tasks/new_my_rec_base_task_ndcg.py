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
import pandas as pd  # <-- 确保 pandas 已导入


# --- (u_dcg 和 compute_u_ndcg 保持不变) ---
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
    ndcg_list = []
    only_one_class = 0

    for ui, pre_and_true in candidates_dict.items():
        pre_i, label_i = pre_and_true
        ui_ndcg = u_dcg(pre_i, label_i)
        if ui_ndcg >= 0:
            ndcg_list.append(ui_ndcg)
            computed_u.append(ui)
        else:
            only_one_class += 1

    ndcg_for_user = np.array(ndcg_list)
    print("computed user (for nDCG):", ndcg_for_user.shape[0], "can not users:", only_one_class)
    u_ndcg = ndcg_for_user.mean()
    print("u-nDCG for validation Cost:", time.time() - start_time, 'u-nDCG:', u_ndcg)
    return u_ndcg, computed_u, ndcg_for_user


# --- (uAUC_me 保持不变) ---
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
        except ValueError:
            only_one_class += 1

    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time() - start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user


# --- (gather_tensor 保持不变) ---
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

    # --- [!!! 关键修复 1 !!!] ---
    # 修改 before_evaluation 以“锁存”来自 train.py 的流行度数据
    # 防止它被 runner 的后续调用覆盖
    def before_evaluation(self, model, dataset, **kwargs):

        if "item_pop_dict" in kwargs:
            # 这是来自 train.py  的有效调用，我们存储它
            self.item_pop_dict = kwargs.get("item_pop_dict", {})
            self.total_item_num = kwargs.get("total_item_num", 1)
            self.k_for_bias = kwargs.get("k_for_bias", 10)
        else:
            # 这是来自 runner  的常规调用
            # 仅当属性*不存在*时才设置默认值
            if not hasattr(self, "item_pop_dict"):
                self.item_pop_dict = {}
            if not hasattr(self, "total_item_num"):
                self.total_item_num = 1
            if not hasattr(self, "k_for_bias"):
                self.k_for_bias = 10

        # 警告逻辑保持不变
        if not self.item_pop_dict:
            logging.warning("item_pop_dict not provided to RecBaseTask. Popularity metrics will be 0.")
        if self.total_item_num == 1 and not self.item_pop_dict:
            logging.warning("total_item_num not provided. Coverage metric will be inaccurate.")
        pass

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(selfF):
        raise NotImplementedError

    def evaluation(self, model, data_loaders, cuda_enabled=True):
        model = model.eval()
        metric_logger = MetricLogger(delimiter="  ")
        # (日志记录器保持不变) ...
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("acc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        header = "Evaluation"
        print_freq = len(data_loaders.loaders[0]) // 5

        results_loss = []
        k = 0
        use_auc = False
        for data_loader in data_loaders.loaders:
            results_logits = []
            labels = []
            users = []
            item_ids = []  # <-- [偏差指标] 收集 Item ID

            for samples in metric_logger.log_every(data_loader, print_freq, header):
                samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
                eval_output = self.valid_step(model=model, samples=samples)

                if 'logits' in eval_output.keys():
                    use_auc = True

                    # [!!! 关键修复 !!!]
                    # 使用 np.atleast_1d() 确保
                    # 即使 batch_size=1，我们 append 的也是 1D 数组，而不是 0D 标量。
                    # 这可以防止 np.concatenate 创建 'object' dtype 数组，从而修复 pandas TypeError。

                    users.append(np.atleast_1d(samples['UserID'].detach().cpu().numpy()))
                    item_ids.append(np.atleast_1d(samples['TargetItemID'].detach().cpu().numpy()))

                    # 1. 获取原始 logits
                    logits_for_metrics = eval_output['logits']

                    # 2. 存储原始概率用于 AUC/nDCG
                    results_logits.append(np.atleast_1d(logits_for_metrics.detach().cpu().numpy()))
                    labels.append(np.atleast_1d(samples['label'].detach().cpu().numpy()))

                    logits_for_acc = logits_for_metrics.clone()
                    pred_labels = (logits_for_acc > 0.5).float()
                    acc = (pred_labels == samples['label']).sum() / samples['label'].shape[0]
                    metric_logger.update(acc=acc.item())
                else:
                    metric_logger.update(acc=0)

                metric_logger.update(loss=eval_output['loss'].item())
                torch.cuda.empty_cache()

            # --- [!!! 关键修复 3 !!!] ---
            # 在 Numpy 层面拼接列表中的数组
            # 这可以正确处理最后一个 batch 长度不一致的问题
            # --- [!!! 关键修复 3 !!!] ---
            # 在 Numpy 层面拼接列表中的数组
            #
            # [!!! 最终修复 !!!]
            # 我们使用 .astype(np.float64) 强制转换数组的数据类型。
            # 这将彻底解决 pandas 因 'dtype=object' 而导致的 'TypeError: No matching signature found'。

            # --- [!!! 关键修复 4 !!!] ---
            # 修复 'TypeError: slice indices must be integers'
            #
            # 1. results_logits_np (Scores) 必须是 float
            results_logits_np = np.concatenate(results_logits).astype(np.float64)

            # 2. labels_np (Labels) 必须是 int
            labels_np = np.concatenate(labels).astype(np.int64)

            # 3. users_np (UserIDs) 必须是 int
            users_np = np.concatenate(users).astype(np.int64)

            # 4. item_ids_np (ItemIDs) 必须是 int
            item_ids_np = np.concatenate(item_ids).astype(np.int64)
            # --- [修复结束] ---
            # --- [修复结束] ---
            # --- [修复结束] ---

            auc = 0
            uauc = 0
            u_ndcg = 0
            ap_k = 0.0  # <-- [偏差指标] 初始化
            coverage_k = 0.0  # <-- [偏差指标] 初始化
            gini_k = 0.0  # <-- [偏差指标] 初始化

            if is_dist_avail_and_initialized():
                # --- (DDP 逻辑，将 _np 数组转回 tensor 以便收集) ---
                print("wating comput metrics.....")
                results_logits_ = torch.from_numpy(results_logits_np).to(eval_output['logits'].device)
                labels_ = torch.from_numpy(labels_np).to(eval_output['logits'].device)
                users_ = torch.from_numpy(users_np).to(eval_output['logits'].device)
                item_ids_ = torch.from_numpy(item_ids_np).to(eval_output['logits'].device)

                rank = dist.get_rank()
                gathered_labels = [labels_.clone() for _ in range(dist.get_world_size())]
                gathered_logits = [results_logits_.clone() for _ in range(dist.get_world_size())]
                gathered_users = [users_.clone() for _ in range(dist.get_world_size())]
                gathered_item_ids = [item_ids_.clone() for _ in range(dist.get_world_size())]

                dist.all_gather(gathered_labels, labels_)
                dist.all_gather(gathered_logits, results_logits_)
                dist.all_gather(gathered_users, users_)
                dist.all_gather(gathered_item_ids, item_ids_)

                labels_a = torch.cat(gathered_labels, dim=0).flatten().cpu().numpy()
                results_logits_a = torch.cat(gathered_logits, dim=0).flatten().cpu().numpy()
                users_a = torch.cat(gathered_users, dim=0).flatten().cpu().numpy()
                item_ids_a = torch.cat(gathered_item_ids, dim=0).flatten().cpu().numpy()

                print("computing metrics....")
                auc = roc_auc_score(labels_a, results_logits_a)
                uauc, _, _ = uAUC_me(users_a, results_logits_a, labels_a)
                u_ndcg, _, _ = compute_u_ndcg(users_a, results_logits_a, labels_a)

                # <-- [偏差指标] 在 DDP 模式下计算
                ap_k, coverage_k, gini_k = self.calculate_bias_metrics(
                    users_a, item_ids_a, results_logits_a
                )
                print("finished comput metrics.....")
            else:
                # --- (非 DDP 逻辑，直接使用 _np 数组) ---
                auc = roc_auc_score(labels_np, results_logits_np)
                uauc, _, _ = uAUC_me(users_np, results_logits_np, labels_np)
                u_ndcg, _, _ = compute_u_ndcg(users_np, results_logits_np, labels_np)

                # <-- [偏差指标] 在非 DDP 模式下计算
                ap_k, coverage_k, gini_k = self.calculate_bias_metrics(
                    users_np, item_ids_np, results_logits_np
                )

            if is_dist_avail_and_initialized():
                dist.barrier()

            metric_logger.synchronize_between_processes()

            # (rank 0 日志保持不变)
            auc_rank0 = 0
            if use_auc and not is_dist_avail_and_initialized():  # 仅在非 DDP 时计算 rank0
                auc_rank0 = roc_auc_score(labels_np, results_logits_np)

            # --- [偏差指标] 添加到最终日志 ---
            logging.info(
                "Averaged stats: " + str(metric_logger.global_avg()) +
                " ***auc: " + str(auc) +
                " ***uauc: " + str(uauc) +
                " ***u-nDCG: " + str(u_ndcg) +
                # <-- 新增日志
                f" ***AP@{self.k_for_bias}: " + str(ap_k) +
                f" ***Coverage@{self.k_for_bias}: " + str(coverage_k) +
                f" ***Gini@{self.k_for_bias}: " + str(gini_k)
            )
            print("rank_0 auc:", str(auc_rank0))

            if use_auc:
                # --- [偏差指标] 添加到返回结果 ---
                results = {
                    'agg_metrics': auc,
                    'acc': metric_logger.meters['acc'].global_avg,
                    'loss': metric_logger.meters['loss'].global_avg,
                    'auc': auc,
                    'uauc': uauc,
                    'u_ndcg': u_ndcg,
                    f'AP@{self.k_for_bias}': ap_k,
                    f'Coverage@{self.k_for_bias}': coverage_k,
                    f'Gini@{self.k_for_bias}': gini_k
                }
            else:
                results = {
                    'agg_metrics': -metric_logger.meters['loss'].global_avg,
                }

        return results

    # --- [新增] Gini 系数辅助函数 ---
    def calculate_gini(self, item_counts):
        """计算 Gini 系数"""
        if item_counts.empty:
            return 0.0

        vals = item_counts.sort_values().values
        n = len(vals)
        cum_vals = np.cumsum(vals)
        total_sum = cum_vals[-1]

        if total_sum == 0:
            return 0.0

        lorenz_curve = cum_vals / total_sum
        area_under_lorenz = np.sum(lorenz_curve) / n
        area_perfect_equality = 0.5

        gini_coefficient = (area_perfect_equality - area_under_lorenz) / area_perfect_equality

        return gini_coefficient

    # --- [新增] 统一计算偏见指标的函数 ---
    def calculate_bias_metrics(self, uids, item_ids, scores):
        """
        计算 AP@K, Coverage@K, 和 Gini@K.
        """
        K = self.k_for_bias
        pop_dict = self.item_pop_dict
        total_item_num = self.total_item_num

        if not pop_dict:
            logging.warning("item_pop_dict is empty. Bias metrics will be 0.")
            return 0.0, 0.0, 0.0

        # 1. 构建 DataFrame 并获取 Top-K
        df = pd.DataFrame({
            'uid': uids,
            'item_id': item_ids,
            'score': scores
        })

        all_top_k_items = []
        all_top_k_pops = []

        grouped = df.groupby('uid')

        for uid, user_df in grouped:
            # 获取每个用户的 Top-K
            # 修复：如果用户数据少于K，取全部
            if len(user_df) < K:
                top_k_df = user_df
            else:
                top_k_df = user_df.nlargest(K, 'score')

            top_k_item_ids = top_k_df['item_id'].values
            all_top_k_items.extend(top_k_item_ids)

            # 记录流行度F
            for item_id in top_k_item_ids:
                all_top_k_pops.append(pop_dict.get(item_id, 0))

        # 2. 计算 AP@K (Average Popularity)
        ap_k = np.mean(all_top_k_pops) if all_top_k_pops else 0.0

        # 3. 计算 Coverage@K
        unique_recommended_items = set(all_top_k_items)
        coverage_k = len(unique_recommended_items) / total_item_num

        # 4. 计算 Gini@K
        item_counts = pd.Series(all_top_k_items).value_counts()
        gini_k = self.calculate_gini(item_counts)

        print(f"Bias Metrics @{K}: AP={ap_k:.4f}, Coverage={coverage_k:.4f}, Gini={gini_k:.4f}")

        return ap_k, coverage_k, gini_k