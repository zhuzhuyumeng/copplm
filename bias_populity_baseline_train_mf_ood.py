from minigpt4.models.rec_model import MatrixFactorization
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random

import os
from minigpt4.tasks import base_task
import time
import numpy as np


# --- [新增] 1. Gini 系数计算函数 ---
def calculate_gini(item_counts):
    """计算 Gini 系数"""
    if item_counts.empty:
        return 0.0

    # 排序
    vals = item_counts.sort_values().values
    n = len(vals)
    cum_vals = np.cumsum(vals)
    total_sum = cum_vals[-1]

    if total_sum == 0:
        return 0.0

    # 计算洛伦兹曲线下的面积
    lorenz_curve = cum_vals / total_sum
    area_under_lorenz = np.sum(lorenz_curve) / n
    area_perfect_equality = 0.5

    # Gini = (Area_Perfect - Area_Lorenz) / Area_Perfect
    gini_coefficient = (area_perfect_equality - area_under_lorenz) / area_perfect_equality

    return gini_coefficient


# --- [新增] 2. 偏见指标计算函数 ---
def calculate_bias_metrics(uids, item_ids, scores, item_pop_dict, total_item_num, K=10):
    """
    计算 AP@K, Coverage@K, 和 Gini@K.
    """
    if not item_pop_dict:
        print("item_pop_dict is empty. Bias metrics will be 0.")
        return 0.0, 0.0, 0.0

    # 1. 构建 DataFrame 并获取 Top-K
    df = pd.DataFrame({
        'uid': uids,
        'item_id': item_ids,
        'score': scores
    })

    all_top_k_items = []
    all_top_k_pops = []

    # 这是一个耗时操作，对于大规模测试集可能较慢
    # 优化：使用 groupby 和 nlargest
    grouped = df.groupby('uid')

    for uid, user_df in grouped:
        # 获取每个用户的 Top-K
        if len(user_df) < K:
            top_k_df = user_df  # 如果候选集不足K个，全选
        else:
            top_k_df = user_df.nlargest(K, 'score')

        top_k_item_ids = top_k_df['item_id'].values
        all_top_k_items.extend(top_k_item_ids)

        # 记录流行度
        for item_id in top_k_item_ids:
            all_top_k_pops.append(item_pop_dict.get(item_id, 0))

    # 2. 计算 AP@K (Average Popularity)
    ap_k = np.mean(all_top_k_pops) if all_top_k_pops else 0.0

    # 3. 计算 Coverage@K
    unique_recommended_items = set(all_top_k_items)
    coverage_k = len(unique_recommended_items) / total_item_num

    # 4. 计算 Gini@K
    item_counts = pd.Series(all_top_k_items).value_counts()
    gini_k = calculate_gini(item_counts)

    print(f"*** Bias Metrics @{K}: AP={ap_k:.4f}, Coverage={coverage_k:.4f}, Gini={gini_k:.4f}")

    return ap_k, coverage_k, gini_k


def uAUC_me(user, predict, label):
    if not isinstance(predict, np.ndarray):
        predict = np.array(predict)
    if not isinstance(label, np.ndarray):
        label = np.array(label)
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
        # print(index_ui, predict.shape)
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
        except:
            only_one_class += 1
            # print("only one class")

    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time() - start_time, 'uauc:', uauc)
    return uauc, computed_u, auc_for_user


def u_dcg(predict, label):
    """计算单个用户的全列表 DCG/IDCG"""
    pos_num = int(label.sum())  # 确保是整数
    labels_unique = np.unique(label)
    # 如果样本太少或没有正/负样本，无法计算，返回 -1
    if labels_unique.shape[0] < 2 or label.shape[-1] < 2:
        return -1

    # 1. 全列表排序
    ranked_id = np.argsort(-predict)
    ranked_label = label[ranked_id]

    # 2. 计算折扣因子 (基于全列表长度)
    # ranked_label.shape[-1] 是该用户所有样本的数量
    flag = 1.0 / np.log2(np.arange(ranked_label.shape[-1]) + 2.0)

    # 3. 计算 DCG (全列表求和)
    dcg = (ranked_label * flag).sum()

    # 4. 计算 IDCG
    # 理想情况下，所有正样本都排在最前面
    idcg = flag[:pos_num].sum()

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_full_ndcg(user, predict, label):
    """计算所有用户的平均全列表 nDCG"""
    if not isinstance(predict, np.ndarray):
        predict = np.array(predict)
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    # 高效分组逻辑 (同 uAUC_me)
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)
    index = np.argsort(inverse)

    ndcg_sum = 0
    computed_cnt = 0
    total_num = 0
    k = 0

    for u_i in u:
        start_id = total_num
        end_id = total_num + counts[k]
        total_num += counts[k]
        k += 1

        idx_ui = index[start_id:end_id]
        u_pred = predict[idx_ui]
        u_label = label[idx_ui]

        # 计算该用户的全列表 nDCG
        val = u_dcg(u_pred, u_label)

        if val >= 0:  # 过滤掉无效用户 (-1)
            ndcg_sum += val
            computed_cnt += 1

    if computed_cnt == 0:
        return 0.0

    return ndcg_sum / computed_cnt


# --- [新增/找回] 3. 平均流行度计算函数 ---
def calculate_avg_popularity(predictions, item_ids, item_popularity_map, threshold=0.5):
    """
    计算推荐物品的平均流行度。
    """
    # 因为模型的输出是 logits, 需要 sigmoid
    predictions = 1.0 / (1.0 + np.exp(-np.array(predictions)))
    item_ids = np.array(item_ids)

    # 找到模型“推荐”的物品（即预测概率 > threshold）
    recommended_indices = np.where(predictions >= threshold)
    if len(recommended_indices[0]) == 0:
        return 0.0

    recommended_item_ids = item_ids[recommended_indices]

    # 确保索引在范围内 (防止 OOD 物品越界)
    valid_mask = recommended_item_ids < len(item_popularity_map)
    recommended_item_ids = recommended_item_ids[valid_mask]

    if len(recommended_item_ids) == 0:
        return 0.0

    avg_pop = np.mean(item_popularity_map[recommended_item_ids])
    return avg_pop

class early_stoper(object):
    def __init__(self, ref_metric='valid_auc', incerase=True, patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience = patience
        # self.metrics = None

    def _registry(self, metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count >= self.patience:
            return True
        else:
            return False


# 请用此函数替换您脚本中的 run_a_trail 函数

def run_a_trail(train_config, log_file=None, save_mode=False, save_file=None, need_train=True, warm_or_cold=None):
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
    data_dir = "collm-datasets/ml-1m/ml-1m/"

    # 1. 加载原始 DataFrame 以计算流行度字典
    train_df_raw = pd.read_pickle(data_dir + "train_ood2.pkl")
    train_data = train_df_raw[['uid', 'iid', 'label']].values

    # 2. 计算流行度字典 (用于 Gini/AP 计算)
    print("正在计算全局流行度...")
    item_counts = train_df_raw['iid'].value_counts()
    # log(count+1) 用于平滑流行度，如果只是想要原始交互次数，去掉 np.log 即可
    item_pop_dict = (np.log(item_counts + 1)).to_dict()

    valid_data = pd.read_pickle(data_dir + "valid_ood2.pkl")[['uid', 'iid', 'label']].values
    test_data = pd.read_pickle(data_dir + "test_ood2.pkl")[['uid', 'iid', 'label']].values

    user_num = max(train_data[:, 0].max(), valid_data[:, 0].max(), test_data[:, 0].max()) + 1
    item_num = max(train_data[:, 1].max(), valid_data[:, 1].max(), test_data[:, 1].max()) + 1

    # 3. [关键] 生成 item_popularity_np (用于 avg_pop 计算)
    item_popularity_np = item_counts.reindex(range(item_num), fill_value=0).values

    # 获取总物品数 (用于 Coverage 计算)
    total_item_num = item_num

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir + "test_warm_cold_ood2.pkl")[['uid', 'iid', 'label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid', 'iid', 'label']].values
            print("warm data size:", test_data.shape[0])
        else:
            test_data = pd.read_pickle(data_dir + "test_warm_cold_ood2.pkl")[['uid', 'iid', 'label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid', 'iid', 'label']].values
            print("cold data size:", test_data.shape[0])

    print("user nums:", user_num, "item nums:", item_num)

    mf_config = {
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
    }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=train_config['batch_size'], shuffle=False)

    model = MatrixFactorization(mf_config).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc', incerase=True, patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    # ================= 纯评估模式 (need_train=False) =================
    if not need_train:
        print(f"Loading model from {save_file}")
        model.load_state_dict(torch.load(save_file))
        model.eval()

        # --- Validation Eval ---
        pre = []
        label = []
        users = []
        items = []
        for batch_id, batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            users.extend(batch_data[:, 0].cpu().numpy())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:, -1].cpu().numpy())
            items.extend(batch_data[:, 1].cpu().numpy())

        valid_auc = roc_auc_score(label, pre)
        valid_uauc, _, _ = uAUC_me(users, pre, label)
        valid_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
        valid_ndcg = compute_full_ndcg(users, pre, label)

        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre >= thre] = 1
        pre[pre < thre] = 0
        val_acc = (label == pre).mean()

        # --- Test Eval ---
        pre = []
        label = []
        users = []
        items = []
        for batch_id, batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:, -1].cpu().numpy())
            users.extend(batch_data[:, 0].cpu().numpy())
            items.extend(batch_data[:, 1].cpu().numpy())

        test_auc = roc_auc_score(label, pre)
        test_uauc, _, _ = uAUC_me(users, pre, label)
        test_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
        test_ndcg = compute_full_ndcg(users, pre, label)

        # --- [修改] 计算偏见指标并捕获返回值 ---
        # 为了节省时间，Bias指标可以只在Test上算，或者Validation也算
        # 这里我们在 Test 上计算详细的 Bias
        print("Computing detailed Bias Metrics (AP, Coverage, Gini)...")
        test_ap, test_cov, test_gini = calculate_bias_metrics(
            uids=users,
            item_ids=items,
            scores=pre,
            item_pop_dict=item_pop_dict,
            total_item_num=total_item_num,
            K=10
        )

        # --- [修改] 统一打印所有指标 ---
        print(
            "valid_auc:{:.4f}, valid_uauc:{:.4f}, valid_ndcg:{:.4f}, valid_pop:{:.2f}, acc:{:.4f} | "
            "test_auc:{:.4f}, test_uauc:{:.4f}, test_ndcg:{:.4f}, test_pop:{:.2f} | "
            "test_AP@10:{:.4f}, test_Cov@10:{:.4f}, test_Gini@10:{:.4f}".format(
                valid_auc, valid_uauc, valid_ndcg, valid_avg_pop, val_acc,
                test_auc, test_uauc, test_ndcg, test_avg_pop,
                test_ap, test_cov, test_gini
            ))
        return

    # ================= 训练模式 (need_train=True) =================
    for epoch in range(train_config['epoch']):
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            loss = criterion(ui_matching, batch_data[:, -1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % train_config['eval_epoch'] == 0:
            model.eval()

            # --- Validation ---
            pre = []
            label = []
            users = []
            items = []
            for batch_id, batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                users.extend(batch_data[:, 0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:, -1].cpu().numpy())
                items.extend(batch_data[:, 1].cpu().numpy())

            valid_auc = roc_auc_score(label, pre)
            valid_uauc, _, _ = uAUC_me(users, pre, label)
            valid_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
            valid_ndcg = compute_full_ndcg(users, pre, label)

            # --- Test ---
            pre = []
            label = []
            users = []
            items = []
            for batch_id, batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                users.extend(batch_data[:, 0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:, -1].cpu().numpy())
                items.extend(batch_data[:, 1].cpu().numpy())

            test_auc = roc_auc_score(label, pre)
            test_uauc, _, _ = uAUC_me(users, pre, label)
            test_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
            test_ndcg = compute_full_ndcg(users, pre, label)

            # --- [新增] 训练时也计算 Bias 指标 ---
            # 注意：这会稍微变慢，如果太慢可以设置 if epoch % 10 == 0
            test_ap, test_cov, test_gini = calculate_bias_metrics(
                uids=users, item_ids=items, scores=pre,
                item_pop_dict=item_pop_dict, total_item_num=total_item_num, K=10
            )

            updated = early_stop.update(
                {'valid_auc': valid_auc, 'valid_uauc': valid_uauc, 'test_auc': test_auc, 'test_uauc': test_uauc,
                 'epoch': epoch})

            if updated and save_mode:
                torch.save(model.state_dict(), save_file)

            # --- [修改] 打印包含 Gini 等指标 ---
            print(
                "epoch:{}, valid_auc:{:.4f}, valid_ndcg:{:.4f} | "
                "test_auc:{:.4f}, test_ndcg:{:.4f}, test_pop:{:.2f}, "
                "test_Gini@10:{:.4f}, test_Cov@10:{:.4f} | early_cnt:{}".format(
                    epoch, valid_auc, valid_ndcg,
                    test_auc, test_ndcg, test_avg_pop,
                    test_gini, test_cov, early_stop.reach_count))

            if early_stop.is_stop():
                print("early stop is reached....!")
                break
            if epoch > 500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                break

    print("train_config:", train_config, "\nbest result:", early_stop.best_metric)
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()

if __name__ == '__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_ = [1e-3]  # 1e-2
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [256]
    save_path = "PretrainedModels/mf/"
    f = None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config = {
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch": 1,
                    "patience": 50,
                    "batch_size": 1024
                }
                print(train_config)
                save_path = "collm-trained-models/my-collm-trained-models/mf_0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth"

                # 使用 need_train=False 来运行评估模式
                run_a_trail(train_config=train_config, log_file=f, save_mode=False, save_file=save_path,
                            need_train=False)

    if f is not None:
        f.close()