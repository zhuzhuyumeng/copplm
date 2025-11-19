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

def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user


def calculate_avg_popularity(predictions, item_ids, item_popularity_map, threshold=0.5):
    """
    计算推荐物品的平均流行度。
    predictions: 模型的预测得分列表 (logits 或 0-1 概率)
    item_ids: 对应的物品 ID 列表
    item_popularity_map: 一个 numpy 数组，索引是 item_id，值是流行度
    threshold: 判定为“推荐”的阈值
    """
    # 因为模型的输出是 logits (BCEWithLogitsLoss), 我们需要 sigmoid 来转成概率
    # 如果你的模型最后有 sigmoid，可以跳过这一步
    # 在这个脚本中，用的是 BCEWithLogitsLoss，所以 pre 是 logits
    predictions = 1.0 / (1.0 + np.exp(-np.array(predictions)))

    item_ids = np.array(item_ids)

    # 找到模型“推荐”的物品（即预测概率 > 0.5）
    # 注意：在OOD设置中，所有标签都是0或1。
    # 我们更关心模型对所有测试样本的打分倾向，而不只是正样本。
    # 也许一个更好的指标是 "所有高分物品的平均流行度"

    # 示例1：计算模型预测为正的物品的平均流行度
    recommended_indices = np.where(predictions >= threshold)
    if len(recommended_indices[0]) == 0:
        return 0.0  # 没有推荐任何物品

    recommended_item_ids = item_ids[recommended_indices]
    avg_pop = np.mean(item_popularity_map[recommended_item_ids])
    return avg_pop

    # 示例2：计算模型对 "实际正样本" 推荐的物品的平均流行度 (需要传入 label)
    # positive_indices = np.where(np.array(labels) == 1)
    # positive_item_ids = item_ids[positive_indices]
    # if len(positive_item_ids) == 0:
    #     return 0.0
    # avg_pop_of_correct = np.mean(item_popularity_map[positive_item_ids])
    # return avg_pop_of_correct

class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
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
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config, log_file=None, save_mode=False, save_file=None, need_train=True, warm_or_cold=None):
    # ... (种子设置代码) ...

    # load dataset
    data_dir = "collm-datasets/ml-1m/ml-1m/"

    # --- 修复开始 ---
    # 1. 将所有数据先加载为 DataFrame
    train_df = pd.read_pickle(data_dir + "train_ood2.pkl")[['uid', 'iid', 'label']]
    valid_df = pd.read_pickle(data_dir + "valid_ood2.pkl")[['uid', 'iid', 'label']]
    test_df = pd.read_pickle(data_dir + "test_ood2.pkl")[['uid', 'iid', 'label']]

    # 2. 现在可以安全地从 DataFrame 计算 user_num 和 item_num
    user_num = max(train_df['uid'].max(), valid_df['uid'].max(), test_df['uid'].max()) + 1
    item_num = max(train_df['iid'].max(), valid_df['iid'].max(), test_df['iid'].max()) + 1

    # 3. 计算流行度
    item_popularity = train_df['iid'].value_counts().reindex(range(item_num), fill_value=0)
    item_popularity_np = item_popularity.values

    # 4. 现在才将它们转换为 .values (numpy 数组) 以便 DataLoader 使用
    train_data = train_df.values
    valid_data = valid_df.values
    test_data = test_df.values
    # --- 修复结束 ---

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            # （我也一并修复了这里的逻辑，以防您使用它）
            test_warm_cold_df = pd.read_pickle(data_dir + "test_warm_cold_ood2.pkl")[['uid', 'iid', 'label', 'warm']]
            test_data = test_warm_cold_df[test_warm_cold_df['warm'].isin([1])][['uid', 'iid', 'label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_warm_cold_df = pd.read_pickle(data_dir + "test_warm_cold_ood2.pkl")[['uid', 'iid', 'label', 'cold']]
            test_data = test_warm_cold_df[test_warm_cold_df['cold'].isin([1])][['uid', 'iid', 'label']].values
            print("cold data size:", test_data.shape[0])
            # pass

    print("user nums:", user_num, "item nums:", item_num)

    # ... (函数的其余部分保持不变) ...

    # if warm_or_cold is not None:
    #     if warm_or_cold == 'warm':
    #         test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
    #         test_data = test_data[test_data['not_cold'].isin([1])][['uid','iid','label']].values
    #         print("warm data size:", test_data.shape[0])
    #         # pass
    #     else:
    #         test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
    #         test_data = test_data[test_data['not_cold'].isin([0])][['uid','iid','label']].values
    #         print("cold data size:", test_data.shape[0])
    #         # pass
    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_warm_cold_ood2.pkl")[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            # pass
    

    print("user nums:", user_num, "item nums:", item_num)

    mf_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
        }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)





    model = MatrixFactorization(mf_config).cuda()
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'],weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:  # 无需训练
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre = []
        label = []
        users = []
        items = []  # <-- 新增
        for batch_id, batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            users.extend(batch_data[:, 0].cpu().numpy())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:, -1].cpu().numpy())
            items.extend(batch_data[:, 1].cpu().numpy())  # <-- 新增

        valid_auc = roc_auc_score(label, pre)
        valid_uauc, _, _ = uAUC_me(users, pre, label)

        # --- 新增 ---
        valid_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
        # --- 新增结束 ---

        # (保留您原来的 acc 计算)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre >= thre] = 1
        pre[pre < thre] = 0
        val_acc = (label == pre).mean()

        pre = []
        label = []
        users = []
        items = []  # <-- 新增
        for batch_id, batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:, -1].cpu().numpy())
            users.extend(batch_data[:, 0].cpu().numpy())
            items.extend(batch_data[:, 1].cpu().numpy())  # <-- 新增

        test_auc = roc_auc_score(label, pre)
        test_uauc, _, _ = uAUC_me(users, pre, label)

        # --- 新增 ---
        test_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
        # --- 新增结束 ---

        # --- 修改 print 语句 ---
        print(
            "valid_auc:{:.4f}, valid_uauc:{:.4f}, test_auc:{:.4f}, test_uauc:{:.4f}, valid_pop:{:.2f}, test_pop:{:.2f}, acc: {:.4f}".format(
                valid_auc, valid_uauc, test_auc, test_uauc, valid_avg_pop, test_avg_pop, val_acc
            ))
        return
    

    for epoch in range(train_config['epoch']):
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:,0].long(),batch_data[:,1].long())
            loss = criterion(ui_matching,batch_data[:,-1].float())
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % train_config['eval_epoch'] == 0:
            model.eval()
            pre = []
            label = []
            users = []
            items = []  # <-- 新增
            for batch_id, batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                users.extend(batch_data[:, 0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:, -1].cpu().numpy())
                items.extend(batch_data[:, 1].cpu().numpy())  # <-- 新增

            valid_auc = roc_auc_score(label, pre)
            valid_uauc, _, _ = uAUC_me(users, pre, label)

            # --- 在这里计算新指标 ---
            valid_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)

            # ... (对 test_data_loader 重复相同操作)
            pre = []
            label = []
            users = []
            items = []  # <-- 新增
            for batch_id, batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                users.extend(batch_data[:, 0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:, -1].cpu().numpy())
                items.extend(batch_data[:, 1].cpu().numpy())  # <-- 新增

            test_auc = roc_auc_score(label, pre)
            test_uauc, _, _ = uAUC_me(users, pre, label)

            # --- 在这里计算新指标 ---
            test_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
            updated = early_stop.update({
                'valid_auc': valid_auc,
                'valid_uauc': valid_uauc,
                'test_auc': test_auc,
                'test_uauc': test_uauc,
                'valid_avg_pop': valid_avg_pop,  # <-- 新增
                'test_avg_pop': test_avg_pop,  # <-- 新增
                'epoch': epoch
            })
            if updated and save_mode:
                torch.save(model.state_dict(), save_file)

            # 更新 print 语句
            print("epoch:{}, valid_auc:{:.4f}, test_auc:{:.4f}, valid_pop:{:.2f}, test_pop:{:.2f}, early_count:{}".format(
                    epoch, valid_auc, test_auc, valid_avg_pop, test_avg_pop, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()


if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-3] #1e-2
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [256]
    save_path = "PretrainedModels/mf/"
    # try:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
    # except:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch":1,
                    "patience":50,
                    "batch_size":1024
                }
                print(train_config)
                # save_path = "/data/zyang/LLM/PretrainedModels/mf/0912_ml100k_oodv2_best_model_d64lr-0.001wd0.0001.pth"
                # save_path = "collm-trained-models/0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth" # 111
                save_path = "collm-trained-models/my-collm-trained-models/mf_0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth" # 222
                # if os.path.exists(save_path + "0912_ml100k_oodv2_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"):
                #     save_path += "0912_ml100k_oodv2_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
                #     print(save_path)
                # else:
                #     save_path += "best_model_d" + str(embedding_size) + ".pth"

                # run_a_trail(train_config=train_config, log_file=f, save_mode=False,save_file=save_path,need_train=False,warm_or_cold='warm')
                run_a_trail(train_config=train_config, log_file=f, save_mode=False,save_file=save_path,need_train=False)

    if f is not None:
        f.close()
        





        
            







