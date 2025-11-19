from minigpt4.models.rec_base_models import MatrixFactorization, LightGCN 
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from minigpt4.datasets.datasets.rec_gnndataset import GnnDataset
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
# (在 import 语句下面)
import time
import os


def calculate_avg_popularity(predictions, item_ids, item_popularity_map, threshold=0.5):
    """
    计算推荐物品的平均流行度。
    predictions: 模型的预测得分 (logits)
    item_ids: 对应的物品 ID 列表
    item_popularity_map: 一个 numpy 数组，索引是 item_id，值是流行度
    threshold: 判定为“推荐”的阈值
    """
    # 因为模型使用 nn.BCEWithLogitsLoss，predictions 是 logits，需要 sigmoid
    predictions = 1.0 / (1.0 + np.exp(-np.array(predictions)))

    item_ids = np.array(item_ids)

    # 找到模型“推荐”的物品（即预测概率 > 0.5）
    recommended_indices = np.where(predictions >= threshold)
    if len(recommended_indices[0]) == 0:
        return 0.0  # 没有推荐任何物品

    recommended_item_ids = item_ids[recommended_indices]

    # 从流行度图中获取这些物品的流行度并计算均值
    avg_pop = np.mean(item_popularity_map[recommended_item_ids])
    return avg_pop



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

# (确保在脚本顶部有 import numpy as np 和 import time)

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
    if not isinstance(predict, np.ndarray):
        predict = np.array(predict)
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    predict = predict.squeeze()
    label = label.squeeze()
    start_time = time.time()
    u, inverse, counts = np.unique(user, return_inverse=True, return_counts=True)  # sort in increasing
    # ... (函数剩余部分)
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
    # (这部分日志可以保留，但为减少刷屏，我先注释掉)
    # print("only one interaction users (for nDCG):", only_one_interaction)
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
    # print("computed user (for nDCG):", ndcg_for_user.shape[0], "can not users:", only_one_class)
    u_ndcg = ndcg_for_user.mean()
    # print("u-nDCG for validation Cost:", time.time() - start_time, 'u-nDCG:', u_ndcg)
    return u_ndcg, computed_u, ndcg_for_user

class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.embed_size = 64
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = '/home/zyang/code-2022/RecUnlearn/data/'
        self.dataset = 'ml-100k' #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
        self.layer_size='[64,64]'
        self.verbose = 1
        self.Ks='[10]'
        self.data_type='retraining'

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 100
        self.A_split = False
        self.dropout = False
        self.pretrain=0
        self.init_emb=1e-4
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)
    
    def hyper_para_info(self):
        print(self.__dict__)


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
def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None):
    seed=2025
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    args = model_hyparameters()
    args.reset(train_config)
    args.hyper_para_info()

    # load dataset
    # data_dir = "/home/zyang/LLM/MiniGPT-4/dataset/ml-100k/"
    # data_dir = "/home/sist/zyang/LLM/datasets/ml-100k/"
    data_dir = "collm-datasets/ml-1m/ml-1m/"

    # --- 修改开始 ---
    # 1. 将所有数据先加载为 DataFrame
    train_df = pd.read_pickle(data_dir + "train_ood2.pkl")[['uid', 'iid', 'label']]
    valid_df = pd.read_pickle(data_dir + "valid_ood2.pkl")[['uid', 'iid', 'label']]
    test_df = pd.read_pickle(data_dir + "test_ood2.pkl")[['uid', 'iid', 'label']]

    # 2. 从 DataFrame 计算 user_num 和 item_num (更稳妥的方法)
    user_num = max(train_df['uid'].max(), valid_df['uid'].max(), test_df['uid'].max()) + 1
    item_num = max(train_df['iid'].max(), valid_df['iid'].max(), test_df['iid'].max()) + 1

    # 3. 计算流行度
    # .reindex 确保数组大小为 item_num，缺失的物品流行度为 0
    item_popularity = train_df['iid'].value_counts().reindex(range(item_num), fill_value=0)
    item_popularity_np = item_popularity.values

    # 4. 转换为 .values (numpy 数组) 以便 DataLoader 使用
    train_data = train_df.values
    valid_data = valid_df.values
    test_data = test_df.values
    # --- 修改结束 ---
    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([1])][['uid','iid','label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([0])][['uid','iid','label']].values
            print("cold data size:", test_data.shape[0])
            # pass

    # train_config={
    #     "lr": 1e-2,
    #     "wd": 1e-4,
    #     "epoch": 5000,
    #     "eval_epoch":1,
    #     "patience":50,
    #     "batch_size":1024
    # }

    user_num = train_data[:,0].max() + 1
    item_num = train_data[:,1].max() + 1

    lgcn_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size']),
        "embed_size": int(train_config['embedding_size']),
        "data_path": '/home/zyang/code-2022/RecUnlearn/data/',
        "dataset": 'ml-1m', #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
        "layer_size": '[64,64]',

        # lightgcn hyper-parameters
        "gcn_layers": train_config['gcn_layer'],
        "keep_prob" : 0.6,
        "A_n_fold": 100,
        "A_split": False,
        "dropout": False,
        "pretrain": 0,
        "init_emb": 1e-1,
        }
    lgcn_config = omegaconf.OmegaConf.create(lgcn_config)
    gnndata = GnnDataset(lgcn_config, data_dir)
    lgcn_config['user_num'] = int(gnndata.m_users)
    lgcn_config['item_num'] = int(gnndata.n_items)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)





    # model = MatrixFactorization(mf_config).cuda()
    model = LightGCN(lgcn_config).cuda()
    model._set_graph(gnndata.Graph)
    
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre = []
        label = []
        users = []
        items = []
        for batch_id, batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:, -1].cpu().numpy())
            users.extend(batch_data[:, 0].cpu().numpy())
            items.extend(batch_data[:, 1].cpu().numpy())  # (我们之前添加的)

        valid_auc = roc_auc_score(label, pre)
        valid_uauc, _, _ = uAUC_me(users, pre, label)
        valid_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)  # (我们之前添加的)

        # --- 修改点 ---
        # 删除了 group_data_for_ranking 和 ndcg_at_k
        valid_u_ndcg, _, _ = compute_u_ndcg(users, pre, label)
        # --- 修改结束 ---

        # (acc 计算)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre >= thre] = 1
        pre[pre < thre] = 0
        val_acc = (label == pre).mean()

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
            items.extend(batch_data[:, 1].cpu().numpy())  # (我们之前添加的)

        test_auc = roc_auc_score(label, pre)
        test_uauc, _, _ = uAUC_me(users, pre, label)
        test_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)  # (我们之前添加的)

        # --- 修改点 ---
        # 删除了 group_data_for_ranking 和 ndcg_at_k
        test_u_ndcg, _, _ = compute_u_ndcg(users, pre, label)
        # --- 修改结束 ---

        # --- 修改 print 语句 (移除了 @10) ---
        print(
            "valid_auc:{:.4f}, valid_uauc:{:.4f}, valid_u_ndcg:{:.4f}, valid_pop:{:.2f}, acc: {:.4f} | test_auc:{:.4f}, test_uauc:{:.4f}, test_u_ndcg:{:.4f}, test_pop:{:.2f}".format(
                valid_auc, valid_uauc, valid_u_ndcg, valid_avg_pop, val_acc,
                test_auc, test_uauc, test_u_ndcg, test_avg_pop
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
            items = []  # <-- 新增 items 列表
            for batch_id, batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:, -1].cpu().numpy())
                users.extend(batch_data[:, 0].cpu().numpy())
                items.extend(batch_data[:, 1].cpu().numpy())  # <-- 新增收集 items

            valid_auc = roc_auc_score(label, pre)
            valid_uauc, _, _ = uAUC_me(users, pre, label)

            # --- 新增指标 ---
            valid_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
            valid_u_ndcg, _, _ = compute_u_ndcg(users, pre, label)
            # --- 新增结束 ---

            pre = []
            label = []
            users = []
            items = []  # <-- 新增 items 列表
            for batch_id, batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:, -1].cpu().numpy())
                users.extend(batch_data[:, 0].cpu().numpy())
                items.extend(batch_data[:, 1].cpu().numpy())  # <-- 新增收集 items

            test_auc = roc_auc_score(label, pre)
            test_uauc, _, _ = uAUC_me(users, pre, label)

            # --- 新增指标 ---
            test_avg_pop = calculate_avg_popularity(pre, items, item_popularity_np)
            test_u_ndcg, _, _ = compute_u_ndcg(users, pre, label)
            # --- 新增结束 ---

            # --- 更新早停字典 ---
            updated = early_stop.update({
                'valid_auc': valid_auc,
                'valid_uauc': valid_uauc,
                'valid_u_ndcg': valid_u_ndcg,  # <-- 新增
                'valid_avg_pop': valid_avg_pop,  # <-- 新增
                'test_auc': test_auc,
                'test_uauc': test_uauc,
                'test_u_ndcg': test_u_ndcg,  # <-- 新增
                'test_avg_pop': test_avg_pop,  # <-- 新增
                'epoch': epoch
            })
            if updated and save_mode:
                torch.save(model.state_dict(), save_file)

            # --- 更新 print 语句 ---
            print(
                "epoch:{}, valid_auc:{:.4f}, valid_uauc:{:.4f}, valid_u_ndcg:{:.4f}, valid_pop:{:.2f} | test_auc:{:.4f}, test_uauc:{:.4f}, test_u_ndcg:{:.4f}, test_pop:{:.2f}, early_count:{}".format(
                    epoch, valid_auc, valid_uauc, valid_u_ndcg, valid_avg_pop,
                    test_auc, test_uauc, test_u_ndcg, test_avg_pop,
                    early_stop.reach_count
                ))


            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.52")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)



# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4]
#     dw_ = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64, 128, 256]
#     gcn_layers = [1, 2, 3]
#     try:
#         # f = open("ml100k-rec_lgcn_search_lr"+str(lr_[0])+".log",'rw+')
#         # f = open("ood-ml100k-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'rw+')
#         f = open("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'rw+')
#     except:
#         # f = open("ml100k-rec_lgcn_search_lr"+str(lr_[0])+".log",'w+')
#         f = open("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer"+str(lr_[0])+".log",'w+')
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 for gcn_layer in gcn_layers:
#                     train_config={
#                         'lr': lr,
#                         'wd': wd,
#                         'embedding_size': embedding_size,
#                         "epoch": 5000,
#                         "eval_epoch":1,
#                         "patience":100,
#                         "batch_size":2048,
#                         "gcn_layer": gcn_layer
#                     }
#                     print(train_config)
#                     run_a_trail(train_config=train_config, log_file=f, save_mode=False)
#     f.close()



#train_config: {'lr': 0.01, 'wd': 0.0001, 'embedding_size': 64, 'epoch': 5000, 'eval_epoch': 1, 'patience': 100, 'batch_size': 2048, 'gcn_layer': 2}

# {'lr': 0.0001, 'wd': 1e-07, 'embedding_size': 256, 'epoch': 5000, 'eval_epoch': 1, 
# 'patience': 100, 'batch_size': 2048, 'gcn_layer': 2}
# save version....
# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4] #1e-2
#     dw_ = [1e-5]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64]
#     save_path = "/data/zyang/LLM/PretrainedModels/lgcn/"
#     # save_path = "/home/sist/zyang/LLM/PretrainedModels/mf/"
#     # try:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
#     # except:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
#     f=None
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size":2048,
#                     "gcn_layer": 2
#                 }
#                 print(train_config)
#                 save_path += "0918-OODv2_lgcn_ml1m_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path)
#     f.close()


# # with prtrain version:
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-2] #1e-2
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [64]
    save_path = "PretrainedModels/mf"
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
                    "batch_size":1024,
                    "gcn_layer": 2
                }
                # print(train_config)
                # if os.path.exists(save_path + "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"):
                #     save_path += "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
                # else:
                #     save_path += "best_model_d" + str(embedding_size) + ".pth"
                # save_path += "0918-OODv2_lgcn_ml1m_best_model_d64lr-0.01wd0.0001.pth"
                save_path = "collm-trained-models/my-collm-trained-models/lightgcn_0912_ml1m_oodv2_best_model_d256lr-0.001wd0.0001.pth"
                run_a_trail(train_config=train_config, log_file=f, save_mode=True, save_file=save_path, need_train=False, warm_or_cold=None)
    if f is not None:
        f.close()
        





        
            







