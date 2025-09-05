
#TODO: 制作类别关系图



# -*- coding: utf-8 -*-
import os
import sys

from torch.cuda import graph

from Graph.gcn import GCN

sys.path.append('./')
import torch
import json
import torch.nn.functional as F
import argparse
from dataset import load_images
from ClipDZ_model import ClipDZ,ClipDZ_patch
from tqdm import tqdm
import numpy as np
import getpass
from nltk.corpus import wordnet as wn
import random
###没啥用，证书问题
# 导入对应库
import ssl
# 全局关闭ssl验证
ssl._create_default_https_context = ssl._create_unverified_context



'''
开始做消融实验：
1.双分支srsloss：69.7 88.02 77.7
2.仅仅目标域srsloss：
'''



#os.environ["CUDA_VISIBLE_DEVICES"] = ""

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 使用固定的种子
set_seed(42)


def inv_lr_scheduler_warmup(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    # 如果在预热期内，使用初始学习率
    if iter_num < 600:
        lr = init_lr
    else:
        # 将迭代次数调整为从0开始
        adjusted_iter = iter_num - 600
        lr = init_lr * (1 + gamma * adjusted_iter) ** (-power)
    
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer



def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer



def compute_accuracy(test_loader, class_list, device, model, test_classes=None):
    with torch.no_grad():
        model.eval()
        class_dic = {j: i for i, j in enumerate(class_list)}
        if test_classes is not None:
            unpredic_classes = []
            for name in class_list:
                if name not in test_classes:
                    unpredic_classes.append(name)
            # unpredic_classes = list(set(class_list) - set(test_classes))
            unpredic_classes_id = []
            for i in unpredic_classes:
                unpredic_classes_id.append(class_dic[i])
        else:
            unpredic_classes_id = None
        # fetch attributes

        predict_labels_total = []
        re_batch_labels_total = []

        for samples in tqdm(test_loader,dynamic_ncols=True):
            x = samples["image"].to(device)
            y = samples["label"].to(device)
            batch_size = y.shape[0]
            sample_features = model(x, y, is_train='Validation')
            if unpredic_classes_id is not None:
                sample_features[:, unpredic_classes_id] -= 100
            _, predict_labels = torch.max(sample_features, 1)
            predict_labels = predict_labels.cpu().numpy()
            true_labels = y.cpu().numpy()

            predict_labels_total = np.append(predict_labels_total, predict_labels)
            re_batch_labels_total = np.append(re_batch_labels_total, true_labels)

        # compute averaged per class accuracy
        predict_labels_total = np.array(predict_labels_total, dtype='int')
        re_batch_labels_total = np.array(re_batch_labels_total, dtype='int')
        unique_labels = np.unique(re_batch_labels_total)
        acc = 0
        num = 0
        acc_1 = 0
        # print("class num: {}".format(unique_labels.shape[0]))
        for l in unique_labels:
            idx = np.nonzero(re_batch_labels_total == l)[0]

            # acc += accuracy_score(re_batch_labels_total[idx], predict_labels_total[idx])

            acc_class = np.sum(predict_labels_total[idx] == l) / idx.shape[0]
            acc_1 += acc_class

            acc += np.sum(predict_labels_total[idx] == l)
            num += idx.shape[0]
        acc_1 = acc_1 / unique_labels.shape[0]
        acc = acc / num
        model.train()
        return acc, acc_1


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def save_best(zsl_acc, gzsl_unseen_acc, gzsl_seen_acc, gzsl_h, i, best_results):
    if zsl_acc > best_results['best_zsl_acc'][0]:
        best_results['best_zsl_acc'][0] = zsl_acc
        best_results['best_zsl_acc'][1] = i
    if gzsl_unseen_acc > best_results['best_gzsl_unseen_acc'][0]:
        best_results['best_gzsl_unseen_acc'][0] = gzsl_unseen_acc
        best_results['best_gzsl_unseen_acc'][1] = i
    if gzsl_seen_acc > best_results['best_gzsl_seen_acc'][0]:
        best_results['best_gzsl_seen_acc'][0] = gzsl_seen_acc
        best_results['best_gzsl_seen_acc'][1] = i
    if gzsl_h > best_results['best_gzsl_h'][0]:
        best_results['best_gzsl_h'][0] = gzsl_h
        best_results['best_gzsl_h'][1] = i
    return best_results

#正片开始
def main():
    parser = argparse.ArgumentParser(description="ClipDZ Training")

    parser.add_argument(
        "--dataset",
        type=str,
        default="I2AwA",
        choices=["I2AwA", "I2WebV"],
        help="dataset name (default: Office-31)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default= 256        ,
        help="batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--entropy_weight",
        type=float,
        default=3,
        help="weight of entropy loss",
    )
    parser.add_argument(
        "--distribute_weight",
        type=float,
        default=5,
        help="weight of max entropy loss",
    )
    parser.add_argument(
        "--fc_w",
        type=float,
        default=5,
        help="weight of fc loss",
    )
    parser.add_argument('--text', type=str, default='word2vec')
    parser.add_argument('--distribute_bound', type=float, default=100.0)
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument(
        "--item",
        type=int,
        #default=2000,
        default=3000,
        help="number of training items",
    )

    parser.add_argument(
        "--test_item",
        type=int,
        default=300,
        help="the time of test",
    )
    parser.add_argument(
        "--consistency_type",
        type=str,
        default="pair",
        help="the type of consistency loss, entropy/l2",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=100,
        #default=1000,
        help="the weight of consistency loss",
    )

    parser.add_argument(
        "--max_warehoure",
        type=int,
        default=768,
        help="the size of warehoure",
    )

    parser.add_argument('--temperature', type=float, default=0.05,
                        help='temperature (default: 0.05)')

    parser.add_argument(
        "--gpu",
        type=str,
        default='0',
        help="the device location",
    )
    parser.add_argument('--bias_weight', type=float, default=0.3,
                        help='temperature (default: 0.05)')
    args = parser.parse_args()

    dataset_dic = {'I2AwA': {'source': '3D2-1', 'target': 'AwA'},
                   'I2WebV': {'source': 'imagenet', 'target': 'webvision'}}

    json_file_dic = {'I2AwA': "./TextGraph/awa2-split.json",
                     'I2WebV': "./TextGraph/web_wnids.json"}

    #torch.manual_seed(1998)

    '''
    img2loss:0.01
    指定随机种子：无
    源域训练轮次：800
    dim：1024
    loss：nn.MSELoss(reduction='mean')
    struloss:100(默认10)
    仅源epoch以后源域loss权重：1.5（默认1.5）
    
    #改结构
    试一下带残差的注意力adapter
    '''

    if args.dataset == 'I2AwA':
        with open(json_file_dic[args.dataset], 'r') as file:
            classes_json = json.load(file)
        args.seen_classes = classes_json['train_names'] #训练域的类别名
        args.unseen_classes = []
        for i in classes_json['test_names']: #测试域的类别名
            if i not in args.seen_classes:
                args.unseen_classes.append(i)

    if args.dataset == 'I2WebV':
        with open(json_file_dic[args.dataset], 'r') as file:
            classes_json = json.load(file)
        args.seen_classes = classes_json['train']
        args.unseen_classes = []
        for i in classes_json['webvision']:
            if i not in args.seen_classes:
                args.unseen_classes.append(i)

    wordnet2name = {}

    #webV训练时只考虑n1234566形式
    if args.dataset == 'I2AwA':
        for i, j in enumerate(classes_json['train_names']):
            wordnet2name[classes_json['train'][i]] = j
        for i, j in enumerate(classes_json['test_names']):
            wordnet2name[classes_json['test'][i]] = j
    if args.dataset == 'I2WebV':
        pass


    args.source_domain = dataset_dic[args.dataset]['source']
    args.target_domain = dataset_dic[args.dataset]['target']

    print(f'source_domain: {args.source_domain}, target_domain: {args.target_domain}')
    args.source_target_domain = [args.source_domain, args.target_domain]
    save_dir_0 = 'experiment'
    save_dir_1 = args.dataset #I2AwA
    save_dir_2 = args.source_target_domain[0] + '-' + args.source_target_domain[1]
    save_name = 'gcnwarm_{}_withwarehousebanlence{}_entropy_{}_consis_{}'.format(args.fc_w,
                                                                                 args.max_warehoure,
                                                                                 args.entropy_weight,
                                                                                 args.consistency_weight,
                                                                                 ) #按照信息存储
    print(save_name)
    #print(args.use_warehouse)

    #保存实验的数据
    save_dir = os.path.join(".", save_dir_0, save_dir_1, save_dir_2, save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = "cuda:{}".format(args.gpu)
    args.device = device
    data_dim = {'I2AwA': "/sshfs/datasets/YJA_dataset/DAZSL/I2AwA_data/",
                'I2WebV': '/sshfs/datasets/YJA_dataset/DAZSL/I2WebV/'
                }

    source_dim = data_dim[args.dataset] + args.source_target_domain[0]
    target_dim = data_dim[args.dataset] + args.source_target_domain[1]

    #json里存的是类别图，初始图
    graphjson = {
        'I2AwA': "./TextGraph/animals_graph_all.json",
        'I2WebV': './TextGraph/web_graph_all_dense.json'
    }
    graphpath = graphjson[args.dataset]
    graph = json.load(open(graphpath))

    #节点
    wnids = graph['wnids']
    n = len(wnids)
    #边
    edges = graph['edges']
    edges = edges + [(v, u) for (u, v) in edges]  #双向边？
    edges = edges + [(u, u) for u in range(n)]  #自连边？

    #词向量
    word_vectors = torch.tensor(graph['vectors']).to(device)
    word_vectors = F.normalize(word_vectors)
    model = ClipDZ(512, word_vectors, args)
    #这一步是：将图卷积挂载上
    #model.Embedding = GCN(n, edges, word_vectors.shape[1], 1024, 'd1024,d', device)              ##TODO:【注意输出维度】
    model.Embedding = GCN(n, edges, word_vectors.shape[1], 1024, 'd1024,d', device)
    model.embeddings = word_vectors #初始类别词向量


    model = model.to(device)

    #Debug:
    '''
    parameter_list = [{"params": model.CNN.layer3.parameters(), "lr": 1},
                      {"params": model.CNN.layer4.parameters(), "lr": 2}]
    '''
    parameter_list = [{"params": model.Vision_mapnet.parameters(), "lr": 1}]
    #优化器1
    optimizer = torch.optim.SGD(parameter_list, lr=1, momentum=0.9, weight_decay=0.0001, nesterov=True)
    #优化器2：纯仅仅优化GCN
    optimizer_c = torch.optim.SGD(model.Embedding.parameters(), lr=5, momentum=0.9, weight_decay=0.0001, nesterov=True)
    tbar = tqdm(range(args.item),dynamic_ncols=True) #进度条表示

    #记录下两个优化器每组参数的lr
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    param_lr_c = []
    for param_group in optimizer_c.param_groups:
        param_lr_c.append(param_group["lr"])
    best_zsl_acc = [0, 0]
    best_gzsl_h = [0, 0]
    best_gzsl_seen_acc = [0, 0]
    best_gzsl_unseen_acc = [0, 0]
    best_results = {"best_zsl_acc": best_zsl_acc,
                    "best_gzsl_h": best_gzsl_h,
                    "best_gzsl_seen_acc": best_gzsl_seen_acc,
                    "best_gzsl_unseen_acc": best_gzsl_unseen_acc
                    }
    #开始traiin
    model.train()
    class_list = []
    name2iid = {}
    seen_classes_id = []
    unseen_classes_id = []
    # 这个图论文提过了，用imagenet的类别训练再补充的类别（此处思考是不是可以构建类似只是图谱的东西，比如一万个词，然后仅仅拿出这几个cls词计算类别关系）
    ##另外，我突然想到，这个思路会不会降维打击 小样本增量学习
    #for k, wn in enumerate(wnids[127:177]):
    if args.dataset == 'I2AwA':
        for k, wn in enumerate(wnids[:50]):
            class_name = wordnet2name[wn] #提取出类别名
            class_list.append(class_name) #把类别名记录下来
            name2iid[class_name] = k #给每一个类别标上序号
            if class_name in args.seen_classes:
                seen_classes_id.append(k)
            elif class_name in args.unseen_classes:
                unseen_classes_id.append(k)
    if args.dataset == 'I2WebV':
        for k, wn in enumerate(wnids[:5000]):
            class_list.append(wn) #把类别名记录下来
            name2iid[wn] = k #给每一个类别标上序号
            if wn in args.seen_classes:
                seen_classes_id.append(k)
            elif wn in args.unseen_classes:
                unseen_classes_id.append(k)


    print(f'number of seen classes: {len(seen_classes_id)}')
    print(f'number of unseen classes: {len(unseen_classes_id)}')
    train_source_loader = load_images(source_dim, class_list, name2iid, args=args, batch_size=args.batch_size,
                                      split="train",
                                      unseen_classes=args.unseen_classes)
    train_target_loader = load_images(target_dim, class_list, name2iid, batch_size=args.batch_size,
                                      split="train_target")
    test_target_seen_loader = load_images(target_dim, class_list, name2iid, batch_size=args.batch_size, split="val",
                                          unseen_classes=args.unseen_classes)
    test_target_unseen_loader = load_images(target_dim, class_list, name2iid, batch_size=args.batch_size, split="val",
                                            unseen_classes=args.seen_classes)
    train_source_iter = iter(train_source_loader)
    train_target_iter = iter(train_target_loader)

    for i in tbar:
        if ((i + 0) % args.test_item) == 0 and i != 0:
        #if ((i + 0) % args.test_item) == 0:
            zsl_acc, zsl_acc_1 = compute_accuracy(test_target_unseen_loader, class_list, device, model,
                                                  args.unseen_classes)
            gzsl_unseen_acc, gzsl_unseen_acc_1 = compute_accuracy(test_target_unseen_loader, class_list, device, model)
            gzsl_seen_acc, gzsl_seen_acc_1 = compute_accuracy(test_target_seen_loader, class_list, device, model)
            gzsl_h = 2 * (gzsl_unseen_acc * gzsl_seen_acc) / (gzsl_unseen_acc + gzsl_seen_acc)
            best_results = save_best(zsl_acc, gzsl_unseen_acc, gzsl_seen_acc, gzsl_h, i, best_results)

            print(
                "epoch: {}\n zsl_acc: {:.2%}, gzsl_unseen_acc: {:.2%}, gzsl_seen_acc: {:.2%}, gzsl_unseen_class: {:.2%}, gzsl_seen_class: {:.2%}".format(
                    i,
                    zsl_acc,
                    gzsl_unseen_acc,
                    gzsl_seen_acc,
                    gzsl_unseen_acc_1,
                    gzsl_seen_acc_1))

        try:
            source_sample = train_source_iter.__next__()
        except:
            train_source_iter = iter(train_source_loader)
            source_sample = train_source_iter.__next__()

        try:
            target_sample = train_target_iter.__next__()
        except:
            train_target_iter = iter(train_target_loader)
            target_sample = train_target_iter.__next__()


        #YJA_gai
        optimizer = inv_lr_scheduler_warmup([1, 2], optimizer, i, 0.001, 0.75, init_lr=args.lr)
        optimizer_c = inv_lr_scheduler_warmup([5], optimizer_c, i, 0.001, 0.75, init_lr=args.lr)

        x_s = source_sample['image'].to(device)
        y_s = source_sample['label'].to(device)
        x_t = target_sample['image'].to(device)
        optimizer.zero_grad()
        optimizer_c.zero_grad()
        #loss, source_loss, structure_loss = model(x_s, y_s, seen_label=seen_classes_id, is_train='True')
        #yja+
        loss, source_loss, structure_loss = model(x_s, y_s, seen_label=seen_classes_id, is_train='soft_source_cls_label')
        #target_loss, entropy_loss, dis_loss, bl_loss = model(x_t, seen_label=unseen_classes_id, is_train='soft_source_cls_label_target')
        target_loss, entropy_loss, dis_loss, bl_loss = model(x_t, seen_label=unseen_classes_id, is_train='Target')

        #yja+
        if i < 600:
            all_loss = loss
        else:
            all_loss = target_loss + 1.5 * loss

        all_loss.backward()
        optimizer.step()
        optimizer_c.step()
        tbar.set_description(
            "source_loss: {:.3f}, target: {:.3f}, bl: {:.3f}".format(loss.item(), target_loss.item() - loss.item(),
                                                                     bl_loss))

        #VIS_TEST_YJA
        if i == 2200:
            torch.save(model.Embedding(model.embeddings)[:50], './vis_test/last_class_embedding.pt')
            torch.save(model.Vision_mapnet.state_dict(), './vis_test/Vision_mapnet.pth')
            torch.save(word_vectors[:50], './vis_test/init_class_embedding.pt')

    print(
        "best_zsl_acc: {0[0]} on batch {0[1]}, best_gzsl_unseen_acc: {1[0]} on batch {1[1]}, best_gzsl_seen_acc: {2[0]} on batch {2[1]}, best_gzsl_h: {3[0]} on batch {3[1]}".format(
            best_results["best_zsl_acc"],
            best_results["best_gzsl_unseen_acc"],
            best_results["best_gzsl_seen_acc"],
            best_results["best_gzsl_h"]))

    with open(os.path.join(save_dir, 'result.txt'), 'w') as f:
        f.write(args.source_target_domain[0] + '-' + args.source_target_domain[
            1] + '\n' + 'maxentropy_withoutwarehouse_entropy_{}_maxentropy_{}_{}_consistency_0'.format(
            args.entropy_weight, args.distribute_weight, args.distribute_bound) + '\n')
        f.write(
            "epoch: {}\n zsl_acc: {:.2%}, gzsl_unseen_acc: {:.2%}, gzsl_seen_acc: {:.2%}, gzsl_unseen_class: {:.2%}, gzsl_seen_class: {:.2%}".format(
                i,
                zsl_acc,
                gzsl_unseen_acc,
                gzsl_seen_acc,
                gzsl_unseen_acc_1,
                gzsl_seen_acc_1))
        f.write('\n')
        f.write(
            "best_zsl_acc: {0[0]} on batch {0[1]}, best_gzsl_unseen_acc: {1[0]} on batch {1[1]}, best_gzsl_seen_acc: {2[0]} on batch {2[1]}, best_gzsl_h: {3[0]} on batch {3[1]}".format(
                best_results["best_zsl_acc"],
                best_results["best_gzsl_unseen_acc"],
                best_results["best_gzsl_seen_acc"],
                best_results[
                    "best_gzsl_h"]))


if __name__ == '__main__':
    main()
