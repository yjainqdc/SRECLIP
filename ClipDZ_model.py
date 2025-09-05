import torch
import torch.nn as nn
from mpmath import clsin
from torchaudio import prototype

import network
import torch.nn.functional as F
from torch.autograd import Function
import random

source_loss_func = nn.CrossEntropyLoss()
distribute_loss_func = nn.KLDivLoss(reduction='batchmean')


import clip
import clip_model

device = "cuda" if torch.cuda.is_available() else "cpu"



#信息熵损失
def entropy(x):
    b = x.log()*x
    b = (-1.0 * b.sum()) / x.size(0)
    return b

def hloss(x):
    x1 = F.softmax(x, dim=1)
    x2 = F.log_softmax(x, dim=1)
    b = x1 * x2
    b = (-1.0 * b.sum()) / x1.size(0)
    return b

#计算向量距离
def euclidean_struc_func(x):
    x_1 = x.view(1, x.shape[0], x.shape[1])
    x_2 = x.view(1, x.shape[0], x.shape[1])
    return torch.cdist(x_1, x_2).squeeze()


class Vision_mapnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Vision_mapnet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.fc = nn.Linear(input_size, output_size)
    def forward(self, x):
        x = self.fc1(x)
        #x = torch.relu(x)  # 可以选择激活函数
        x = self.fc2(x)
        #x = self.fc(x)
        return x


class Vision_mapnet_attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Vision_mapnet_attention, self).__init__()
        self.fcq = nn.Linear(input_size, output_size)
        self.fck = nn.Linear(input_size, output_size)
        self.fcv = nn.Linear(input_size, output_size)
        #self.fcmapv = nn.Linear(output_size, output_size)
        #self.weight = nn.Parameter(torch.Tensor(1))
        #self.fc = nn.Linear(input_size, output_size)
    def forward(self, x):
        q = self.fcq(x)
        k = self.fck(x)
        v = self.fcv(x)
        #map = self.fcmapv(v)
        # 计算注意力权重
        attention_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1)).float()), dim=-1)
        output = torch.matmul(attention_weights, v) + v
        return output





class ClipDZ(nn.Module):
    '''
    embeddings应该在创建模型时传入类别向量（GCN前）
    然后每次forword会再次传入新的（GCN后）
    后面还会将GCN挂到模型实体熵
    '''
    def __init__(self, embedding_dim, embeddings, args):
        super(ClipDZ, self).__init__()
        #ClipDZ new,preprocess已经写好在Dataset处理过了
        self.ClipV, preprocess = clip.load("ViT-B/32", device=device)
        #clip后接的，暂时还没用上，后期改

        #self.Vision_mapnet = Vision_mapnet(512,512,1024)
        self.Vision_mapnet = Vision_mapnet_attention(512, 512, 1024)

        # 先关掉clip的梯度
        for param in self.ClipV.parameters():
            param.requires_grad = False


        self.embeddings = embeddings
        self.embeddings.requires_grad = False
        self.softmax = nn.Softmax(dim=1)
        self.entropy_weight = args.entropy_weight
        self.entropy_weight = args.entropy_weight #熵损失的超参数，权重
        self.args = args
        self.temperature = args.temperature #温度
        self.distribute_weight = args.distribute_weight
        self.fc_w = args.fc_w
        self.mse_loss = nn.MSELoss()
        self.distribute_bound = args.distribute_bound
        self.warehouse = None
        self.pointer = 0
        self.max_warehoure = args.max_warehoure
        #self.use_warehouse = args.use_warehouse
        self.structure_type = args.consistency_type
        self.struc_weight = args.consistency_weight
        self.L2_LOSS = torch.nn.MSELoss()
        structure_loss = {'l2': self.l2_loss,
                          'entropy': self.cross_entropy_rank,
                          'pair': self.pairwise_rank_loss,
                          'consistency': self.consistency_loss}
        self.structure_func = euclidean_struc_func
        self.structure_loss = structure_loss[self.structure_type]
        with torch.no_grad():
            if self.args.dataset == 'I2WebV':
                self.struc = self.structure_func(self.embeddings[:5000])
                self.class_num = 100
            elif self.args.dataset == 'I2AwA':
                #self.struc = self.structure_func(self.embeddings[127:177]) #原始的embedding的距离
                self.struc = self.structure_func(self.embeddings[:50])
                #这里很奇怪，做个试验试试 2024 10 11 23：43
                self.class_num = 50
        self.rank = [i for i in range(self.class_num)]

        if self.args.dataset == 'I2WebV':
            class_num = self.class_num
        else:
            class_num = self.class_num
        self.pair1 = torch.zeros(class_num, int(class_num * (class_num + 1) / 2)).to(args.device)
        self.pair2 = torch.zeros(class_num, int(class_num * (class_num + 1) / 2)).to(args.device)
        k = 0

        for i in range(class_num):
            for j in range(class_num - i):
                self.pair1[i, k] = 1
                self.pair2[j, k] = 1
                k += 1
        assert k == int(class_num * (class_num + 1) / 2)




    def bl(self, unseen, c):
        return torch.mean(unseen) + (c**2)/torch.mean(unseen)

    def l2_loss(self, embedding, prototype):
        embedding = embedding / torch.mean(embedding)
        prototype = prototype / torch.mean(prototype)
        return torch.mean((embedding - prototype) ** 2)

    #直接return0啊？
    def consistency_loss(self, embedding, prototype):
        return torch.tensor(0, dtype='torch.float32', device=self.args.device)

    def cross_entropy_rank(self, embedding, prototype):
        embedding = embedding / torch.mean(embedding)
        embedding = (embedding.max().clone().detach() - embedding)
        prototype = prototype / torch.mean(prototype)
        prototype = (prototype.max().clone().detach() - prototype)
        p_embedding = F.log_softmax(prototype, dim=1)
        e_embedding = F.softmax(embedding, dim=1)
        loss_func = nn.KLDivLoss(reduction='sum')
        return loss_func(p_embedding, e_embedding)

    def pairwise_rank_loss(self, embedding, prototype):
        num_rank = len(self.rank) #5000 获取self.rank列表的长度，假设为num_rank
        sample = random.sample(self.rank, num_rank) # 从self.rank列表中随机选择num_rank个元素，存储在sample中
        embedding = embedding[sample, sample] # 从embedding中获取与sample对应的子集数据
        prototype = prototype[sample, sample] # 从prototype中获取与sample对应的子集数据
        embedding1 = torch.matmul(embedding, self.pair1)
        embedding2 = torch.matmul(embedding, self.pair2)
        embedding_dis = embedding1 - embedding2
        prototype1 = torch.matmul(prototype, self.pair1)
        prototype2 = torch.matmul(prototype, self.pair2)
        prototype_dis = prototype1 - prototype2
        loss = torch.mean(torch.relu(prototype_dis[embedding_dis < 0]))
        return loss

    #YJA+
    def img2cls_loss(self, embeddings,p):
        # 找到每个输入图像对应的最相关的嵌入的索引
        max_indices = torch.argmax(p, dim=1)
        # 从 embeddings 中提取相应的嵌入
        relevant_embeddings = torch.index_select(embeddings, dim=0, index=max_indices)
        target_img2cls = torch.mm(relevant_embeddings, embeddings.T)
        mse_loss = nn.MSELoss(reduction='mean')
        loss = mse_loss(p,target_img2cls)
        return loss


    def forward(self, x, y=None, seen_label=None, is_train='True'):
        #x是图像，y是label
        x = self.ClipV.encode_image(x) #bs*512
        x = x.float()
        x = self.Vision_mapnet(x) #bs*2048
        x = F.normalize(x, p=2.0, dim=1) / self.temperature
        #########################################待定
        if self.args.dataset == 'I2WebV':
            embeddings = self.Embedding(self.embeddings)[:5000]
        elif self.args.dataset == 'I2AwA':
            embeddings = self.Embedding(self.embeddings)[:50] #输入的图后面改
        #########################################待定
        embeddings = F.normalize(embeddings, p=2.0, dim=1)  #图卷积后的textemb，应该也是2048才对

        #计算loss
        if is_train == 'True':
            p = torch.mm(x, embeddings.T) #计算图片和类别向量的相似度 256*50
            struc_e = self.structure_func(embeddings) #计算优化后类别向量的欧氏距离的函数 embedding:50*2048
            struc_loss = self.structure_loss(self.struc, struc_e) #self.struc是优化前的类别距离用来教育优化后的类别距离50*50
            source_loss = source_loss_func(p, y) #就是计算相似度后作交叉熵
            loss = source_loss + self.struc_weight*struc_loss #zongloss：struc_loss是10-5量级，权重10太小了
            return loss, source_loss.item(), struc_loss.item()

        #YJA+
        elif is_train == 'soft_source_cls_label':
            p = torch.mm(x, embeddings.T) #计算图片和类别向量的相似度 256*50 【256*5000】
            struc_e = self.structure_func(embeddings) #计算优化后类别向量的欧氏距离的函数 embedding:50*1024 【5000 * 1024】
            struc_loss = self.structure_loss(self.struc, struc_e) #self.struc是优化前的类别距离用来教育优化后的类别距离50*50,GCNloss 【5000 * 5000】
            source_loss = source_loss_func(p, y) #就是计算相似度后作交叉熵
            img2cls_loss = self.img2cls_loss(embeddings, p)
            loss = source_loss + self.struc_weight*struc_loss + 0.01 * img2cls_loss  #大数据
            #loss = source_loss + self.struc_weight * struc_loss + 0.1 * img2cls_loss  #小数据

            return loss, source_loss.item(), struc_loss.item()


        elif is_train == 'Validation':
            p = torch.mm(x, embeddings.T)
            return self.softmax(p)

        elif is_train == 'Target':
            p = torch.mm(x, embeddings.clone().detach().T)  #训练目标域不用优化GCN
            distribute_loss = torch.tensor(0)
            distribute_loss1 = torch.tensor(0)
            if self.warehouse is None:
                self.warehouse = x.detach().clone()  # 复制了一份x，训练数据的特征，这是memory bank
            elif self.warehouse.shape[0] < self.max_warehoure:
                self.warehouse = torch.cat((self.warehouse, x.detach().clone()), dim=0)
            else:
                self.warehouse[self.pointer:self.pointer + x.shape[0], :] = x.detach().clone()
                self.pointer += x.shape[0]
                if self.pointer + x.shape[0] > self.max_warehoure:
                    self.pointer = 0
                p_2 = torch.mm(self.warehouse, embeddings.T)  # 所有的memory bank的视觉特征都和类别原型计算一下
                p_1 = self.softmax(p_2)
                p_1 = torch.mean(p_1, dim=0, keepdim=True)
                distribute_loss = -1 * min(entropy(p_1), self.distribute_bound)  # 那个平均k个batch的loss
                p_2 = torch.mm(x, embeddings.T)

                p_3 = torch.sum(self.softmax(p_2)[:, seen_label], dim=1)
                if self.args.dataset == 'I2WebV':
                    distribute_loss1 = self.bl(p_3, 0.82)
                elif self.args.dataset == 'I2AwA':
                    distribute_loss1 = self.bl(p_3, self.args.bias_weight)
            return self.entropy_weight * hloss(
                p) + self.distribute_weight * distribute_loss + self.args.fc_w * distribute_loss1, hloss(
                p).item(), distribute_loss.item(), distribute_loss1.item()
        #YJA+
        elif is_train == 'soft_source_cls_label_target':

            p = torch.mm(x, embeddings.clone().detach().T)  # 训练目标域不用优化GCN
            distribute_loss = torch.tensor(0)
            distribute_loss1 = torch.tensor(0)
            if self.warehouse is None:
                self.warehouse = x.detach().clone()  # 复制了一份x，训练数据的特征，这是memory bank
            elif self.warehouse.shape[0] < self.max_warehoure:
                self.warehouse = torch.cat((self.warehouse, x.detach().clone()), dim=0)
            else:
                self.warehouse[self.pointer:self.pointer + x.shape[0], :] = x.detach().clone()
                self.pointer += x.shape[0]
                if self.pointer + x.shape[0] > self.max_warehoure:
                    self.pointer = 0
                p_2 = torch.mm(self.warehouse, embeddings.T)  # 所有的memory bank的视觉特征都和类别原型计算一下
                p_1 = self.softmax(p_2)
                p_1 = torch.mean(p_1, dim=0, keepdim=True)
                distribute_loss = -1 * min(entropy(p_1), self.distribute_bound)  # 那个平均k个batch的loss
                p_2 = torch.mm(x, embeddings.T)

                p_3 = torch.sum(self.softmax(p_2)[:, seen_label], dim=1)
                if self.args.dataset == 'I2WebV':
                    distribute_loss1 = self.bl(p_3, 0.82)
                elif self.args.dataset == 'I2AwA':
                    distribute_loss1 = self.bl(p_3, self.args.bias_weight)

            #YJA+
            img2cls_loss = 0.1 * self.img2cls_loss(embeddings, p)

            return (self.entropy_weight * hloss(p) + self.distribute_weight * distribute_loss + self.args.fc_w * distribute_loss1 + img2cls_loss,
                    hloss(p).item(),
                    distribute_loss.item(),
                    distribute_loss1.item())




            p = torch.mm(x, embeddings.T) #计算图片和类别向量的相似度 256*50 【256*5000】
            struc_e = self.structure_func(embeddings) #计算优化后类别向量的欧氏距离的函数 embedding:50*1024 【5000 * 1024】
            struc_loss = self.structure_loss(self.struc, struc_e) #self.struc是优化前的类别距离用来教育优化后的类别距离50*50,GCNloss 【5000 * 5000】
            source_loss = source_loss_func(p, y) #就是计算相似度后作交叉熵
            img2cls_loss = self.img2cls_loss(embeddings, p)
            #loss = source_loss + self.struc_weight*struc_loss + 0.01 * img2cls_loss  #大数据
            loss = source_loss + self.struc_weight * struc_loss + 0.1 * img2cls_loss  #小数据

            return loss, source_loss.item(), struc_loss.item()






class Vision_mapnet_CNNAlpha(nn.Module):
    def __init__(self, in_channels, hid_channels, key_channels,  T=0.2):
        super(Vision_mapnet_CNNAlpha, self).__init__()
        self.fc = nn.Linear(in_channels, key_channels)
        self.T = T
        self.query = nn.Conv1d(key_channels, key_channels//2, kernel_size=1, padding=0, dilation=1, bias=False)
        self.cross_attention = nn.Conv1d(key_channels//2, 16, kernel_size=1, padding=0,
                                         dilation=1, bias=False)

    def forward(self, x):
        maped_x = self.fc(x) # B,50,2048

        sorted_x = maped_x.sort(dim=1, descending=True)[0]
        query = self.query(sorted_x.permute(0, 2, 1))  # (B, 512, L) -> (B, 256, L)
        cross_attention = self.cross_attention(query)  # (B, 256, L) -> (B, 16, L)
        cross_attention = F.sigmoid((cross_attention))
        B, _, L = cross_attention.size()# _ = 192
        #cross_attention = cross_attention.reshape(B, -1, L)  # (B, 16, L)
        cross_attention = torch.mean(cross_attention, dim=1, keepdim=False) #(B,16, L) -> (B, L)
        cross_attention = F.softmax(cross_attention / self.T, dim=-1)  #(B, L)
        #pooled_x = torch.matmul(cross_attention, sorted_x)  # (B, L), (B, L, C) -> (B, C)
        #pooled_x = torch.sum(cross_attention * sorted_x,dim=1, keepdim=False)
        pooled_x = sorted_x * cross_attention.unsqueeze(-1)
        pooled_x = pooled_x.sum(dim=1, keepdim=False)
        return pooled_x  ##(B, C), (B, L)




class ClipDZ_patch(nn.Module):
    '''
    embeddings应该在创建模型时传入类别向量（GCN前）
    然后每次forword会再次传入新的（GCN后）
    后面还会将GCN挂到模型实体熵
    '''
    def __init__(self, embedding_dim, embeddings, args):
        super(ClipDZ_patch, self).__init__()
        #ClipDZ new,preprocess已经写好在Dataset处理过了
        self.ClipV, preprocess = clip_model.load("ViT-B/32", device=device)

        self.Vision_mapnet = Vision_mapnet_CNNAlpha(512,512,1024)

        #先关掉clip的梯度
        # for param in self.visual.ClipV:
        #     param.requires_grad = True


        self.embeddings = embeddings
        self.embeddings.requires_grad = False
        self.softmax = nn.Softmax(dim=1)
        self.entropy_weight = args.entropy_weight
        self.entropy_weight = args.entropy_weight #熵损失的超参数，权重
        self.args = args
        self.temperature = args.temperature #温度
        self.distribute_weight = args.distribute_weight
        self.fc_w = args.fc_w
        self.mse_loss = nn.MSELoss()
        self.distribute_bound = args.distribute_bound
        self.warehouse = None
        self.pointer = 0
        self.max_warehoure = args.max_warehoure
        #self.use_warehouse = args.use_warehouse
        self.structure_type = args.consistency_type
        self.struc_weight = args.consistency_weight
        self.L2_LOSS = torch.nn.MSELoss()
        structure_loss = {'l2': self.l2_loss,
                          'entropy': self.cross_entropy_rank,
                          'pair': self.pairwise_rank_loss,
                          'consistency': self.consistency_loss}
        self.structure_func = euclidean_struc_func
        self.structure_loss = structure_loss[self.structure_type]
        with torch.no_grad():
            if self.args.dataset == 'I2WebV':
                self.struc = self.structure_func(self.embeddings[:5000])
                self.class_num = 5000
            elif self.args.dataset == 'I2AwA':
                self.struc = self.structure_func(self.embeddings[127:177]) #原始的embedding的距离
                self.class_num = 50
        self.rank = [i for i in range(self.class_num)]

        if self.args.dataset == 'I2WebV':
            class_num = 100
        else:
            class_num = self.class_num
        self.pair1 = torch.zeros(class_num, int(class_num * (class_num + 1) / 2)).to(args.device)
        self.pair2 = torch.zeros(class_num, int(class_num * (class_num + 1) / 2)).to(args.device)
        k = 0

        for i in range(class_num):
            for j in range(class_num - i):
                self.pair1[i, k] = 1
                self.pair2[j, k] = 1
                k += 1
        assert k == int(class_num * (class_num + 1) / 2)


    def bl(self, unseen, c):
        return torch.mean(unseen) + (c**2)/torch.mean(unseen)

    def l2_loss(self, embedding, prototype):
        embedding = embedding / torch.mean(embedding)
        prototype = prototype / torch.mean(prototype)
        return torch.mean((embedding - prototype) ** 2)

    #直接return0啊？
    def consistency_loss(self, embedding, prototype):
        return torch.tensor(0, dtype='torch.float32', device=self.args.device)

    def cross_entropy_rank(self, embedding, prototype):
        embedding = embedding / torch.mean(embedding)
        embedding = (embedding.max().clone().detach() - embedding)
        prototype = prototype / torch.mean(prototype)
        prototype = (prototype.max().clone().detach() - prototype)
        p_embedding = F.log_softmax(prototype, dim=1)
        e_embedding = F.softmax(embedding, dim=1)
        loss_func = nn.KLDivLoss(reduction='sum')
        return loss_func(p_embedding, e_embedding)

    def pairwise_rank_loss(self, embedding, prototype):
        num_rank = len(self.rank)
        sample = random.sample(self.rank, num_rank)
        embedding = embedding[sample, sample]
        prototype = prototype[sample, sample]
        embedding1 = torch.matmul(embedding, self.pair1)
        embedding2 = torch.matmul(embedding, self.pair2)
        embedding_dis = embedding1 - embedding2
        prototype1 = torch.matmul(prototype, self.pair1)
        prototype2 = torch.matmul(prototype, self.pair2)
        prototype_dis = prototype1 - prototype2
        loss = torch.mean(torch.relu(prototype_dis[embedding_dis < 0]))
        return loss

    def forward(self, x, y=None, seen_label=None, is_train='True'):
        #x是图像，y是label
        x = self.ClipV.encode_image_patch(x) #bs*512
        x = x.float()
        x = self.Vision_mapnet(x) #bs*2048
        x = F.normalize(x, p=2.0, dim=1) / self.temperature
        #########################################待定
        if self.args.dataset == 'I2WebV':
            embeddings = self.Embedding(self.embeddings)[:5000]
        elif self.args.dataset == 'I2AwA':
            embeddings = self.Embedding(self.embeddings)[:50] #输入的图后面改
        #########################################待定
        embeddings = F.normalize(embeddings, p=2.0, dim=1)  #图卷积后的textemb，应该也是2048才对

        #计算loss
        if is_train == 'True':
            p = torch.mm(x, embeddings.T) #计算图片和类别向量的相似度 256*50
            struc_e = self.structure_func(embeddings) #计算优化后类别向量的欧氏距离的函数 embedding:50*2048
            struc_loss = self.structure_loss(self.struc, struc_e) #self.struc是优化前的类别距离用来教育优化后的类别距离50*50,
            source_loss = source_loss_func(p, y) #就是计算相似度后作交叉熵
            loss = source_loss + self.struc_weight*struc_loss #zongloss：struc_loss是10-5量级，权重10太小了
            return loss, source_loss.item(), struc_loss.item()

        elif is_train == 'Validation':
            p = torch.mm(x, embeddings.T)
            return self.softmax(p)

        elif is_train == 'Target':
            p = torch.mm(x, embeddings.clone().detach().T)  #训练目标域不用优化GCN
            distribute_loss = torch.tensor(0)
            distribute_loss1 = torch.tensor(0)
            if self.warehouse is None:
                self.warehouse = x.detach().clone()  # 复制了一份x，训练数据的特征，这是memory bank
            elif self.warehouse.shape[0] < self.max_warehoure:
                self.warehouse = torch.cat((self.warehouse, x.detach().clone()), dim=0)
            else:
                self.warehouse[self.pointer:self.pointer + x.shape[0], :] = x.detach().clone()
                self.pointer += x.shape[0]
                if self.pointer + x.shape[0] > self.max_warehoure:
                    self.pointer = 0
                p_2 = torch.mm(self.warehouse, embeddings.T)  # 所有的memory bank的视觉特征都和类别原型计算一下
                p_1 = self.softmax(p_2)
                p_1 = torch.mean(p_1, dim=0, keepdim=True)
                distribute_loss = -1 * min(entropy(p_1), self.distribute_bound)  # 那个平均k个batch的loss
                p_2 = torch.mm(x, embeddings.T)

                p_3 = torch.sum(self.softmax(p_2)[:, seen_label], dim=1)
                if self.args.dataset == 'I2WebV':
                    distribute_loss1 = self.bl(p_3, 0.82)
                elif self.args.dataset == 'I2AwA':
                    distribute_loss1 = self.bl(p_3, self.args.bias_weight)
            return self.entropy_weight * hloss(
                p) + self.distribute_weight * distribute_loss + self.args.fc_w * distribute_loss1, hloss(
                p).item(), distribute_loss.item(), distribute_loss1.item()

