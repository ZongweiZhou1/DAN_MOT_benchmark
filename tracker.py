from layer.sst import build_sst
from config.config import config
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class TrackUtil:
    @staticmethod
    def convert_detection(detection):
        '''
        transform the current detection center to [-1, 1]
        :param detection: detection
        :return: translated detection
        '''
        # get the center, and format it in (-1, 1)
        center = (2 * detection[:, 0:2] + detection[:, 2:4]) - 1.0  # 2（xy+wh/2) - 1.0
        center = torch.from_numpy(center.astype(float)).float()  # numpy-> floattensor
        center.unsqueeze_(0)
        center.unsqueeze_(2)
        center.unsqueeze_(3)

        
        if TrackerConfig.cuda:
            return Variable(center.cuda())
        return Variable(center)

    @staticmethod
    def convert_image(image):
        ''' 将图像转成1x3xHxW的tensor
        transform image to the FloatTensor (1, 3,size, size)
        :param image: same as update parameter
        :return: the transformed image FloatTensor (i.e. 1x3x900x900)
        '''
        image = cv2.resize(image, TrackerConfig.image_size).astype(np.float32)
        image -= TrackerConfig.mean_pixel
        image = torch.FloatTensor(image)
        image = image.permute(2, 0, 1)
        image.unsqueeze_(dim=0)
        if TrackerConfig.cuda:
            return Variable(image.cuda())
        return Variable(image)

    @staticmethod
    def get_iou(pre_boxes, next_boxes):
        # 姜蒜两个box数组的iou， 都是 tlwh格式
        h = len(pre_boxes)  # 预测box个数
        w = len(next_boxes)  # 检测的box个数
        if h == 0 or w == 0:  # 有一个为空则返回空
            return []

        iou = np.zeros((h, w), dtype=float)
        for i in range(h):
            b1 = np.copy(pre_boxes[i, :])
            b1[2:] = b1[:2] + b1[2:]  # 转成 tlbr
            for j in range(w):
                b2 = np.copy(next_boxes[j, :])
                b2[2:] = b2[:2] + b2[2:]  # z转成tlbr
                delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
                delta_w = min(b1[3], b2[3])-max(b1[1], b2[1])
                if delta_h < 0 or delta_w < 0:  
                    # 标准的iou，这种情形表示不重叠，应该置为0， 这里iou允许为负数
                    # iou计算为 两个框的最小外接矩形中不属于两个box的面积与两个boxes面积的比值，负数
                    expand_area = (max(b1[2], b2[2]) - min(b1[0], b2[0])) * (max(b1[3], b2[3]) - min(b1[1], b2[1]))  # 外接矩形面积
                    area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1])  # 面积和
                    iou[i,j] = -(expand_area - area) / area
                else:  # 这个是标准的iou计算
                    overlap = delta_h * delta_w
                    area = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - max(overlap, 0)
                    iou[i,j] = overlap / area

        return iou

    @staticmethod
    def get_node_similarity(n1, n2, frame_index, recorder):
        # n1, n2      两个node类型
        # frame_index   frame的索引
        if n1.frame_index > n2.frame_index:
            n_max = n1
            n_min = n2
        elif n1.frame_index < n2.frame_index:
            n_max = n2
            n_min = n1
        else: # in the same frame_index
            return None
        
        f_max = n_max.frame_index
        f_min = n_min.frame_index
        # n_max 表示较新的node， n_min表示较早的node
        # f_max表示较新的node的frame index， f_min表示较早的node的frame index
        
        # not recorded in recorder
        if frame_index - f_min >= TrackerConfig.max_track_node:
            return None

        return recorder.all_similarity[f_max][f_min][n_min.id, n_max.id]  # 

    @staticmethod
    def get_merge_similarity(t1, t2, frame_index, recorder):
        '''
        Get the similarity between two tracks
        :param t1: track 1
        :param t2: track 2
        :param frame_index: current frame_index
        :param recorder: recorder
        :return: the similairty (float value). if valid, return None
        两条估计合并的概率
        t1 ___________ _ __ _______   ________   __
        t2 ______________  ___________ _______ ____
                      + +  +       + +        +  +
        计算的是两条轨迹对应 + 号表示的frame时节点的相似度
        '''
        merge_value = []
        if t1 is t2:
            return None

        all_f1 = [n.frame_index for n in t1.nodes] # 轨迹1所经过的帧
        all_f2 = [n.frame_index for n in t2.nodes] # 轨迹2所经过的帧

        for i, f1 in enumerate(all_f1):  
            for j, f2 in enumerate(all_f2):
                compare_f = [f1 + 1, f1 - 1]  
                for f in compare_f:
                    if f not in all_f1 and f == f2:
                        n1 = t1.nodes[i]
                        n2 = t2.nodes[j]
                        s = TrackUtil.get_node_similarity(n1, n2, frame_index, recorder)
                        if s is None:
                            continue
                        merge_value += [s]

        if len(merge_value) == 0:
            return None
        return np.mean(np.array(merge_value))

    @staticmethod
    def merge(t1, t2):
        '''
        merge t2 to t1, after that t2 is set invalid
        :param t1: track 1
        :param t2: track 2
        :return: None
        合并轨迹，将轨迹2上特有的点merge到轨迹1上
        '''
        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f2 in enumerate(all_f2):  # 轨迹2对应的帧
            if f2 not in all_f1:  # 轨迹1在当前帧没出现
                insert_pos = 0
                for j, f1 in enumerate(all_f1):
                    if f2 < f1:  #找到当前帧之前轨迹1的所有node，即待插入的位置
                        break
                    insert_pos += 1
                t1.nodes.insert(insert_pos, t2.nodes[i])  # 将轨迹2在当前帧的node插入轨迹1

        # remove some nodes in t1 in order to keep t1 satisfy the max nodes
        if len(t1.nodes) > TrackerConfig.max_track_node:
            t1.nodes = t1.nodes[-TrackerConfig.max_track_node:]  # 轨迹保存的node数有个上限，即考虑的历史帧数
        t1.age = min(t1.age, t2.age)  # 距最新更新时刻的时间间隔
        t2.valid = False  # 轨迹2无效

class TrackerConfig:
    max_record_frame = 30     # 轨迹保存的最大历史帧数
    max_track_age = 30        # 允许最大的跟丢帧数 
    max_track_node = 30       # 轨迹保存的最大node数
    max_draw_track_node = 30  # 绘制轨迹鬼影最大数目

    max_object = config['max_object']  # 每帧最多可能的目标书
    sst_model_path = config['resume']  # 跟踪器的预训练权重
    cuda = config['cuda']              # 
    mean_pixel = config['mean_pixel']  # mean pixel，用于图像中心化
    image_size = (config['sst_dim'], config['sst_dim'])  # resize之后的图像大小，模型的输入大小
    
    # 下面两行定义的间隔一定帧对应的最小iou阈值
    min_iou_frame_gap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_iou = [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0]
    # min_iou = [pow(0.3, i) for i in min_iou_frame_gap]

    min_merge_threshold = 0.9  # 合并阈值

    max_bad_node = 0.9  # 轨迹允许的最大坏点数

    decay = 0.995   # 衰减率

    roi_verify_max_iteration = 2  # 验证roi的最大迭代数
    roi_verify_punish_rate = 0.6 # 验证roi的惩罚率

    @staticmethod
    def set_configure(all_choice):
        # 针对于不同的数据集设定不同的参数

        min_iou_frame_gaps = [
            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            ]
        min_ious = [
            # [0.4, 0.3, 0.25, 0.2, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0],
            [0.3, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
            [0.3, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [0.2, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -7.0],
            [-1.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            [0.4, 0.3, 0.25, 0.2, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0],
        ]

        decays = [1-0.01*i for i in range(11)]

        roi_verify_max_iterations = [2, 3, 4, 5, 6]

        roi_verify_punish_rates = [0.6, 0.4, 0.2, 0.1, 0.0, 1.0]

        max_track_ages = [i*3 for i in range(1,11)]
        max_track_nodes = [i*3 for i in range(1,11)]

        if all_choice is None:
            return
        TrackerConfig.min_iou_frame_gap = min_iou_frame_gaps[all_choice[0]]
        TrackerConfig.min_iou = min_ious[all_choice[0]]
        TrackerConfig.decay = decays[all_choice[1]]
        TrackerConfig.roi_verify_max_iteration = roi_verify_max_iterations[all_choice[2]]
        TrackerConfig.roi_verify_punish_rate = roi_verify_punish_rates[all_choice[3]]
        TrackerConfig.max_track_age = max_track_ages[all_choice[4]]
        TrackerConfig.max_track_node = max_track_nodes[all_choice[5]]

    @staticmethod
    def get_configure_str(all_choice):
        # 返回选择的超参数对应的索引组成的字符串
        return "{}_{}_{}_{}_{}_{}".format(all_choice[0], all_choice[1], all_choice[2], all_choice[3], all_choice[4], all_choice[5])

    @staticmethod
    def get_all_choices():
        # 遍历所有可能的组合，可用于网格搜索
        # return [(1, 1, 0, 0, 4, 2)]
        return [(i1, i2, i3, i4, i5, i6) for i1 in range(5) for i2 in range(5) for i3 in range(5) for i4 in range(5) for i5 in range(5) for i6 in range(5)]

    @staticmethod
    def get_all_choices_decay():
        # 遍历所有的decay的可能
        return [(1, i2, 0, 0, 4, 2) for i2 in range(11)]

    @staticmethod
    def get_all_choices_max_track_node():
        # 这个估计存在问题？？？
        return [(1, i2, 0, 0, 4, 2) for i2 in range(11)]

    @staticmethod
    def get_choices_age_node():
        # 遍历所有的age 和 node组合
        return [(0, 0, 0, 0, a, n) for a in range(10) for n in range(10)]

    @staticmethod
    def get_ua_choice():
        # ua数据集的选择
        return (5, 0, 4, 1, 5, 5)
class FeatureRecorder:
    '''
    Record features and boxes every frame
    '''

    def __init__(self):
        self.max_record_frame = TrackerConfig.max_record_frame  # 最多纪录的帧数
        self.all_frame_index = np.array([], dtype=int)  # frame_index
        self.all_features = {}  # 每一条轨迹每一帧对应的feature
        self.all_boxes = {}     # 每一条轨迹每一帧对应的box
        self.all_similarity = {}
        self.all_iou = {}

    def update(self, sst, frame_index, features, boxes):
        # if the coming frame in the new frame
        if frame_index not in self.all_frame_index:  # 新来的帧
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:  
                # 若record的记录数大于阈值则删除最早的纪录
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                del self.all_iou[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes

            self.all_similarity[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                delta = pow(TrackerConfig.decay, (frame_index - pre_index)/3.0)  # 由时间间隔带来的衰减
                pre_similarity = sst.forward_stacker_features(Variable(self.all_features[pre_index]), Variable(features), fill_up_column=False)  # 对阵帧中目标与当前帧通过SST计算特征相似度
                self.all_similarity[frame_index][pre_index] = pre_similarity*delta  # 每一帧都对应着与之前recorder中的任何一帧的相似度

            self.all_iou[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:  # 这个似乎可以和上一个循环放到一起， 计算当前帧与历史帧中目标之间的iou
                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_iou[frame_index][pre_index] = iou
        else:  # 已经存在的帧
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            index = self.all_frame_index.__index__(frame_index)

            for pre_index in self.all_frame_index[:index+1]:  # 更新 frame_index之前的历史记录与最近的帧中目标的相似度，包括frame_index
                if pre_index == self.all_frame_index[-1]:
                    continue

                pre_similarity = sst.forward_stacker_features(Variable(self.all_features[pre_index]), Variable(self.all_features[-1]))
                self.all_similarity[frame_index][pre_index] = pre_similarity

                iou = TrackUtil.get_iou(self.all_boxes[pre_index], boxes)
                self.all_similarity[frame_index][pre_index] = iou

    def get_feature(self, frame_index, detection_index):
        ''' 指定帧指定目标的feature
        get the feature by the specified frame index and detection index
        :param frame_index: start from 0
        :param detection_index: start from 0
        :return: the corresponding feature at frame index and detection index
        '''

        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
            if len(features) == 0:
                return None
            if detection_index < len(features):
                return features[detection_index]

        return None

    def get_box(self, frame_index, detection_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
            if len(boxes) == 0:
                return None

            if detection_index < len(boxes):
                return boxes[detection_index]
        return None

    def get_features(self, frame_index):
        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
        else:
            return None
        if len(features) == 0:
            return None
        return features

    def get_boxes(self, frame_index):
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
        else:
            return None

        if len(boxes) == 0:
            return None
        return boxes

class Node:
    '''
    The Node is the basic element of a track. it contains the following information:
    1) extracted feature (it'll get removed when it isn't active
    2) box (a box (l, t, r, b)
    3) label (active label indicating keeping the features)
    4) detection, the formated box
    节点类
    '''
    def __init__(self, frame_index, id):
        self.frame_index = frame_index  # 节点所在的帧
        self.id = id  # 节点对应的索引id

    def get_box(self, frame_index, recoder):
        if frame_index - self.frame_index >= TrackerConfig.max_record_frame:
            # 判断要查询的node是否还在recorder中
            return None
        return recoder.all_boxes[self.frame_index][self.id, :]  # 返回node的box

    def get_iou(self, frame_index, recoder, box_id):
        if frame_index - self.frame_index >= TrackerConfig.max_track_node:
            return None
        return recoder.all_iou[frame_index][self.frame_index][self.id, box_id]  # 返回node与当前帧中box_id的iou

class Track:
    '''
    Track is the class of track. it contains all the node and manages the node. it contains the following information:
    1) all the nodes  所有属于该track的node
    2) track id. it is unique it identify each track  当前track的id
    3) track pool id. it is a number to give a new id to a new track  track池中的trackid
    4) age. age indicates how old is the track  轨迹连续跟丢的时间
    5) max_age. indicates the dead age of this track  轨迹丢失的时间
    '''
    _id_pool = 0

    def __init__(self):
        # 创建一条轨迹
        self.nodes = list()  # 存储轨迹的node
        self.id = Track._id_pool  # 全局变量，为后续轨迹分配标号
        Track._id_pool += 1
        self.age = 0
        self.valid = True   # indicate this track is merged
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())  # 每条轨迹分配一个颜色用于显示

    def __del__(self):
        # 删除某条轨迹，则将轨迹中的节点全部删除
        for n in self.nodes:
            del n

    def add_age(self):
        # 轨迹age增长
        self.age += 1

    def reset_age(self):
        # 轨迹的age重置
        self.age = 0

    def add_node(self, frame_index, recorder, node):
        # iou judge
        if len(self.nodes) > 0:
            n = self.nodes[-1]
            iou = n.get_iou(frame_index, recorder, node.id)  # 计算最后一个轨迹与当前要添加的轨迹的iou，如果符合条件才添加，否则不添加
            delta_frame = frame_index - n.frame_index 
            if delta_frame in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_frame)
                # if iou < TrackerConfig.min_iou[iou_index]:
                if iou < TrackerConfig.min_iou[-1]:  # 这里只选择了最大的容忍值
                    return False
        self.nodes.append(node)
        self.reset_age()  # 重置age，表示跟踪到
        return True

    def get_similarity(self, frame_index, recorder):
        similarity = []
        for n in self.nodes:
            f = n.frame_index
            id = n.id
            if frame_index - f >= TrackerConfig.max_track_node:
                continue
            similarity += [recorder.all_similarity[frame_index][f][id, :]]
            #  添加的是轨迹历史的node 与当前帧中所有node的相似度

        if len(similarity) == 0:
            return None
        a = np.array(similarity)
        return np.sum(np.array(similarity), axis=0)  # 所有的相似度求和，表示轨迹与frame_index中每一个节点的相似度

    def verify(self, frame_index, recorder, box_id):
        # 验证轨迹的node是否与frame_index的box_id的ious都在阈值范围内
        for n in self.nodes:
            delta_f = frame_index - n.frame_index
            if delta_f in TrackerConfig.min_iou_frame_gap:
                iou_index = TrackerConfig.min_iou_frame_gap.index(delta_f)
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou is None:
                    continue
                if iou < TrackerConfig.min_iou[iou_index]:
                    return False
        return True

class Tracks:
    '''
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous image and features
    '''
    def __init__(self):
        self.tracks = list() # the set of tracks
        self.max_drawing_track = TrackerConfig.max_draw_track_node  # 每条轨迹绘制的最大的节点数


    def __getitem__(self, item):
        # 取第item条轨迹
        return self.tracks[item]

    def append(self, track):
        # 添加轨迹
        self.tracks.append(track)
        self.volatile_tracks()  # 删除部分轨迹

    def volatile_tracks(self):
        # 如果轨迹数多余最大轨迹数目，则需要删除最久没跟踪到的那条
        if len(self.tracks) > TrackerConfig.max_object:  
            # start to delete the most oldest tracks
            all_ages = [t.age for t in self.tracks]
            oldest_track_index = np.argmax(all_ages)
            del self.tracks[oldest_track_index]

    def get_track_by_id(self, id):
        # 通过id访问轨迹
        for t in self.tracks:
            if t.id == id:
                return t
        return None

    def get_similarity(self, frame_index, recorder):
        # 
        ids = []
        similarity = []
        for t in self.tracks:
            s = t.get_similarity(frame_index, recorder)  # 当前轨迹与frame_idx中所有node的相似度
            if s is None:
                continue
            similarity += [s]
            ids += [t.id]  # 对应轨迹id

        similarity = np.array(similarity)  # 每一行对应一条轨迹的相似度

        track_num = similarity.shape[0]  # 轨迹数目
        if track_num > 0:
            box_num = similarity.shape[1]  # frame_index中node的个数
        else:
            box_num = 0

        if track_num == 0 :
            return np.array(similarity), np.array(ids)
        # 轨迹数目不为0时
        similarity = np.repeat(similarity, [1]*(box_num-1)+[track_num], axis=1)  # 最后一个box的repeat  track_num次
        return np.array(similarity), np.array(ids)

    def one_frame_pass(self):
        # 过滤丢失时间超出阈值的轨迹
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.add_age()
            if t.age > TrackerConfig.max_track_age:
                continue
            keep_track_set.append(i)

        self.tracks = [self.tracks[i] for i in keep_track_set]

    def merge(self, frame_index, recorder):
        t_l = len(self.tracks) # 轨迹个数
        res = np.zeros((t_l, t_l), dtype=float)  # 轨迹与轨迹之间的相似度
        # get track similarity matrix
        for i, t1 in enumerate(self.tracks):
            for j, t2 in enumerate(self.tracks):
                s = TrackUtil.get_merge_similarity(t1, t2, frame_index, recorder)  # 轨迹合并的相似度
                if s is None:
                    res[i, j] = 0
                else:
                    res[i, j] = s

        # get the track pair which needs merged
        used_indexes = []
        merge_pair = []
        for i, t1 in enumerate(self.tracks):
            if i in used_indexes:
                continue
            max_track_index = np.argmax(res[i, :])  # 第i条轨迹最相似的轨迹
            if i != max_track_index and res[i, max_track_index] > TrackerConfig.min_merge_threshold:  # 相似度大于某一阈值则合并
                used_indexes += [max_track_index]
                merge_pair += [(i, max_track_index)]  # 带合并的轨迹对， 后面轨迹合到前面轨迹上

        # start merge
        for i, j in merge_pair:  # 开始合并轨迹
            TrackUtil.merge(self.tracks[i], self.tracks[j])

        # remove the invalid tracks
        self.tracks = [t for t in self.tracks if t.valid]  # 删除被合并的轨迹 


    def show(self, frame_index, recorder, image):
        # 显示轨迹
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks: # 遍历每一条轨迹
            if len(t.nodes) > 0 and t.age < 2:  # 轨迹存在且最近1帧都检测到
                b = t.nodes[-1].get_box(frame_index, recorder)  #得到最近的轨迹框 
                if b is None:
                    continue
                txt = '({}, {})'.format(t.id, t.nodes[-1].id)  # 轨迹的id， 轨迹的node box在当前帧存储的id
                image = cv2.putText(image, txt, (int(b[0]*w),int((b[1])*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3)
                image = cv2.rectangle(image, (int(b[0]*w),int((b[1])*h)), (int((b[0]+b[2])*w), int((b[1]+b[3])*h)), t.color, 2)
                # 绘制矩形框和文本
        # draw line  绘制轨迹尾巴
        for t in self.tracks:
            if t.age > 1:
                continue
            if len(t.nodes) > self.max_drawing_track:
                start = len(t.nodes) - self.max_drawing_track
            else:
                start = 0
            for n1, n2 in zip(t.nodes[start:], t.nodes[start+1:]):
                b1 = n1.get_box(frame_index, recorder)
                b2 = n2.get_box(frame_index, recorder)
                if b1 is None or b2 is None:
                    continue
                c1 = (int((b1[0] + b1[2]/2.0)*w), int((b1[1] + b1[3])*h))
                c2 = (int((b2[0] + b2[2] / 2.0) * w), int((b2[1] + b2[3]) * h))
                image = cv2.line(image, c1, c2, t.color, 2)

        return image

# The tracker is compatible with pytorch (cuda)
class SSTTracker:
    def __init__(self):
        Track._id_pool = 0                                # 轨迹id初始化
        self.first_run = True
        self.image_size = TrackerConfig.image_size        # 输入图像大小
        self.model_path = TrackerConfig.sst_model_path    # 模型预训练参数路径
        self.cuda = TrackerConfig.cuda                    # 是否cuda
        self.mean_pixel = TrackerConfig.mean_pixel        # 图像中心化
        self.max_object = TrackerConfig.max_object        # 每一帧最多目标个数
        self.frame_index = 0                              # 图像帧索引
        self.load_model()                                 # 加载模型
        self.recorder = FeatureRecorder()                 # 记录跟踪短时信息
        self.tracks = Tracks()                            # 轨迹的集合类

    def load_model(self):
        # load the model
        self.sst = build_sst('test', 900)                 # 创建网络
        if self.cuda:                                     # 加载模型
            cudnn.benchmark = True
            self.sst.load_state_dict(torch.load(config['resume']))
            self.sst = self.sst.cuda()
        else:
            self.sst.load_state_dict(torch.load(config['resume'], map_location='cpu'))
        self.sst.eval()

    def update(self, image, detection, show_image, frame_index, force_init=False):
        '''  对于新的帧图像和对应的检测，进行跟踪
            image： 图像帧
            detection    检测结果
            show_iamge   是否显示图像
            frame_index  当前图像的帧号
            
        Update the state of tracker, the following jobs should be done:
        1) extract the features
        2) stack the features together
        3) get the similarity matrix
        4) do assignment work
        5) save the previous image
        :param image: the opencv readed image, format is hxwx3
        :param detections: detection array. numpy array (l, r, w, h) and they all formated in (0, 1)  所有的检测都归一化到0-1之间
        '''

        self.frame_index = frame_index

        # format the image and detection
        h, w, _ = image.shape
        image_org = np.copy(image)
        image = TrackUtil.convert_image(image)         # 图像转成 1x3xHxW的tensor
        detection_org = np.copy(detection)
        detection = TrackUtil.convert_detection(detection)  # 将detection转成-1到1 的 1xNx1x1x2 的tensor

        # features can be (1, 10, 450)
        features = self.sst.forward_feature_extracter(image, detection)  # 提取每一个target对应的features

        # update recorder
        self.recorder.update(self.sst, self.frame_index, features.data, detection_org)  # 记录当前帧信息
        # 记录信息包括
        # 当前帧的每个target的box， feature
        # 以及当前帧与recorder中依然存在的frame之间targets的相似度矩阵和iou矩阵

        if self.frame_index == 0 or force_init or len(self.tracks.tracks) == 0:  # 跟踪器的起始
            for i in range(detection.shape[1]):  # 每个检测创建一条轨迹
                t = Track()
                n = Node(self.frame_index, i)
                t.add_node(self.frame_index, self.recorder, n)  # 轨迹中有个一个节点
                self.tracks.append(t)
            self.tracks.one_frame_pass()  # 过滤轨迹集合
            # self.frame_index += 1
            return self.tracks.show(self.frame_index, self.recorder, image_org)  # 返回绘制轨迹的图像

        # get tracks similarity
        y, ids = self.tracks.get_similarity(self.frame_index, self.recorder)  # 返回轨迹与当前帧中目标的相似度，以及对应轨迹的id，tracks x dets

        if len(y) > 0:  # 已存在轨迹
            #3) find the corresponding by the similar matrix
            row_index, col_index = linear_sum_assignment(-y)  # 对相似度矩阵匈牙利算法进行匹配
            col_index[col_index >= detection_org.shape[0]] = -1  # 表示轨迹跟丢

            # verification by iou
            verify_iteration = 0
            while verify_iteration < TrackerConfig.roi_verify_max_iteration:
                is_change_y = False
                for i in row_index:
                    box_id = col_index[i]
                    track_id = ids[i]

                    if box_id < 0:
                        continue
                    t = self.tracks.get_track_by_id(track_id)
                    if not t.verify(self.frame_index, self.recorder, box_id):  # 如果存在某一次历史记录与当前目标的iou不满足阈值
                        y[i, box_id] *= TrackerConfig.roi_verify_punish_rate   # 则将相似度矩阵采用iou进行衰减
                        is_change_y = True
                if is_change_y:  # 如果相似度调整了则匹配关系也要进行相应调整
                    row_index, col_index = linear_sum_assignment(-y)
                    col_index[col_index >= detection_org.shape[0]] = -1
                else:
                    break
                verify_iteration += 1

            # print(verify_iteration)  # 输出迭代二分的次数

            #4) update the tracks
            for i in row_index:  # 更新匹配到的轨迹
                track_id = ids[i]
                t = self.tracks.get_track_by_id(track_id)
                col_id = col_index[i]
                if col_id < 0:
                    continue
                node = Node(self.frame_index, col_id)
                t.add_node(self.frame_index, self.recorder, node)

            #5) add new track
            for col in range(len(detection_org)):  # 创建新的轨迹
                if col not in col_index:
                    node = Node(self.frame_index, col)
                    t = Track()
                    t.add_node(self.frame_index, self.recorder, node)
                    self.tracks.append(t)

        # remove the old track
        self.tracks.one_frame_pass()  # 更新recorder

        # merge the tracks
        # if self.frame_index % 20 == 0:
        #     self.tracks.merge(self.frame_index, self.recorder)

        # if show_image:
        image_org = self.tracks.show(self.frame_index, self.recorder, image_org)  # 绘制图像
        # self.frame_index += 1
        return image_org

        # self.frame_index += 1
        # image_org = cv2.resize(image_org, (320, 240))
        # vw.write(image_org)

        # plt.imshow(image_org)
