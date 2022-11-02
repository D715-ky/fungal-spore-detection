import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import shutil
import imgaug as ia
from imgaug import augmenters as iaa
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, AffinityPropagation

ia.seed(1)

##################################### 读取一图像的所有标签#################################################################
def read_xml_annotation(root, image_id):
    # 图像标签根目录
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()

    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)
    return bndboxlist
###########################################################

########################################### 调整一个图的标签##############################################################
def change_xml_list_annotation(root, image_id, new_target, saveroot, id): #id更改后的文件名
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    # 修改xml中的filename800
    elem = tree.find('filename')
    elem.text = (str(id) + '.jpg') # 修改elem元素的值
    # 修改xml中的path
    elem = tree.find('path')
    elem.text = ('VOCdevkit\VOC2007\JPEGImages\\' + str(id) + '.jpg')
    # 修改xml中的图像大小
    Size = tree.find('size')
    Width = Size.find('width')
    Width.text = str(1280)
    Height = Size.find('height')
    Height.text = str(960)

    xmlroot = tree.getroot()
    index = 0
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        lag = new_target[index][4]

        if lag == 0:
            xmlroot.remove(object)
        else:
            new_xmin = new_target[index][0]
            new_ymin = new_target[index][1]
            new_xmax = new_target[index][2]
            new_ymax = new_target[index][3]

            # 判断标签是否合理，是保存，不是删除该object
            if new_xmin <= (new_xmax - 10) and new_ymin <= (new_ymax - 10) and new_xmin >= 0 and new_xmax <= 1280 and new_ymin >= 0 and new_ymax <= 960:
                xmin = bndbox.find('xmin')
                xmin.text = str(new_xmin)
                ymin = bndbox.find('ymin')
                ymin.text = str(new_ymin)
                xmax = bndbox.find('xmax')
                xmax.text = str(new_xmax)
                ymax = bndbox.find('ymax')
                ymax.text = str(new_ymax)
            else:
                xmlroot.remove(object)

        index = index + 1

    tree.write(os.path.join(saveroot, str(id + '.xml')))
###############################################################

############################################# 创建目录##################################################################
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
#########################################################

#######################################################################################################################
def get_cluster_center(bndboxs):
    # 获取边框中心及个数
    box_nums = 0  # 每张影像的标签数量
    boxs = [] # 存放边框信息
    center = []
    for bndbox in bndboxs:
        box_nums += 1

        xmin = bndbox[0]
        ymin = bndbox[1]
        xmax = bndbox[2]
        ymax = bndbox[3]
        x = (xmin + xmax)/2
        y = (ymin + ymax)/2
        boxs.append([x, y, xmin, ymin, xmax, ymax])
        center.append([x, y])
    if center == []:
        print("center为空，没有标签")
    # 获取聚类中心
    LOOP = 0  # 每张影像增强的数量
    if box_nums <= 30:
        if box_nums <= 5:
            LOOP = box_nums
        elif box_nums > 5 and box_nums <= 10:
            LOOP = int(box_nums / 2)
        elif box_nums > 10 and box_nums <= 30:
            LOOP = int(box_nums / 3)

        kmeans = MiniBatchKMeans(n_clusters=LOOP)
        kmeans.fit(center)
        cluster_centers = kmeans.cluster_centers_

        center_flag = 0
        while center_flag == 0:
            for clus_center in cluster_centers:
                center_flag = 0
                for box in boxs:
                    if (box[0] <= clus_center[0] and box[2] >= (clus_center[0] - 640)) or (box[0] >= clus_center[0] and box[4] <= (clus_center[0] + 640)):
                        if (box[1] <= clus_center[1] and box[3] >= (clus_center[1] - 480)) or (box[1] >= clus_center[1] and box[5] <= (clus_center[1] + 480)):
                            center_flag = 1
                    if center_flag == 1: # 当前中心裁剪范围有孢子囊
                        break
                if center_flag == 0: # 存在中心裁剪范围没有孢子囊
                    cluster_centers = []
                    LOOP += 1
                    print(LOOP)
                    kmeans = MiniBatchKMeans(n_clusters=LOOP)
                    kmeans.fit(center)
                    cluster_centers = kmeans.cluster_centers_
                    break

    else:
        mshift = MeanShift(bandwidth=300)
        mshift.fit(center)
        cluster_centers = mshift.cluster_centers_

    return cluster_centers
##################################################################

###############################################获取裁剪位置及次数##########################################################
def get_site_augnum(cluster_centers):
    x = []
    y = []
    augnums = 0
    for center in cluster_centers:
        augnums += 1
        if center[0] < 640:
            center[0] = 640
        if center[0] > 3968:
            center[0] = 3968

        if center[1] < 480:
            center[1] = 480
        if center[1] > 2976:
            center[1] = 2976

        x.append(center[0])
        y.append(center[1])

    return x, y, augnums
#####################################################################

if __name__ == "__main__":
#------------------------------创建文件夹----------------------------------------------------
    IMG_DIR = "./VOCdevkit/VOC2007/JPEGImages"
    XML_DIR = "./VOCdevkit/VOC2007/Annotations"

    AUG_XML_DIR = "./AUG/Annotations"  # 存储增强后的XML文件夹路径
    try:
        shutil.rmtree(AUG_XML_DIR)  # 递归地删除文件
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "./AUG/JPEGImages"  # 存储增强后的影像文件夹路径
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)
#--------------------------------------------------------------------------------------------
    for root, sub_folders, files in os.walk(XML_DIR):
        # root：根目录字母，此处为XML_DIR；sub_folders：根目录下的所有子目录，此处为空；files：根目录下的所有文件。
        for name in files: # name指各文件
            print(name)
            # 确定该图的裁剪位置与裁剪次数
            bndbox = read_xml_annotation(XML_DIR, name)  # 获取对应该图的所有框
            cluster_center = get_cluster_center(bndbox) # 获取聚类中心
            print(cluster_center)
            x, y, AUGLOOP = get_site_augnum(cluster_center) # 获取裁剪位置及次数

            # 确定裁剪增强序列-------------------------------------------------
            seq = [] # 存放裁剪增强序列

            x_site1 = [] # 存放裁剪位置的左上右下的两个坐标
            y_site1 = []
            x_site2 = []
            y_site2 = []

            for epoch1 in range(AUGLOOP):
                x_site1.append(x[epoch1] - 640)
                y_site1.append(y[epoch1] - 480)
                x_site2.append(x[epoch1] + 640)
                y_site2.append(y[epoch1] + 480)
                x1 = (x[epoch1] - 640)/3328
                y1 = (y[epoch1] - 480)/2496
                x2 = 1 - x1
                y2 = 1 - y1
                seq.append(iaa.CropToFixedSize(width=1280, height=960, position=(x2, y2)))

            # 图像增强
            for epoch2 in range(AUGLOOP): # 一次循环增加一张图片
                new_bndbox_list = []
                # 存放是否满足条件的标记
                seq_det = seq[epoch2].to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                # sp = img.size
                img = np.asarray(img)
                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    if bndbox[i][0] >= x_site1[epoch2] and bndbox[i][1] >= y_site1[epoch2] and bndbox[i][2] <= x_site2[epoch2] and bndbox[i][3] <= y_site2[epoch2]:
                        flag = 1
                    else:
                        flag = 0
                    bbs = ia.BoundingBoxesOnImage(  # 复制原图的边框
                        [ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3])],
                        shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]  # 对边框进行增强

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))

                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2, flag])  # for结束后存储对应该图的所有框

                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]  # 对图像进行增强
                path = os.path.join(AUG_IMG_DIR, str(str(name[:-4]) + '_' + str(epoch2)) + '.jpg')
                Image.fromarray(image_aug).save(path)
                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, # name[:-4]去掉.xml的文件名
                                           str(name[:-4]) + '_' + str(epoch2))
                # print(str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
