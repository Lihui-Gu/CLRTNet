import json
import numpy as np
import cv2
import os
import argparse
import random


"""
def gen_label_for_port(args, image_set):
    H, W = 1200, 1920
    SEG_WIDTH = 30
    save_dir = args.savedir
    os.makedirs(os.path.join(args.root, args.savedir, "list"), exist_ok=True)
    list_f = open(
        os.path.join(args.root, args.savedir, "list",
                     "{}_gt.txt".format(image_set)), "w")
    #json_path = os.path.join(args.root, args.savedir, "{}.json".format(image_set))
    img_file = os.path.join(args.root, "labels")
    image_list = os.listdir(img_file)
    # print(image_list)
    for img_name in image_list:
        seg_img = np.zeros((H, W, 3))
        list_str = []  # str to be written to list.txt
        # 车道线分类，最后5位
        list_str.append('1')
        img = cv2.imread(img_file + "/" + img_name)
        idx = np.unique(img)
        # print(img_name)
        # print(idx)
        lane = [0, 0, 0, 0, 0, 0]
        for i in idx:
            if not i:
                continue
            elif(i == 1):
                lane[0] = 1
            elif(i == 2):
                lane[1] = 1
            elif(i == 4):
                lane[2] = 1
            elif(i == 5):
                lane[3] = 1
            elif(i == 7 or i == 6):
                lane[4] = 1
            elif(i == 8):
                lane[5] = 1
            else:
                print("Read Error!")
        for i in lane:
            if i == 1:
                list_str.append('1')
            else:
                list_str.append('0')

        seg_path = os.path.join(args.root, args.savedir)
        img_path = "/".join([
            "images", img_name
        ])
        seg_path = "/".join([
            args.savedir, img_name[:-3] + "png"
        ])
        if seg_path[0] != '/':
            seg_path = '/' + seg_path
        if img_path[0] != '/':
            img_path = '/' + img_path
        list_str.insert(0, seg_path)
        list_str.insert(0, img_path)
        list_str = " ".join((list_str)) + "\n"
        list_f.write(list_str)
"""

# step2
def generate_img_path_set(args):
    img_file = os.path.join(args.root, "clips", "1010")
    img_list = os.listdir(img_file)
    img_path_set = []
    for img_name in img_list:
        img_name = "/".join([
            img_file, img_name
        ])
        if img_name != '/':
            img_name = '/' + img_name
        img_path_set.append(img_name)
    # print(img_path_set)
    return img_path_set


# step1 生成一个json文件包含所有
def generate_label(args):
    save_dir = os.path.join(args.root, args.savedir)
    os.makedirs(save_dir, exist_ok=True)
    img_path_set = generate_img_path_set(args)
    print("generate port dataset...")
    generate_json_file(args, "label_data_1010.json", img_path_set)
    split_dataset("label_data_1010.json")
    # gen_label_for_port(args, "train_val")


# step3
def generate_json_file(args, json_file, image_path_set):
    with open(os.path.join(args.root, json_file), "w") as outfile:
        for img_path in image_path_set:
            # 行方向采样点
            h_samples = list(range(160, 720, 10))
            # 图片名称
            img_name = img_path.split('/')[-1][:-3] + 'png'
            img_path = 'clips/1010/' + img_name[:-3] + 'jpg'
            # 获取图片
            img = cv2.imread(os.path.join(args.root, "labels", img_name))
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_NEAREST)
            # 出现标签
            idx = np.unique(img)
            # print(idx)
            lane = [None for i in range(6)]
            for i in idx:
                if not i:
                    continue
                elif (i == 1):
                    lane[0] = [-2 for j in h_samples]
                elif (i == 2):
                    lane[1] = [-2 for j in h_samples]
                elif (i == 4):
                    lane[2] = [-2 for j in h_samples]
                elif (i == 5):
                    lane[3] = [-2 for j in h_samples]
                elif (i == 7 or i == 6):
                    lane[4] = [-2 for j in h_samples]
                elif (i == 8):
                    lane[5] = [-2 for j in h_samples]
                else:
                    print("Read Error!")

            for h in range(len(h_samples)):
                # print("当前是{}行".format(h_samples[h]))
                img_line = img[h_samples[h]].sum(axis=1) / 3
                for i in idx:
                    l = np.argwhere(img_line == i)
                    if i == 0 or len(l) == 0:
                        continue
                    elif (i == 1):
                        lane[0][h] = int(np.ceil(np.median(l)))
                    elif (i == 2):
                        lane[1][h] = int(np.ceil(np.median(l)))
                    elif (i == 4):
                        lane[2][h] = int(np.ceil(np.median(l)))
                    elif (i == 5):
                        lane[3][h] = int(np.ceil(np.median(l)))
                    elif (i == 7 or i == 6):
                        lane[4][h] = int(np.ceil(np.median(l)))
                    elif (i == 8):
                        lane[5][h] = int(np.ceil(np.median(l)))
                    else:
                        print("Read Error!")
            # 通过h_sameple 发现标记记录到lane中
            # 把None去掉！！！
            _lane = []
            for i in lane:
                if i is not None:
                    _lane.append(i)
            json_line = {"lanes": _lane, "h_samples": h_samples, "raw_file": img_path}
            outfile.write(str(json_line) + "\n")


# step 4 划分数据集
def split_dataset(json_file):
    json_line = []
    with open(os.path.join(args.root, json_file)) as infile:
        for line in infile:
            json_line.append(line)
    random.shuffle(json_line)
    sum_num = len(json_line)
    train_num = int(0.6 * sum_num)
    valid_num = int(0.2 * sum_num)
    test_num = sum_num - train_num - valid_num
    print("未数据增强训练集数量{}".format(train_num))
    print("验证集数量{}".format(valid_num))
    print("测试集数量{}".format(test_num))
    with open(os.path.join(args.root, 'label_data_train.json'), 'w') as outfile:
        for line in json_line[0:train_num]:
            outfile.write(str(line))
    with open(os.path.join(args.root, 'label_data_valid.json'), 'w') as outfile:
        for line in json_line[train_num:train_num + valid_num]:
            outfile.write(str(line))
    with open(os.path.join(args.root, 'label_data_test.json'), 'w') as outfile:
        for line in json_line[valid_num:]:
            outfile.write(str(line))


def add_gaussian_noise(image_in, noise_sigma):
    """
    给图片添加高斯噪声
    image_in:输入图片
    noise_sigma：
    """
    temp_image = np.float64(np.copy(image_in))

    h, w, _ = temp_image.shape
    # 标准正态分布*noise_sigma
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
    return noisy_image

def data_augmentation(args, json_file):
    # 获取原始图像的文件
    json_path = os.path.join(args.root, json_file)
    json_noisy = os.path.join(args.root, "label_data_train_noisy.json")
    with open(json_noisy, 'w') as outfile:
        with open(json_path) as infile:
            for line in infile:
                json_dict = eval(line)
                raw_file = json_dict['raw_file']
                raw_img = cv2.imread(os.path.join(args.root, raw_file))
                noisy_img = add_gaussian_noise(raw_img, 10)
                cv2.imwrite(os.path.join(args.root, raw_file[:-4] + 'gauss_10.jpg'), noisy_img)
                img_name = raw_file.split('/')[-1][:-4]
                seg_img = cv2.imread(os.path.join(args.root, "labels", img_name + '.png'))
                cv2.imwrite(os.path.join(args.root, "labels", img_name + 'gauss_10.png'), seg_img)
                outfile.write(line) # 写入初始数据
                json_dict['raw_file'] = raw_file[:-4] + 'gauss_10.jpg'
                outfile.write(str(json_dict) + '\n')



if __name__ == '__main__':
    # python tools/generate_port_json.py --root data/port
    parser = argparse.ArgumentParser()
    # data/port
    parser.add_argument('--root',
                        required=True,
                        help='The root of the Port dataset')
    parser.add_argument('--savedir',
                        type=str,
                        default='seg_label',
                        help='The root of the Port dataset')
    args = parser.parse_args()
    generate_label(args)
    # 数据增强 仅仅对训练集进行操作
    data_augmentation(args, "label_data_train.json")