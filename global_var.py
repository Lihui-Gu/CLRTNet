import math

def _init():  # 初始化
    global loss
    loss = {}
    global step
    step = 0
    global K, T
    K = 10
    T = 0.5

def add_part_loss(cls_loss, reg_loss, seg_loss, iou_loss):
    global loss
    loss[step] = [cls_loss, reg_loss, seg_loss, iou_loss]

def get_sum_loss(cls_loss, reg_loss, seg_loss, iou_loss):
    global step, loss, K, T
    # weight init
    if step == 0 or step == 1:
        cls_weight = 1.
        reg_weight = 1.
        iou_weight = 1.
        seg_weight = 1.
    else:
        w_0 = loss[step - 1][0]/loss[step - 2][0]
        w_1 = loss[step - 1][1]/loss[step - 2][1]
        w_2 = loss[step - 1][2]/loss[step - 2][2]
        w_3 = loss[step - 1][3]/loss[step - 2][3]
        exp_sum = math.exp(w_0 / T) + math.exp(w_1 / T) + math.exp(w_2 / T) + math.exp(w_3 / T)
        cls_weight = K * math.exp(w_0 / T ) / exp_sum
        reg_weight = K * math.exp(w_1 / T ) / exp_sum
        seg_weight = K * math.exp(w_2 / T ) / exp_sum
        iou_weight = K * math.exp(w_3 / T ) / exp_sum
    # add part loss for this step
    loss[step] = [cls_loss, reg_loss, seg_loss, iou_loss]
    step += 1
    return cls_loss * cls_weight + reg_weight * reg_loss  + iou_weight * iou_loss  + seg_weight * seg_loss * 1.