import os
import os.path as osp
import time
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import f1_score, recall_score, precision_score
import argparse
import numba

import sys
# 添加父目录到系统路径，以便导入spot_pipe
sys.path.append('..')  # import the upper directory of the current file into the search path
 # 从spot_pipe导入SPOT类
from spot_pipe import SPOT
# 导入评估方法中的调整预测函数
from eval_methods import adjust_predicts


#定义辅助函数

#`calc_ewma`: 指数加权移动平均（支持两种计算方式，adjust参数控制）
def calc_ewma(input_arr, alpha=0.2, adjust=True):
    """
    Here, we use EWMA as a simple predictor.     在此，我们把EWMA当作一种简单的预测器

    Args:                                        参数：
        input_arr: 1-D input array               input_arr：一维输入数组
        alpha: smoothing factor, (0, 1]          alpha：平滑因子，范围为(0, 1]
        adjust:
            (1) When adjust=True(faster), the EW function is calculated using weights w_i=(1-alpha)^i;
            当adjust=True（速度更快）时，会运用权重w_i =(1 - alpha)^i来计算EW函数；
            (2) When adjust=False, the exponentially weighted function is calculated recursively.
            当adjust=False时，则通过递归的方式计算指数加权函数。
    Returns:
        Exponentially Weighted Average Value     指数加权平均值
    """
    arr_len = len(input_arr)

    if adjust:
        #首先生成一个从arr_len - 1到 0 的幂次数组
        power_arr = np.array(range(len(input_arr)-1, -1, -1))
        #接着创建一个由1 - alpha组成的数组
        a = np.full(arr_len, 1-alpha)
        #然后计算权重数组(1 - alpha)^i
        weight_arr = np.power(a, power_arr)
        #最后按照标准的加权平均公式来计算结果
        ret = np.sum(input_arr * weight_arr) / np.sum(weight_arr)

    else:
        #先初始化一个数组，将第一个元素设为输入数组的首元素
        ret_arr = [input_arr[0]]
        #按照递归公式S_t = alpha * x_t+(1 - alpha)*S_{t - 1}依次计算后续元素
        for i in range(1, arr_len):
            temp = alpha * input_arr[i] + (1 - alpha) * ret_arr[-1]
            ret_arr.append(temp)
        #最终返回的是数组的最后一个元素
        ret = ret_arr[-1]
    return ret

#`calc_ewma_v2`: 使用numba加速的EWMA计算（非调整方式，循环计算）
@numba.jit(nopython=True)
def calc_ewma_v2(input_arr, alpha=0.2):
    arr_len = len(input_arr)
    ret_arr = [input_arr[0]]
    for i in range(1, arr_len):
        temp = alpha * input_arr[i] + (1 - alpha) * ret_arr[-1]
        ret_arr.append(temp)
    ret = ret_arr[-1]
    return ret

#`calc_first_smooth`: 一阶平滑（计算当前窗口标准差与前一个窗口标准差的差值，取非负）
def calc_first_smooth(input_arr):
    return max(np.nanstd(input_arr) - np.nanstd(input_arr[:-1]), 0)  # if std_diff < 0, return 0

#`calc_second_smooth`: 二阶平滑（计算当前窗口最大值与前一个窗口最大值的差值，取非负）
def calc_second_smooth(input_arr):
    return max(np.nanmax(input_arr) - np.nanmax(input_arr[:-1]), 0)  # if max_diff < 0, return 0


"""
data_arr	一维数据序列，比如一天的温度、CPU 利用率等
train_len	用来“训练阈值”的数据长度，之前的数据用来学习正常分布
period	    数据的周期，比如一天是 1440 分钟就填 1440
smoothing	平滑处理步骤数（1或2）
s_w	        用于预测的滑动窗口大小
p_w	        用于周期性平滑的周期数（往前回看多少天）
half_d_w	用于“处理数据漂移”的窗口一半大小
q	        SPOT检测的风险系数（越小越敏感）
estimator	用什么方法估计数据分布，“MOM”更稳，“MLE”更准但对异常敏感
"""
def detect(data_arr, train_len, period, smoothing=2,
           s_w=10, p_w=7, half_d_w=2, q=0.001,
           estimator="MOM"):
    """
    Args:
        data_arr: 1-D data array.
        train_len: data length for training.
        period: data period, usually indicate the point num of one day;
                one-min level: 1440, one-hour level: 24.
        smoothing: number of smoothing operations;
                   1->only first-step smoothing, 2->two-step smoothing.
        s_w: sequential window size for detecting anomaly.
        p_w: periodic window size for detecting anomaly.
        half_d_w: half window size of handling data drift, for detecting anomaly.
        q: risk coefficient of SPOT for detecting anomaly;
           usually between 10^-3 and 10^-5 to have a good performance.
        estimator: estimation method for data distribution in SPOT, "MOM" or "MLE".
    Returns:
        alarms: detection results, 0->normal, 1->abnormal.
    """


    data_len = len(data_arr)# 数据总长度
    spot = SPOT(q, estimator=estimator)  # 创建一个SPOT检测器

    d_w = half_d_w * 2 # d_w 是完整的数据漂移窗口大小（对称的）

    # Calculate the start index to extract anomaly features
    fs_idx = s_w * 2  # start index for first smoothing
    fs_lm_idx = fs_idx + d_w  # start index for local max array of first smoothing
    ss_idx = fs_idx + half_d_w + period * (p_w - 1)  # start index for second smoothing

    #原始预测误差（原始数据 - 预测值）
    pred_err = np.full(data_len, np.nan)  # prediction error array (predictor: ewma)
    #第一次平滑误差
    fs_err = np.full(data_len, np.nan)  # first smoothing error array
    #第一次平滑误差的局部最大值
    fs_err_lm = np.full(data_len, np.nan)  # local max array for the first smoothing error
    #第二次平滑误差
    ss_err = np.full(data_len, np.nan)  # second smoothing error array
    
    #报警结果 & 阈值列表
    th, alarms = [], []

    #模式一：一级平滑
    if smoothing == 1:
        for i in range(s_w, data_len):
            # calculate the predicted value Pi and prediction error Ei
            Pi = calc_ewma_v2(data_arr[i - s_w: i])
            Ei = data_arr[i] - Pi
            pred_err[i] = Ei

            # first smoothing
            if i >= fs_idx:
                FSEi = calc_first_smooth(pred_err[i - s_w: i + 1])  # fixed index
                fs_err[i] = FSEi

            # SPOT Detection
            if i == train_len - 1:  # initialize SPOT detector using training data
                init_data = fs_err[fs_idx: i + 1]
                spot.fit(init_data)
                spot.initialize()

            if i >= train_len:  # detect the testing point one by one
                # th_s: the current threshold(dynamic); alarm_s: 0->normal, 1->abnormal
                th_s, alarm_s = spot.run_step(fs_err[i])  # apply the detection

                th.append(th_s)
                alarms.append(alarm_s)

    elif smoothing == 2:
        for i in range(s_w, data_len):
            # calculate the predicted value Pi and prediction error Ei
            Pi = calc_ewma_v2(data_arr[i - s_w: i])
            Ei = data_arr[i] - Pi
            pred_err[i] = Ei

            if i >= fs_idx:
                # the first smoothing
                FSEi = calc_first_smooth(pred_err[i - s_w: i + 1])  # fixed index
                fs_err[i] = FSEi

                # extract the local max value
                if i >= fs_lm_idx:
                    FSEi_lm = max(fs_err[i - d_w: i + 1])
                    fs_err_lm[i - half_d_w] = FSEi_lm  # fixed index

                # the second smoothing
                if i >= ss_idx:
                    tem_arr = np.append(fs_err_lm[i - period * (p_w - 1): i: period], fs_err[i])
                    SSEi = calc_second_smooth(tem_arr)
                    ss_err[i] = SSEi

            # SPOT Detection
            if i == train_len - 1:  # initialize SPOT detector using training data
                init_data = ss_err[ss_idx: i + 1]
                spot.fit(init_data)
                spot.initialize()

            if i >= train_len:  # detect the testing point one by one
                # th_s: the current threshold(dynamic); alarm_s: 0->normal, 1->abnormal
                th_s, alarm_s = spot.run_step(ss_err[i])  # apply the detection

                # if detect an anomaly, update its features;
                # avoid affecting feature extraction of subsequent points
                if alarm_s:
                    fs_err[i] = np.nan
                    FSEi_lm = max(fs_err[i - d_w: i + 1])
                    fs_err_lm[i - half_d_w] = FSEi_lm

                th.append(th_s)
                alarms.append(alarm_s)

    alarms = np.array(alarms)
    return alarms


def read_yahoo_data(path):
    file_name = path.split("/")[-1][:-4]
    dir_id = int(path.split("/")[-2][1])

    if dir_id < 3:
        timestamp_col = "timestamp"
        value_col = "value"
        label_col = "is_anomaly"
    else:
        timestamp_col = "timestamps"
        value_col = "value"
        label_col = "anomaly"

    df = pd.read_csv(path)[[timestamp_col, value_col, label_col]]
    # convert to int dtype
    df[[timestamp_col, label_col]] = df[[timestamp_col, label_col]].astype(int)
    df = df.rename(columns={timestamp_col: "timestamp",
                            value_col: "value",
                            label_col: "label"})

    return df, file_name, dir_id

# 处理yahoo数据集的主函数
def main_yahoo(args, data_dir):
    ret_file_path = osp.join(data_dir, args.ret_file).format(args.estimator,
                                                             args.s_w, args.p_w,
                                                             args.half_d_w, args.q)

    file_list = []
    for _id in [1, 2, 3, 4]:
        sub_dir = osp.join(data_dir, "A{}Benchmark".format(_id))
        file_list += glob(sub_dir + "/*.csv")

    y_true, y_pred = [], []
    for _path in file_list:
        data_df, file_name, dir_id = read_yahoo_data(_path)

        print(file_name)
        # timestamp = data_df["timestamp"].values  # timestep array
        value = data_df["value"].values  # data array
        label = data_df["label"].values  # label array

        period = 24  # hour-level data

        if not args.train_len:
            train_len = len(value) // 2
        else:
            train_len = args.train_len

        if dir_id == 2:
            smoothing = 1
        else:
            smoothing = 2

        label_test = label[train_len:]
        alarms = detect(value, train_len, period, smoothing,
                        args.s_w, args.p_w, args.half_d_w, args.q,
                        estimator=args.estimator)

        ret_test = adjust_predicts(predict=alarms, label=label_test, delay=args.delay)

        y_true.append(label_test)
        y_pred.append(ret_test)

    y_true_arr, y_pred_arr = np.concatenate(y_true), np.concatenate(y_pred)
    f_score = f1_score(y_true_arr, y_pred_arr)
    recall = recall_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr)

    with open(ret_file_path, "a") as f:
        f.write("Total F1/Recall/Precision score: {}, {}, {}\n".format(f_score, recall, precision))

# 处理kpi数据集的主函数
def main_kpi(args, base_dir, data_path):
    ret_dir = osp.join(base_dir, "results")
    ret_file_path = osp.join(ret_dir, args.ret_file).format(args.estimator,
                                                             args.s_w, args.p_w,
                                                             args.half_d_w, args.q)
    if not osp.exists(ret_dir):
        os.makedirs(ret_dir)

    # read data and convert several data type to int
    data_df = pd.read_csv(data_path)
    data_df[["timestamp", "label", "missing", "is_test"]] = \
        data_df[["timestamp", "label", "missing", "is_test"]].astype(int)

    y_true, y_pred = [], []
    for name, group in data_df.sort_values(by=["KPI ID", "timestamp"], ascending=True).groupby("KPI ID"):
        print(name)

        group.reset_index(drop=True, inplace=True)
        timestamp = group["timestamp"].values
        value = group["value"].values
        label = group["label"].values
        missing = group["missing"].values

        if not args.train_len:
            train_len = sum(group["is_test"].values == 0)
        else:
            train_len = args.train_len

        interval = timestamp[1] - timestamp[0]
        period = 1440 * 60 // interval

        smoothing = 2
        label_test = label[train_len:]
        test_missing = missing[train_len:]
        alarms = detect(value, train_len, period, smoothing,
                        args.s_w, args.p_w, args.half_d_w, args.q,
                        estimator=args.estimator)

        alarms[np.where(test_missing == 1)] = 0  # set the results of missing points to 0
        ret_test = adjust_predicts(predict=alarms, label=label_test, delay=args.delay)

        y_true.append(label_test)
        y_pred.append(ret_test)

    y_true_arr, y_pred_arr = np.concatenate(y_true), np.concatenate(y_pred)

    f_score = f1_score(y_true_arr, y_pred_arr)
    recall = recall_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr)

    with open(ret_file_path, "a") as f:
        f.write("Total F1/Recall/Precision score: {}, {}, {}\n".format(f_score, recall, precision))

# 主程序入口（解析参数，调用相应主函数）
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Detection of FluxEV")
    parser.add_argument('--dataset', type=str, default='Yahoo')
    parser.add_argument('--delay', type=int, default=7,
                        help="delay point num for evaluation")
    parser.add_argument('--q', type=float, default=0.003,
                        help="risk coefficient for SPOT")

    parser.add_argument('--s_w', type=int, default=10,
                        help="sequential window size "
                             "to extract the local fluctuation and do the first-step smoothing")
    parser.add_argument('--p_w', type=int, default=5,
                        help="periodic window size to do the second-step smoothing")
    parser.add_argument('--half_d_w', type=int, default=2,
                        help="half window size for handling data drift")

    parser.add_argument('--estimator', type=str, default="MOM",
                        help="estimation method for SPOT, 'MOM' or 'MLE'")
    parser.add_argument('--train_len', type=int, default=None,
                        help="data length for training (initialize SPOT), "
                             "if None(default), the program will set it as the half of the data length")

    parser.add_argument('--ret_file', type=str, default='{}-s{}-p{}-d{}-q{}.txt')

    Flags = parser.parse_args()
    if Flags.dataset == "KPI":
        base_dir = "./data/AIOps/"
        data_path = osp.join(base_dir, "total_data.csv")
        Flags.delay = 7
        Flags.q = 0.003
        main_kpi(Flags, base_dir, data_path)
    elif Flags.dataset == "Yahoo":
        data_dir = "./data/Yahoo"
        Flags.delay = 3
        Flags.q = 0.001
        main_yahoo(Flags, data_dir)

