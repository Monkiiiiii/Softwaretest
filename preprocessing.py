import os
import pandas as pd
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from utils import complete_timestamp, standardize_kpi


def process_kpi_data(train_path, test_path, out_path, standard=False, filled_type="linear"):
    """
    预处理KPI数据集
    填充缺失数据，然后将训练和测试数据保存到同一个.csv文件中

    参数:
        train_path: 训练数据路径
        test_path: 测试数据路径
        out_path: 处理后数据保存路径
        standard: 是否标准化曲线 [True/False]
        filled_type: 填充缺失数据的方法类型 [linear/periodic]
    """
    # 读取数据并选择需要的列
    train_df = read_data(train_path)[['timestamp', 'value', 'label', 'KPI ID']]
    test_df = read_data(test_path)[['timestamp', 'value', 'label', 'KPI ID']]

    # 创建结果DataFrame
    data_df = pd.DataFrame(columns=['timestamp', 'value', 'label', 'KPI ID', "missing", "is_test"])
    group_list = [train_df.groupby("KPI ID"), test_df.groupby("KPI ID")]

    # 用于存储标准化所需的均值和标准差
    mean_dict = {}
    std_dict = {}
    
    # 处理训练数据和测试数据
    for i in range(len(group_list)):
        for name, group in group_list[i]:
            print(name)
            # 创建临时DataFrame
            temp_df = pd.DataFrame(columns=['timestamp', 'value', 'label', 'KPI ID', "missing", "is_test"])
            timestamp = group["timestamp"].values
            value = group["value"].values
            label = group["label"].values

            # 完成时间戳，识别缺失值，返回完整时间戳和缺失标记
            timestamp, missing, (value, label), interval, max_miss_num = complete_timestamp(timestamp, (value, label))

            # 标准化训练和测试数据
            if standard:
                if i == 0:  # 训练数据：计算均值和标准差
                    value, mean, std = standardize_kpi(value, excludes=np.logical_or(label, missing))
                    mean_dict[name] = mean
                    std_dict[name] = std
                else:  # 测试数据：使用训练数据的均值和标准差
                    mean = mean_dict[name]
                    std = std_dict[name]
                    value, _, _ = standardize_kpi(value, mean=mean, std=std)

            # 将label中的NaN替换为0
            label[np.isnan(label)] = 0
            print("max_miss_num: ", max_miss_num)

            # 填充临时DataFrame
            temp_df['timestamp'], temp_df["missing"], temp_df["value"], temp_df["label"] = \
                timestamp, missing, value, label
            temp_df["KPI ID"], temp_df["is_test"] = name, i  # i标记训练或测试数据

            # 根据选择的填充类型进行缺失值填充
            period = 1440 * 60 // interval  # 计算周期（假设一天的分钟数）
            length = len(value)
            num_padding = (length // period + 1) * period - length  # 计算需要填充的数量
            
            if filled_type == "linear":
                # 线性插值填充
                temp_df['value'].interpolate(method='linear', inplace=True)
            elif filled_type == "periodic":
                # 周期性填充：用前一个周期的值填充长缺失段
                tmp_value = np.concatenate((value, np.full([num_padding], np.nan)))
                tmp_2d_array = np.reshape(tmp_value, (-1, period))
                # 计算每行的NaN数量
                nan_num = np.sum(tmp_2d_array != tmp_2d_array, axis=1)
                
                for k in range(tmp_2d_array.shape[0]):
                    if nan_num[k] > 5:  # 如果NaN数量超过5
                        # 找到所有连续的NaN段
                        filled_idx = np.where(np.isnan(tmp_2d_array[k]))[0]
                        filled_idx_list = [list(g) for g in mit.consecutive_groups(filled_idx)]
                        
                        for m in range(len(filled_idx_list)):
                            idx_seg = np.array(filled_idx_list[m])
                            if 5 < len(idx_seg) < 0.6 * period:  # 中等长度缺失段
                                # 使用前一周期的值加上平均值差异的一半
                                mean_diff = np.nanmean(tmp_2d_array[k]) - np.nanmean(tmp_2d_array[k - 1])
                                tmp_2d_array[k, idx_seg] = tmp_2d_array[k - 1, idx_seg] + mean_diff / 2
                            elif len(idx_seg) > 0.6 * period:  # 长缺失段
                                # 直接使用前一周期的值
                                tmp_2d_array[k, idx_seg] = tmp_2d_array[k - 1, idx_seg]
                
                # 展平数组并截取到原始长度
                flatten_value = tmp_2d_array.flatten()[: length]
                temp_df['value'] = flatten_value
                
                # 最后对短缺失段进行线性插值
                temp_df['value'].interpolate(method='linear', inplace=True)
            else:
                raise TypeError("此填充类型不可用！")
            
            # 将处理好的临时DataFrame添加到结果中
            data_df = pd.concat([data_df, temp_df], ignore_index=True)
            # show_filled_data(name, filled_type, temp_df['value'].values, temp_df['missing'].values)

    # 保存处理好的数据到CSV文件
    data_df.to_csv(out_path, index=False)
    # 可选：保存标准化信息
    # with open("mean_std_info.txt", "w") as f:
    #     f.write(str({"mean": mean_dict, "std": std_dict}))


def show_filled_data(kip_id, fill_type, data, missing, label=None):
    """
    可视化填充后的数据
    
    参数:
        kip_id: KPI标识
        fill_type: 填充类型
        data: 填充后的数据
        missing: 缺失标记
        label: 标签(可选)
    """
    # 找到连续的缺失段
    missing_group = [list(g) for g in mit.consecutive_groups(np.where(missing)[0])]
    missing_segs = [(g[0], g[-1]) if g[0] != g[-1] else (g[0] - 1, g[0] + 1) for g in missing_group]

    # 绘制填充结果
    _len = len(data)
    xs = np.linspace(0, _len - 1, _len)
    plt.figure(figsize=(9, 6))
    plt.title("id: {}, type: {}".format(kip_id, fill_type))
    plt.xticks([])
    plt.yticks([])
    plt.plot(xs, data, "mediumblue")
    
    # 用绿色标记填充的缺失段
    for seg in missing_segs:
        seg_x = np.linspace(seg[0], seg[1], seg[1] - seg[0] + 1).astype(dtype=int)
        plt.plot(seg_x, data[seg_x], color="g")

    # 可选：用红色标记异常点
    if label is not None:
        label_group = [list(g) for g in mit.consecutive_groups(np.where(label)[0])]
        label_segs = [(g[0], g[-1]) if g[0] != g[-1] else (g[0] - 1, g[0] + 1) for g in label_group]
        for seg in label_segs:
            seg_x = np.linspace(seg[0], seg[1], seg[1] - seg[0] + 1).astype(dtype=int)
            plt.plot(seg_x, data[seg_x], color="r")
    plt.show()


def read_data(path):
    """根据文件扩展名读取不同格式的数据文件"""
    if path.endswith(".hdf"):
        df = pd.read_hdf(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise TypeError("当前文件类型不可用!")

    return df

# 主函数：设置路径和参数，调用数据处理函数
if __name__ == "__main__":
    train_path = "./data/AIOps/phase2_train.csv"
    test_path = "./data/AIOps/phase2_ground_truth.hdf"
    out_path = "./data/AIOps/total_data.csv"
    process_kpi_data(train_path, test_path, out_path, standard=False, filled_type="periodic")
