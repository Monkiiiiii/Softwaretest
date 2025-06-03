import numpy as np
"""
适应实际评估中对延迟检测的容忍（即允许在异常开始后的一个延迟窗口内检测到异常都算正确）

比如电商网站流量监控
异常事件：
    促销活动导致流量激增（t=0 时刻开始）。
业务需求：
    系统不需要在流量刚增加时就报警（这可能是正常促销效果）。
    但如果流量持续异常高（例如超过 30 分钟），则需要触发警报。
    延迟容忍：设置delay=30（假设单位为分钟），允许系统在 30 分钟内判断是否为真正的异常。
"""

#修正预测结果
def adjust_predicts(predict, label, delay=7):
    """
    Calculate adjusted predict labels.

    This function is from AIOps Challenge, KPI Anomaly Detection Competition,
    https://github.com/iopsai/iops/blob/master/evaluation/evaluation.py
    """
    # 1. 找到所有标签变化点（从正常变异常或异常变正常的位置）
    splits = np.where(label[1:] != label[:-1])[0] + 1
    # 2. 确定序列起始状态（第一个点是否是异常）
    is_anomaly = label[0] == 1
    # 3. 创建预测结果的副本，用于存储调整后的结果
    new_predict = np.array(predict)
    # 4. 初始化当前处理的起始位置
    pos = 0

    # 5. 遍历所有变化点
    for sp in splits:
        # 6. 如果当前段是异常段
        if is_anomaly:
            # 7. 检查在延迟窗口内是否有预测为异常的点
            #    - 查看从当前起始位置 pos 到 pos+delay 范围内是否有1
            #    - 但不能超过当前段的结束位置 sp
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                # 8. 如果延迟窗口内有异常预测，将整个异常段标记为异常
                new_predict[pos: sp] = 1
            else:
                # 9. 否则整个异常段标记为正常
                new_predict[pos: sp] = 0
        # 10. 切换状态（异常<->正常）
        is_anomaly = not is_anomaly

        # 11. 更新起始位置到当前变化点
        pos = sp
    # 12. 处理最后一段（序列末尾部分）
    sp = len(label)

    # 13. 如果最后一段是异常段
    if is_anomaly:  # anomaly in the end
        # 14. 同样检查延迟窗口内是否有异常预测
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            # 15. 有则标记整个段为异常
            new_predict[pos: sp] = 1
        else:
            # 16. 无则标记整个段为正常
            new_predict[pos: sp] = 0

    # 17. 返回调整后的预测结果
    return new_predict

"""
整体的思路就是
允许算法在异常开始后的 delay 时间内检测到异常，只要能在窗口内反应过来这个数据是有问题的就行，不要求一定要在第一时间反应
将时间序列划分为连续的异常段和正常段，只对异常的段落进行判断
"""
