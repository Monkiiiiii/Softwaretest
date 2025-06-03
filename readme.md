# FluxEV
The code is for our paper ["FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection"](https://dl.acm.org/doi/10.1145/3437963.3441823) 
and this paper has been accepted by WSDM 2021. [\[中文解读\]](https://mp.weixin.qq.com/s/zUQJnbWQx5qf1A_gXnnAbQ)

  声明这是论文《FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection》的代码，该论文被WSDM 2021接收

😉 About Name: "Flux" means the Fluctuation, "EV" denotes the Extreme Value.     

  “Flux”代表波动（Fluctuation），“EV”代表极值（Extreme Value）。

💫 Good News: FluxEV has been integrated into [**OATS**](https://github.com/georgian-io/pyoats), a convenient Python library providing various approaches to time series anomaly detection. Thanks for their awesome work!

## Requirements
* numpy
* numba
* scipy
* pandas
* scikit-learn
* matplotlib (plot)
* more-itertools (plot)


## Datasets
1. KPI.        
* Competition Website Link: <http://iops.ai/dataset_detail/?id=10> (⚠️ Seems like the official competition website is down!)        
* Tsinghua Netman GitHub Link: <https://github.com/NetManAIOps/KPI-Anomaly-Detection/tree/master/Finals_dataset>
  来自AIOps竞赛的运维监控指标（如服务器CPU、网络流量等），是**时间序列异常检测**的标准数据集

1. Yahoo.
*  Official Website Link: <https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70>
  雅虎实验室发布的带标注的时序数据（如网站访问量、广告点击量等）

  代码支持两个数据集：KPI和Yahoo
  提供了数据集的来源链接
  KPI数据集：如果官方链接失效，需要使用提供的GitHub链接下载。注意数据预处理（如填充缺失点）已经包含在`preprocessing.py`中

## Instructions
`preprocessing.py`: 
* Fill the missing points for KPI dataset.
  专门处理KPI数据集的缺失值填充

`spot_pipe.py`: 
* SPOT function is modified to be a part of FluxEV for streaming detection;
* MOM(Method of Moments) is added as one of parameter estimation methods;
* For the original code, please refer to [SPOT (Streaming Peaks-Over-Threshold)](https://github.com/Amossys-team/SPOT)
  修改了SPOT（Streaming Peaks-Over-Threshold）方法（Streaming Peaks-Over-Threshold（流式阈值超越峰检测）：对数据流实时计算动态阈值；当数据点超过阈值时标记为异常；基于极值理论(EVT)建模尾部分布）
  将其作为FluxEV的一部分用于流式检测；（数据像水流一样持续到达）
  增加了MOM（矩估计）作为参数估计方法之一；
  并注明了原始SPOT代码的来源


`eval_methods.py`: 
* The adjustment strategy is consistent with AIOps Challenge, [KPI Anomaly Detection Competition](http://iops.ai/competition_detail/?competition_id=5&flag=1).
* The original evaluation script is available at [iops](https://github.com/iopsai/iops/blob/master/evaluation/evaluation.py).
  评估方法，与AIOps Challenge的评价策略一致。原始评估脚本来自iops仓库

`main.py`: 
* Implement streaming detection of FluxEV.
  实现FluxEV的流式检测



## Run
```
python main.py --dataset=KPI
```

```
python main.py --dataset=Yahoo
```

## Citation
```
@inproceedings{li2021fluxev,
  title={FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection},
  author={Li, Jia and Di, Shimin and Shen, Yanyan and Chen, Lei},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  pages={824--832},
  year={2021}
}
```

