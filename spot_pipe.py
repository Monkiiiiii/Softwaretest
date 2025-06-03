#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:08:16 2016

@author: Alban Siffer
@company: Amossys
@license: GNU GPLv3
"""

from scipy.optimize import minimize
from math import log, floor
import numpy as np
import pandas as pd

# colors for plot
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'

"""
================================= MAIN CLASS ==================================
"""


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
    这个类允许在单变量数据集上运行SPOT算法

    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        检测水平（风险值），由用户设定，proba=0.001 意味着在 1000 个正常数据点中，平均允许 1 个被误判为异常。
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        当前阈值（正常和异常事件的分界线），高于此值的数据点被视为异常。随着新数据的流入，这个阈值会不断更新
        
    data : numpy.array
        stream
        数据流
        
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
        初始观测数据批次（用于校准/初始化步骤）
        算法需要一段初始数据（init_data）来估计正常数据的分布
        
    init_threshold : float
        initial threshold computed during the calibration step
        在校准步骤中计算的初始阈值
        
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
        峰值数组（超过初始阈值的极值）
        峰值（peaks）是指超过初始阈值的数据点
        
    n : int
        number of observed values
        已观测值的数量
        
    Nt : int
        number of observed peaks
        已观测峰值的数量
    """

    #spot的构造函数
    def __init__(self, q=1e-4, estimator="MOM"):
        """
        Constructor

	    Parameters
	    ----------
	    q
		    Detection level (risk)
		estimator
		    "MLE": maximum likelihood
		    "MOM": method of moments

	    Returns
	    ----------
    	SPOT object
        """
        self.proba = q
        # 将用户指定的风险值q赋值给实例变量proba
        
        self.extreme_quantile = None
        # 初始化极端分位数（阈值）为None，稍后会在初始化步骤中计算
        
        self.data = None
        # 用于存储完整数据流的变量，初始化为None
        
        self.init_data = None
        # 用于存储初始化数据的变量，初始化为None
        
        self.init_threshold = None
        # 初始化阈值，在初始化阶段计算
        
        self.peaks = None
        # 用于存储超过阈值的峰值数据，初始化为None
        
        self.n = 0
        # 记录已观测数据点的总数，初始化为0
        
        self.Nt = 0
        # 记录已观测峰值的数量，初始化为0
        
        if estimator == "MLE":
            self.estimator = self._grimshaw
            # 如果用户选择最大似然估计法，则使用_grimshaw方法进行参数估计
        elif estimator == "MOM":
            self.estimator = self._MOM
            # 如果用户选择矩估计法，则使用_MOM方法进行参数估计
        else:
            raise TypeError("Unsupported Estimator Type!")
            # 如果用户输入了不支持的估计方法，抛出类型错误


    # 打印 SPOT 对象时，会调用这个方法返回一个格式化的字符串，显示算法的当前状态和参数信息
    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s
            # 如果数据未导入，显示提示并提前返回

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold
            # 如果算法已初始化，显示初始阈值

            r = self.n - self.init_data.size
            # 计算已处理的数据流长度（总数据点减去初始化数据点）
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    #fit方法负责接收初始数据并转换为内部格式
    def fit(self, init_data):
        """
        Import initial data to SPOT object

        Args:
            init_data: list, numpy.array or pandas.Series.
                       initial batch to calibrate the algorithm

        Returns:

        """
        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
            # 如果输入是列表，转换为numpy数组
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
            # 如果输入已是numpy数组，直接使用
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
            # 如果输入是pandas Series，提取其值作为numpy数组
        else:
            print('The initial data cannot be set')
            return

    # initialize方法基于这些数据计算初始阈值和极端分位数
    def initialize(self, level=0.98, verbose=True):
        """
        Run the calibration (initialization) step

        Args:
            level: float
                  (default 0.98) Probability associated with the initial threshold t
            verbose: bool
                  (default = True) If True, gives details about the batch initialization
        """

        level = level - floor(level)
        #初始阈值

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        # 对初始化数据进行排序，以便计算经验分位数

        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        # 提取超过初始阈值的数据点，并计算它们与阈值的差值（即峰值）
    
        self.Nt = self.peaks.size
        # if self.Nt == 0:
        #     self.Nt = 1
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self.estimator()
        self.extreme_quantile = self._quantile(g, s)
        #根据估计的参数计算极端分位数（即动态阈值）

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    #数值处理函数，作用是寻找一个函数可能的根
    #jac : 函数的一阶导数； bounds : 给出的搜索根的(min,max)区间；npoints是输出的最大的根的数量
    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function

        Parameters
        ----------
        fun : function
		    scalar function
        jac : function
            first order derivative of the function
        bounds : tuple
            (min,max) interval for the roots search
        npoints : int
            maximum number of roots to output
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval

        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)


    #计算广义帕累托分布（GPD）的对数似然值（todo
    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)

        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    #使用 Grimshaw 方法估计 GPD 的参数（todo
    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick

        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best


    """
    使用矩估计法计算GPD参数
    优点：计算速度快，无需迭代
    缺点：可能不如MLE准确
    """
    def _MOM(self):  # Method of Moments
        Yi = self.peaks
        avg = np.mean(Yi)
        var = np.var(Yi, ddof=1)
        sigma = 0.5 * avg * (1 + avg**2/var)
        gamma = 0.5 * (1 - avg**2/var)
        return gamma, sigma, 0


    #计算极端阈值
    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q

        Parameters
        ----------
        gamma : float
		    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        # 标准GPD分布
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        # 当gamma=0时，退化为指数分布
        else:
            return self.init_threshold - sigma * log(r)
            # return self.init_threshold - sigma / gamma


    #处理单个新数据点，执行异常检测并更新模型
    def run_step(self, data_point):
        """
        Args:
            data_point:

        Returns:

        """

        alarm = 0  # 0: normal point; 1: abnormal point

        # If the observed value exceeds the current threshold (alarm case)
        if data_point > self.extreme_quantile:
            alarm = 1

        # case where the value exceeds the initial threshold but not the alarm ones
        elif data_point > self.init_threshold:
            # we add it in the peaks
            self.peaks = np.append(self.peaks, data_point - self.init_threshold)
            self.Nt += 1
            self.n += 1
            # and we update the thresholds

            g, s, l = self.estimator()
            self.extreme_quantile = self._quantile(g, s)
        else:
            self.n += 1

        th = self.extreme_quantile  # thresholds record

        return th, alarm

