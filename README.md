# rl_arm
Using RL to train an arm in a simulated environment

Reacher问题在经典的path planning上面已经被研究透了，但是以RL方式解决目前应该还处于起步。
很多RL训练的Reacher的action space都作用在手臂的angular velocity上面，但是实际情况不一定能够实现，因为比如速度从0增加到20之间是需要时间的。
这个实验就是将action作用在力矩上来贴合实际，因为力矩的输出在连续的两个时间点上不需要连续，但问题是在这样的情况下会出现一些非线性的情况，类似弹簧的overdamped和underdamped。
目前的目标是训练reacher从起始点移动到终点可以做到critical damp，虽然damping的问题算是解决了，但是很多方面还在实验阶段，比如在训练方面还需要些超参的调整。

### 模型概括
模型主要参考deterministic policy gradient，但个人做了较大的改动。例如移除base value并不影响policy训练的稳定性。
在Model base和Model free上选择了Model base，这样可以大大减少sample量。
另外训练的时候分别用off-policy去bootstrap，然后用on-policy去迭代优化。

训练目标是最大化reward：-Dist1(Pred(s, π(s)) - Target)

其中
* π:: s -> a (policy)
* Pred:: s, a -> s (预测模型， Advantage function）, 
* Dist1 （曼哈顿距离）


### Demo
如果有时间可跑一下test_demo_0.py和test_demo.py，分别对比改良算法前后policy，了但是需要安装以下的库：
* pytorch
* numpy
* matplotlib
* opencv 
* pymunk 轻量级2D物理引擎，安装小运行快，整个simulation主要就是靠它
```
conda install -c conda-forge pymunk
```    
  
### 备注
一般production的时候还会用argparse和一个Config的类把超参数理好，但是现在训练脚本还有很多的heuristic在里面，就是时机还没到。

另外如果可以翻墙到YouTube的话可以看一下参数优化的动态loss(reward) plot:
* [GD with Momentum](https://www.youtube.com/watch?v=ZT5cK5BZjaI)

内在模型训练的一些数据：

```
    0 step: loss 0.0045, std 0.003, on average 0.08 deviation, max movement 3.6 (8000 epochs 1200s)
    1 step: loss 0.0065, std 0.0038, on average 0.09 deviation, max movement 6.4 (8000 epochs 1400s)
    1 step: loss 0.0040, std 0.001, on average 0.06 deviation, max movement 3.2 (8000 epochs 1400s)
    2 step: loss 0.0080, std 0.003, on average 0.09 deviation, max movement 7.8 (7000 epochs 1200s)
    3 step: loss 0.009, std 0.002, on average 0.1 deviation, max movement 7.8 (7000 epochs 1200s)
    4 step: loss 0.012, std 0.003, on average 0.12 deviation, max movement 8.2 (7000 epochs 1200s)
    5 step: loss 0.018, std 0.004, on average 0.14 deviation, max movement 9.8 (7000 epochs 1200s)
    6 step: loss 0.020, std 0.005, on average 0.15 deviation, max movement 10.7 (7000 epochs 1300s)
    7 step: loss 0.030, std 0.007, on average 0.17 deviation, max movement 11.7 (7000 epochs 1300s)
    8 step: loss 0.035, std 0.007, on average 0.18 deviation, max movement 11.7 (7000 epochs 1300s)
    8 step: loss 0.013, std 0.0023, on average 0.11 deviation, max movement 9.7 (200 epochs 55s from 8 step ratio3 pre-train) 
    16 step: loss 0.150, std 0.020, on average 0.4 deviation, max movement 21.6 (7000 epochs 3200s) 
    24 step: loss 0.600, std 0.13, on average 0.7 deviation, max movement 32 (7000 epochs 3200s) added on 30/11/2019
    50 step: loss 0.500, std 0.12, on average 0.7 deviation, max movement 32 (5000 epochs 4800s）
    100 step: loss 7.500, std 0.72, on average 2.8 deviation, max movement 71 (7000 epochs retrained from scratch 5800s) 
```


