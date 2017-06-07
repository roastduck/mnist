# MNIST Experiment

运行方法

## 需求

1. Python >= 3.5
2. Tensorflow >= 1.1 （低于1.1将导致CNN模型的预处理部分无法运行）

## 用已训练网络生成结果

首先切换到`src`文件夹，然后使用相应模型运行：

1. 使用CNN模型（效果最佳），运行`python3 cnn.py run`，或
2. 使用MLP模型，运行`python3 mlp.py run`。

结果将保存在`data/submission.csv`

## 训练网络（非必须）

首先切换到`src`文件夹，建立`src/experiments/<训练ID>/<运行ID>`目录，创建`src/experiments/<训练ID>/<运行ID>/conf.json`配置文件，设定"startEpisode"、"endEpisode"表示起止训练轮数，"learningRate"表示学习率、"fromCheckpoint"表示从<运行ID-1>目录的上一次运行中的哪一个保存点开始继续训练。

以CNN为例执行`python3 cnn.py train <训练ID> <运行ID>`训练，每100轮训练会保存一个保存点到相应文件夹，执行`python3 cnn.py test <训练ID> <运行ID> <保存点ID>`以某个保存点的网络生成结果。MLP类似。
