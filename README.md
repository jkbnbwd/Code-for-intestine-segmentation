# Code-for-intestine-segmentation
项目概述
本项目包含多个 Python 脚本，主要用于医学图像分析和处理，涉及数据处理、模型训练、评估指标计算、可视化等多个方面。具体功能包括：
数据路径列表文件的生成
3D 医学图像的骨架提取和处理
模型训练，如使用交叉伪监督进行 3D 模型训练
评估指标的计算，如 Dice 系数、召回率和精确率
数据可视化，如绘制箱线图和柱状图
本项目上传了主要的网络结构相关代码，请自行编写数据加载部分。
代码结构
coe_for_github/writetxt.py：生成数据路径列表文件，将数据划分为训练集和测试集。
coe_for_github/dataloaders/worddata.py：生成单词数据的路径列表文件，将数据划分为训练集和验证集。
coe_for_github/config.py：获取配置信息。
coe_for_github/train_cross_pseudo_supervision_3D.py：使用交叉伪监督进行 3D 模型训练。
coe_for_github/test.py：计算评估指标，如 Dice 系数、召回率和精确率。
coe_for_github/skele3D.py：进行 3D 医学图像的骨架提取和处理。
coe_for_github/doublebarchart.py：绘制双柱状图，展示不同方法在不同标注数量下的 Dice 分数。
coe_for_github/figurewithSD.py：绘制带有标准差的柱状图，展示不同条件下的分割结果。
依赖库
os
sys
torch
numpy
matplotlib
pandas
SimpleITK
nibabel
skimage
cc3d
sknw
tensorboardX
tqdm
使用方法
数据准备：确保数据路径正确，并根据需要修改代码中的数据路径。
生成数据路径列表文件：运行coe_for_github/writetxt.py和coe_for_github/dataloaders/worddata.py。
模型训练：运行coe_for_github/train_cross_pseudo_supervision_3D.py。
评估指标计算：运行coe_for_github/test.py。
骨架提取和处理：运行coe_for_github/skele3D.py。
数据可视化：运行coe_for_github/doublebarchart.py和coe_for_github/figurewithSD.py。
注意事项
请确保所有依赖库已正确安装。
根据实际情况修改代码中的参数，如数据路径、模型类型、训练轮数等。
