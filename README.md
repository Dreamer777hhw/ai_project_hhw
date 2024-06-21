# ai_project_9

本项目按照要求，包括以下项目：

1. --9-README.md，即本文件，系接口调用实例说明
2. --9-report.pdf，即本项目的完整实验报告
   **3. --9-classify.py，接口类文件**
3. --9-requirements.txt，本项目的实验环境配置文件
4. --9-aie_generate.py，是用于生成AEI，即对抗样本推理相关的内容
5. --9-train，用于训练的相关文件的文件夹，包含以下内容
   1. --9-dataset.py，数据集定义文件
   2. --9-model.py，模型定义文件
   3. --9-train.py，训练主要文件
6. --9-CNN.py 用于训练CNN模型的一个样例，注意，该文件也使用了train文件夹中的dataset.py数据集模块
7. --9-simple_cnn.pth，是我们使用上述CNN.py训练出来的一个基于CNN的暴力图像识别模型
8. --9-readme.md，接口类文件的使用说明

我们的模型文件较大，因此存在网盘中，地址如下：

文件大小: 246.1 MB

分享内容: resnet50_pretrain_test.zip 

链接地址:https://jbox.sjtu.edu.cn/l/w1gVhO   

来自于: 胡皓文  

该压缩包中的文件路径如下“resnet50_pretrain_test.zip\resnet50_pretrain_test\version_2\checkpoints\”，文件名是“resnet50_new-epoch=31-val_loss=0.12.ckpt”

## 使用说明

### 接口函数相关

请您先从网盘中下载我们的模型

在使用我们的接口之前，请确保已安装以下依赖项：

- `torch`
- `torchvision`
- `Pillow`

可以使用以下命令安装它们：

```bash
pip install torch torchvision Pillow
```

此外，我们的程序需要同目录下有model.py文件，在最后注释部分是我们用于测试的部分，与接口函数无关。

### 对抗样本图像相关

在本目录下的aie_generate.py是用于生成对抗样本图像的程序，该程序的输入是图片，能够输出原图像加上图像噪声后的图像。在本实验中主要用于对同源的数据集进行对抗样本的生成。

### Baseline（CNN）相关

在本目录下有CNN.py文件，是用于训练CNN模型的文件，其训练结果模型在如下交大网盘链接中

文件大小: 178.5 MB

分享内容: simple_cnn.zip 

链接地址:https://jbox.sjtu.edu.cn/l/41lodL   

来自于: 胡皓文  

