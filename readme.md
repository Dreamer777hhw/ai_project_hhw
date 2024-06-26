# Violence Classifier 接口调用实例使用说明

本文档将介绍如何使用 `ViolenceClass` 对暴力图像进行分类。

该接口基于 PyTorch 和 torchvision 构建，需要使用我们预训练过的模型进行操作。

## 先决条件

在使用该接口之前，请确保已安装以下依赖项：

- `torch`
- `torchvision`
- `Pillow`

可以使用以下命令安装它们：

```bash
pip install torch torchvision Pillow
```

##  `ViolenceClass`类

`ViolenceClass` 用于加载预训练的暴力分类模型并使用它对图像进行分类，是我们项目中的主要接口函数。以下初始化，方法: `misc`，方法: `classify`，该类中的三大组成部分

### 初始化

```python
ViolenceClass(ckpt_path, device='cuda:0')
```

#### 参数:

- `ckpt_path` : 模型检查点文件的路径。
- `device` : 运行模型的设备。默认为 'cuda:0'。

#### 示例:

```python
from violence_class import ViolenceClass

ckpt_path = 'path/to/your/resnet50_pretrain_test/version_0/checkpoints/resnet50_pretrain_test-epoch=xx-val_loss=xx.ckpt'
classifier = ViolenceClass(ckpt_path, device='cuda:0')
```

### 方法: `misc`

该方法预处理图像路径列表并将其转换为适合模型输入的张量。

```python
misc(img_paths)
```

#### 参数:

- `img_paths`: 图像文件路径列表。

#### 返回:

- `torch.Tensor`: 形状为 `(batch_size, 3, 224, 224)` 的张量，包含预处理后的图像。

#### 示例:

```python
img_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
preprocessed_imgs = classifier.misc(img_paths)
```

### 方法: `classify`

该方法接收预处理后的图像张量并返回分类结果。

```python
classify(imgs)
```

#### 参数:

- `imgs` : 形状为 `(batch_size, 3, 224, 224)` 的预处理图像张量。

#### 返回:

- `list`: 每个输入图像的分类结果列表。

#### 示例:

```python
results = classifier.classify(preprocessed_imgs)
print(results)  # 输出: [0, 1, 0] 其中 0 和 1 是分类标签，0是无害，1是有害
```

### 一个完整的示例

```python
from violence_class import ViolenceClass

# 使用模型检查点初始化分类器
ckpt_path = 'path/to/your/resnet50_pretrain_test/version_0/checkpoints/resnet50_pretrain_test-epoch=xx-val_loss=xx.ckpt'
classifier = ViolenceClass(ckpt_path, device='cuda:0')

# 图像文件路径列表
img_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# 预处理图像
preprocessed_imgs = classifier.misc(img_paths)

# 进行分类
results = classifier.classify(preprocessed_imgs)

# 输出分类结果
print(results)  # 输出: [0, 1, 0] 其中 0 和 1 是分类标签，0是无害，1是有害
```
