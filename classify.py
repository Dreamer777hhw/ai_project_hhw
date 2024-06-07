import torch
from PIL import Image
import torchvision.transforms as transforms
from model import ViolenceClassifier
import os
from sklearn.metrics import accuracy_score, f1_score

class ViolenceClass:
    def __init__(self, ckpt_path, device='cuda:0'):
        """
        初始化函数，加载模型和设置参数
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
        self.model.to(self.device)
        self.model.eval()  # 设置模型为评估模式
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小为224x224
            transforms.ToTensor(),  # 将图像转换为Tensor，并归一化到0-1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
        ])

    def misc(self, img_paths: list) -> torch.Tensor:
        """
        图像预处理函数，将图像路径列表转换为形状为 (batch_size, 3, 224, 224) 的张量

        参数:
            img_paths (list): 图像路径列表

        返回:
            torch.Tensor: 预处理后的图像张量，形状为 (batch_size, channels, height, width)
        """
        images = [self.preprocess(Image.open(img_path).convert('RGB')) for img_path in img_paths]
        return torch.stack(images)  # 将列表转换为形状为 (batch_size, channels, height, width) 的张量

    def classify(self, imgs: torch.Tensor) -> list:
        """
        图像分类函数

        参数:
            imgs (torch.Tensor): 输入的图像张量，形状为 (batch_size, channels, height, width)

        返回:
            list: 分类结果，形状为 (batch_size,)
        """
        imgs = imgs.to(self.device)
        with torch.no_grad():  # 关闭梯度计算
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().tolist()

# 使用示例

# 假设你的测试图像路径列表是/home/huhw/violence_detector/test_image/中的所有文件
test_image_paths = [f"/root/autodl-tmp/test_image/{i}" for i in os.listdir("/root/autodl-tmp/test_image")]

# 实例化并使用 ViolenceClass 进行分类
ckpt_path = "/root/autodl-tmp/resnet50_new/version_2/checkpoints/resnet50_new-epoch=31-val_loss=0.12.ckpt"
# ckpt_path = "/root/autodl-tmp/train_logs/resnet50_new_dataset/version_2/checkpoints/resnet50_new_dataset-epoch=24-val_loss=0.05.ckpt"
classifier = ViolenceClass(ckpt_path)

# 调用 misc 方法进行预处理
test_images = classifier.misc(test_image_paths)

# 调用 classify 方法进行分类
predictions = classifier.classify(test_images)

# 生成实际标签
true_labels = [0 if os.path.basename(path).startswith('0_') else 1 for path in test_image_paths]

# 计算准确率和F1指标
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# 打印每个图像的预测结果
results = ['non-violence', 'violence']
for img_path, pred in zip(test_image_paths, predictions):
    print(f"Image: {img_path}, Prediction: {results[pred]}")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")