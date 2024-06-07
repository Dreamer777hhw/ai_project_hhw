from torch import nn
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
import torch

class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)

        # 对抗训练
        epsilon = 0.1
        x_adv = x + epsilon * torch.sign(torch.autograd.grad(loss, x, retain_graph=True)[0])
        x_adv = torch.clamp(x_adv, 0, 1)
        logits_adv = self(x_adv)
        loss_adv = self.loss_fn(logits_adv, y)
        loss = (loss + loss_adv) / 2
        self.log('train_loss_adv', loss_adv)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        self.log('test_acc', acc)
        return acc
