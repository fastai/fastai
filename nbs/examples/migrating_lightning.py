# The fastai DataLoader is a drop-in replacement for Pytorch's;
#   no code changes are required other than changing the import line
from fastai.data.load import DataLoader
import os,torch
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from lightning import LightningModule

class LitModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x): return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss}

    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

    def validation_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(avg_loss)
        return {'val_loss': avg_loss}

    def val_dataloader(self):
        # TODO: do a real train/val split
        dataset = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, num_workers=4)
        return loader

