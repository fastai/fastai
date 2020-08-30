# The fastai DataLoader is a drop-in replacement for Pytorch's;
#   no code changes are required other than changing the import line
from fastai.data.load import DataLoader
import os,torch
from torch.nn import functional as F
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics

model = torch.nn.Linear(28 * 28, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
}

class CustomRunner(dl.Runner):
    def predict_batch(self, batch): return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

    def _handle_batch(self, batch):
        x, y = batch
        y_hat = self.model(x.view(x.size(0), -1))

        loss = F.cross_entropy(y_hat, y)
        accuracy01, accuracy03 = metrics.accuracy(y_hat, y, topk=(1, 3))
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

