import torch
from torch.nn import functional as F
from models.Flowestimator import FlowEstimator
import pytorch_lightning as pl
from utils import warper
from dataloader.sintelloader import SintelLoader
from utils.photometricloss import photometricloss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FlowTrainer(pl.LightningModule):
    def __init__(self):
        super(FlowTrainer, self).__init__()
        # not the best model...
        self.deepflow = FlowEstimator()

    def forward(self, x):
        return self.deepflow(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        flow, occlusion = self.forward(batch['displacement'])
        loss = photometricloss(batch, flow, occlusion)
        tensorboard_logs = {'photometric_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        flow, occlusion = self.forward(batch['displacement'])
        loss = photometricloss(batch, flow, occlusion)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.09)
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer,],[scheduler,]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return SintelLoader(batch_size=10).load()

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return SintelLoader(batch_size=10).load()

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return SintelLoader(batch_size=10).load()
