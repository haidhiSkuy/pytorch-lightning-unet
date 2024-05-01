import torch 
import numpy as np 
import lightning as L
from callback import get_callbacks
from model.lightning_module import LitUnet
from lightning.pytorch.loggers import WandbLogger
from dataset.lightning_dataset import BrainDataModule


model = LitUnet()
data = BrainDataModule("dataset/brain", split_size=0.8, batch_size=16)
wandb_logger = WandbLogger(project="Unet", log_model=True)
callbacks = get_callbacks()

# Trainer
trainer = L.Trainer(max_epochs=15, logger=wandb_logger, callbacks=callbacks)
trainer.fit(model, data)
