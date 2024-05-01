import lightning as L 
from .torch_model import get_model 
from .losses_metrics import * 

class LitUnet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        prediction = self.model(x).squeeze(1)

        loss = criterion(prediction, y)
        self.log("train_loss", loss)

        iou = iou_coef(prediction, y)  
        self.log("train_iou", iou)
        
        dice = dice_coef(prediction, y) 
        self.log("train_dice", dice)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self.model(x).squeeze(1)

        loss = criterion(prediction, y)
        self.log("val_loss", loss)

        iou = iou_coef(prediction, y)  
        self.log("val_iou", iou)
        
        dice = dice_coef(prediction, y) 
        self.log("val_dice", dice)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer