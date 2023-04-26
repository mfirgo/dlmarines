import pytorch_lightning as pl
import numpy as np

class LogPredictionSamplesCallback(pl.Callback):
    def __init__(self, id_to_class):
        super().__init__()
        self.id_to_class = id_to_class
        self.accuracies = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Called when the validation batch ends."""
        x, y = batch
        y_preds = outputs[1].argmax(-1)
        if batch_idx == 0:
            n = 16

            probabilities = outputs[1][np.arange(len(y)), y]
            
            images = [img for img in x[:n]]
            captions = [
                f'Ground Truth: {self.id_to_class[y_i.item()]} Prediction: {self.id_to_class[y_pred.item()]} Certainty: {prob.item():.2%}' 
                for y_i, y_pred, prob in zip(y[:n], y_preds[:n], probabilities[:n])]
            
            
            trainer.logger.log_image(
                key='sample_images', 
                images=images, 
                caption=captions
            )
        self.accuracies.append((y_preds==y).float().mean().item())
            

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        pl_module.log('val_acc', np.mean(self.accuracies))
        self.accuracies = []
        

