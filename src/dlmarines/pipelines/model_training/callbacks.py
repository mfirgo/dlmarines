import pytorch_lightning as pl
import wandb

class LogPredictionSamplesCallback(pl.Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            n = 16
            x, y = batch
            y_preds = outputs[1].argmax(-1)
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' 
                for y_i, y_pred in zip(y[:n], y_preds[:n])]
            
            
            # Option 1: log images with `WandbLogger.log_image`
            trainer.logger.log_image(
                key='sample_images', 
                images=images, 
                caption=captions
            )


            # Option 2: log images and predictions as a W&B Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], y_preds[:n]))]
            trainer.logger.log_table(
                key='sample_table',
                columns=columns,
                data=data
            )

