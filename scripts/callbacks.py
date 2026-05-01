# from lightning.pytorch.callbacks import Callback

# class ProgressBarLR(Callback):
#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         lr = trainer.optimizers[0].param_groups[0]['lr']
#         # Update the progress bar dictionary
#         trainer.progress_bar_callback.train_progress_bar.set_postfix({"lr": f"{lr:.8e}"})