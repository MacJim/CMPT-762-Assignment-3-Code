import csv

import matplotlib.pyplot as plt

import segmentation_epoch_logger


log_list = segmentation_epoch_logger.read_epoch_details_from_file("seg_output/loss_log.csv")
epochs = [entry[0] for entry in log_list]
train_losses = [entry[1] for entry in log_list]

fig = plt.figure(figsize=(15, 8))
plt.plot(epochs, train_losses, "k-")
plt.ylabel("Total Training Loss")
plt.xlabel("Epoch")
plt.grid(True, axis="y")
plt.savefig("/tmp/seg_train_loss.png", dpi=300)
plt.close(fig)
