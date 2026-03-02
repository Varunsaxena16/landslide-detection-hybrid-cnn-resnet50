import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

class F1ScoreCallback(tf.keras.callbacks.Callback):

    def __init__(self, val_data, y_val_true, verbose=1):
        super().__init__()
        self.val_data = val_data
        self.y_val_true = y_val_true
        self.best_f1 = 0
        self.best_thresh = 0.5
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):

        y_pred_probs = self.model.predict(self.val_data, verbose=0).flatten()

        best_f1 = 0
        best_thresh = 0.5

        for thresh in np.arange(0.1, 0.91, 0.01):
            preds = (y_pred_probs > thresh).astype(int)
            f1 = f1_score(self.y_val_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        self.best_f1 = best_f1
        self.best_thresh = best_thresh

        precision = precision_score(
            self.y_val_true,
            (y_pred_probs > best_thresh).astype(int)
        )

        recall = recall_score(
            self.y_val_true,
            (y_pred_probs > best_thresh).astype(int)
        )

        if self.verbose:
            print(
                f"\nEpoch {epoch+1}: "
                f"Best Thresh={best_thresh:.2f}, "
                f"F1={best_f1:.4f}, "
                f"Precision={precision:.4f}, "
                f"Recall={recall:.4f}"
            )

# Some built-in Keras callbacks used: Checkpoint, EarlyStopping, ReduceLR.
