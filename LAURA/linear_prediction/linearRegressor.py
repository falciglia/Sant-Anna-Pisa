import pytorch_lightning as pl
import numpy as np

from linear_prediction.linearHyperparams_commonstructure import *

class linearRegressor(pl.LightningModule):

  def __init__(self):
      super(linearRegressor, self).__init__()


  def forward(self, source_sequence):
      num_days = len(source_sequence)
      num_bins = len(source_sequence[0])

      # Initialize arrays for coefficients a and b
      a = np.zeros(num_bins)
      b = np.zeros(num_bins)

      if num_days == 2:
          # If only two arrays are provided, use the existing logic
          day1 = source_sequence[-2].numpy()
          day2 = source_sequence[-1].numpy()

          for i in range(num_bins):
              X = np.vstack([day1[i], np.ones(1)]).T  # Design matrix for bin i
              y = np.array([day2[i]])
              if day1[i] != 0:
                  coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                  a[i], b[i] = coefficients
              else:
                  a[i] = 0
                  b[i] = day2[i]  # If day1[i] is 0, the intercept is day2[i]
      else:
          # For more than two arrays, use linear interpolation
          X = np.array([day.numpy() for day in source_sequence[:-1]]).T  # Design matrix
          Y = np.array(source_sequence[1:]).T  # Target matrix

          for i in range(num_bins):
              if np.any(X[i, :] != 0):  # Check if there are non-zero elements in the row
                  # Add a column of ones to X for the intercept term
                  Xi = np.hstack([X[i, :].reshape(-1, 1), np.ones((num_days - 1, 1))])
                  yi = Y[i, :]
                  coefficients, residuals, rank, s = np.linalg.lstsq(Xi, yi, rcond=None)
                  a[i], b[i] = coefficients
              else:
                  a[i] = 0
                  b[i] = Y[i, -1]  # If all elements in X[i, :] are 0, use the last target value

      # Use the fitted model to make predictions
      predictions = [source_sequence[-2].numpy(), source_sequence[-1].numpy()]
      for _ in range(6):
          next_day = a * predictions[-1] + b
          next_day = (next_day - np.min(next_day)) / (np.max(next_day) - np.min(next_day))
          predictions.append(next_day)

      return predictions

  def predict_step(self, pred_batch, batch_idx):
      beta_distribution_data, beta_distribution_dayAfter, patient_id, segment_id = pred_batch
      beta_distribution_data, beta_distribution_dayAfter, patient_id, segment_id = beta_distribution_data.to(device), beta_distribution_dayAfter.to(device), patient_id.to(device), segment_id.to(device)

      output = self(beta_distribution_data.squeeze())
      return output, beta_distribution_dayAfter, beta_distribution_data, patient_id, segment_id