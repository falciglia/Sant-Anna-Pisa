import torch
import torch.nn as nn
from torchmetrics.regression import WeightedMeanAbsolutePercentageError
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import *

from src.framework.transformer import *

'''#############################################################'''
'''################# The Proposed Architecture #################'''
'''#############################################################'''

class LAURA_Architecture(pl.LightningModule):

  def __init__(self,
               transformer_hyperparameters):
      super(LAURA_Architecture, self).__init__()
      
      # ---- Transformer Hypeparameters --------------------------------------------------------
      self.input_size = transformer_hyperparameters['input_size']
      self.num_heads = transformer_hyperparameters['num_heads']
      self.hidden_size = transformer_hyperparameters['hidden_size']
      self.num_bins = transformer_hyperparameters['num_bins']
      self.num_layers_encoder = transformer_hyperparameters['num_layers_encoder']
      self.max_sequence_length = transformer_hyperparameters['max_sequence_length']
      self.num_layers_decoder = transformer_hyperparameters['num_layers_decoder']
      # ---- Special Tokens ----------------------------------------------------------------
      #self.start_token_index = -1
      #self.end_token_index = -2

      self.transformer = TransformerModel(self.input_size, self.num_heads, self.hidden_size, self.num_bins, self.num_layers_encoder, self.max_sequence_length, self.num_layers_decoder)
      self.criterion = WeightedMeanAbsolutePercentageError()

  def forward(self, source_sequence):
      # Encode the source sequence
      output = self.transformer(source_sequence)
      return output

  def sample_next_token(self, output):
      # Greedy Sampling
      _, next_token = output.max(dim=1)
      return next_token

  def MAPELoss(self, output, target):
      # MAPE loss
      return torch.mean(torch.abs((target - output) / target))

  def configure_optimizers(self):
      LAURA_optimizer = torch.optim.RMSprop(list(self.transformer.parameters()), lr=1e-4)
      return LAURA_optimizer

  def training_step(self, train_batch, batch_idx):
      LAURA_optimizer = self.optimizers()
      torch.set_grad_enabled(True)
      #loss_criterion = nn.MSELoss()

      beta_distribution_data, beta_distribution_dayAfter = train_batch
      beta_distribution_data, beta_distribution_dayAfter = beta_distribution_data.to(device), beta_distribution_dayAfter.to(device)

      # ----- START TRAINING ---------
      output = self(beta_distribution_data)

      # Metrics
      B = output.shape[0]
      ### R-Squared
      r_sqr_list = []
      for b in range(B):
          r_sqr = r2_score(beta_distribution_dayAfter.detach()[b].cpu().numpy(), output.detach()[b].cpu().numpy())
          r_sqr_list.append(r_sqr)
      r_sqr_mean = sum(r_sqr_list)/B
      ### Mean Absolute Percentage Error (MAPE)
      mape_list = []
      for b in range(B):
          mape = mean_absolute_percentage_error(beta_distribution_dayAfter.detach()[b].cpu().numpy(), output.detach()[b].cpu().numpy())
          mape_list.append(mape)
      mape_mean = sum(mape_list)/B
      ### Root Mean Squared Error (RMSE)
      rmse_list = []
      for b in range(B):
          rmse = np.sqrt(mean_squared_error(beta_distribution_dayAfter.detach()[b].cpu().numpy(), output.detach()[b].cpu().numpy()))
          rmse_list.append(rmse)
      rmse_mean = sum(rmse_list)/B

      #MSE = loss_criterion(beta_distribution_dayAfter, output)
      MSE = self.criterion(beta_distribution_dayAfter, output)

      LAURA_loss = MSE
      LAURA_loss.backward(retain_graph=True)
      self.log('train_loss', LAURA_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
      self.log('train R_2', r_sqr_mean, on_epoch=True, prog_bar=True, logger=True)
      self.log('train MAPE', mape_mean, on_epoch=True, prog_bar=True, logger=True)
      self.log('train RMSE', rmse_mean, on_epoch=True, prog_bar=True, logger=True)

      return LAURA_loss

  def validation_step(self, val_batch, batch_idx):
      #loss_criterion = nn.MSELoss()

      beta_distribution_data, beta_distribution_dayAfter = val_batch
      beta_distribution_data, beta_distribution_dayAfter = beta_distribution_data.to(device), beta_distribution_dayAfter.to(device)

      output = self(beta_distribution_data)
      #print(output.size())

      # Metrics
      B = output.shape[0]
      ### R-Squared
      r_sqr_list = []
      for b in range(B):
          r_sqr = r2_score(beta_distribution_dayAfter.detach()[b].cpu().numpy(), output.detach()[b].cpu().numpy())
          r_sqr_list.append(r_sqr)
      r_sqr_mean = sum(r_sqr_list)/B
      ### Mean Absolute Percentage Error (MAPE)
      mape_list = []
      for b in range(B):
          mape = mean_absolute_percentage_error(beta_distribution_dayAfter.detach()[b].cpu().numpy(), output.detach()[b].cpu().numpy())
          mape_list.append(mape)
      mape_mean = sum(mape_list)/B
      ### Root Mean Squared Error (RMSE)
      rmse_list = []
      for b in range(B):
          rmse = np.sqrt(mean_squared_error(beta_distribution_dayAfter.detach()[b].cpu().numpy(), output.detach()[b].cpu().numpy()))
          rmse_list.append(rmse)
      rmse_mean = sum(rmse_list)/B

      #print(beta_distribution_dayAfter.size())
      #print(output.size())
      MSE = self.criterion(beta_distribution_dayAfter, output)

      LAURA_loss = MSE
      self.log('val_loss', LAURA_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
      self.log('val R_2', r_sqr_mean, on_epoch=True, prog_bar=True, logger=True)
      self.log('val MAPE', mape_mean, on_epoch=True, prog_bar=True, logger=True)
      self.log('val RMSE', rmse_mean, on_epoch=True, prog_bar=True, logger=True)

  def predict_step(self, pred_batch, batch_idx):
      self.transformer.eval()
      
      beta_distribution_data, beta_distribution_dayAfter, patient_id, segment_id = pred_batch
      beta_distribution_data, beta_distribution_dayAfter, patient_id, segment_id = beta_distribution_data.to(device), beta_distribution_dayAfter.to(device), patient_id.to(device), segment_id.to(device)

      output = self(beta_distribution_data)
      return output, beta_distribution_dayAfter, beta_distribution_data, patient_id, segment_id