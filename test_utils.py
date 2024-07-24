import os
import glob
import torch
import pickle
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from data_utils import MapDataset

def get_test_dl(test_data_folder, test_batch_size, scaler, building_value, unsampled_value, sampled_value, percentage=0.01):
  test_pickle_path = os.path.join(test_data_folder, f'test_{percentage:.2f}%_*.pickle')
  test_pickles = glob.glob(test_pickle_path)
  test_ds = MapDataset(test_pickles, scaler, building_value=building_value, unsampled_value=unsampled_value, sampled_value=sampled_value)
  test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=1)
  return test_dl

def get_model_error(test_data_folder, model_path, scaler_path, building_value=None, unsampled_value=None, sampled_value=None):
  with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

  test_dls = []
  percentages =  np.arange(0.02, 0.42, 0.02)
  for percentage in percentages:
    test_dls.append(get_test_dl(
        test_data_folder, scaler, building_value=building_value, unsampled_value=unsampled_value, sampled_value=sampled_value, percentage=percentage))

  model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  model.eval()

  error = [model.evaluate(test_dl, scaler) for test_dl in test_dls]
  return error

def get_error_df(results_folder, filename, model_name = None):
  percentages =  np.arange(0.02, 0.42, 0.02)
  with open(os.path.join(results_folder, filename), 'rb') as f:
    error = pickle.load(f)

  if model_name:
    df = pd.DataFrame.from_dict({'error':error, 'percentages':percentages, 'model': model_name})
  else:
    df = pd.DataFrame.from_dict({'error':error, 'percentages':percentages, 'model': os.path.basename(filename).split('.')[0]})

  return df


def visualize_results(results_folder):
  filenames = glob.glob(os.path.join(results_folder, '*.pickle'))
  dfs = [get_error_df(results_folder, filename) for filename in filenames]
  dfs = pd.concat(dfs)
  fig = px.line(dfs, x="percentages", y="error", color="model", line_group="model", line_shape="spline", render_mode="svg", markers=True, width=1200, height=600)
  fig.show()