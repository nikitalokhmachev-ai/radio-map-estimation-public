import os
import glob
import torch
import pickle
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_utils import MapDataset

def get_test_dl(test_data_folder, test_batch_size, scaler, building_value, unsampled_value, sampled_value, percentage=0.01):
  test_pickle_path = os.path.join(test_data_folder, f'test_{percentage:.2f}%_*.pickle')
  test_pickles = glob.glob(test_pickle_path)
  test_ds = MapDataset(test_pickles, scaler, building_value=building_value, unsampled_value=unsampled_value, sampled_value=sampled_value)
  test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=1)
  return test_dl

def get_model_error(test_data_folder, test_batch_size, model_path, scaler_path, building_value=None, unsampled_value=None, sampled_value=None):
  with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

  test_dls = []
  percentages =  np.arange(0.02, 0.42, 0.02)
  for percentage in percentages:
    dl = get_test_dl(
        test_data_folder, test_batch_size, scaler, building_value=building_value, unsampled_value=unsampled_value, sampled_value=sampled_value, percentage=percentage)
    test_dls.append(dl)
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


def get_sample_error(results_folder, filename, model_name = None):
  percentages =  np.arange(0.02, 0.42, 0.02)
  with open(os.path.join(results_folder, filename), 'rb') as f:
    error = pickle.load(f)
  if model_name:
    df = pd.DataFrame.from_dict({'Error':error, 'Percentages':percentages, 'Model': model_name})
  else:
    df = pd.DataFrame.from_dict({'Error':error, 'Percentages':percentages, 'Model': os.path.basename(filename).split('.')[0]})
  return df


def visualize_sample_error(results_folder, width=800, height=700, text_size=18, display_names=None, line_styles=None, marker_size=10, consistent_colors=None, y_range=None, x_range=None):
  filenames = glob.glob(os.path.join(results_folder, '*.pickle'))
  dfs = [get_sample_error(results_folder, filename) for filename in filenames]
  dfs = pd.concat(dfs)
  dfs = dfs.drop_duplicates(['Model', 'Percentages'])
  if display_names:
    dfs = dfs.replace({'Model': display_names})
  if consistent_colors:
    dfs['Colors'] = dfs['Model'].copy()
    dfs.replace({'Colors': consistent_colors})
    fig = px.line(dfs, x="Percentages", y="Error", color="Model", line_group="Model", line_shape="spline", render_mode="svg", markers=True, width=width, height=height)
  else:
    fig = px.line(dfs, x="Percentages", y="Error", line_group="Model", line_shape="spline", render_mode="svg", markers=True, width=width, height=height)
  fig.update_layout(shapes=[go.layout.Shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line={'width': 1, 'color': 'black', 'dash':'solid'})])
  fig.update_xaxes(
    ticks="outside",
    tickson="labels")
  fig.update_yaxes(
    ticks="outside",
    tickson="labels")
  fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', yaxis_title='RMSE (dB)')
  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
  fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
  fig.update_layout(legend_font=dict(size=20))
  fig.update_layout(legend_title=None)
  fig.update_layout(legend_borderwidth=1)
  fig.update_layout(legend_bgcolor='hsla(1,1,1,0.5)')
  fig.update_layout(font=dict(size=text_size))
  fig.update_xaxes(title=dict(text='Sampling Rate'))
  fig.update_traces(mode='lines+markers', marker=dict(size=marker_size))
  if y_range:
    fig.update_yaxes(range=y_range)
  if x_range:
    fig.update_xaxes(range=x_range)
  if line_styles:
    for i, model in enumerate(fig.data):
      model.line['dash'] = line_styles[i][0]
      model.marker['symbol'] = line_styles[i][1]
  fig.show()
  return fig


def get_average_error(results_folder):
  filenames = glob.glob(os.path.join(results_folder, '*.pickle'))
  dfs = [get_sample_error(results_folder, filename) for filename in filenames]
  avgs = [(np.sqrt(df['Error'].pow(2).mean()), df.loc[0,'Model']) for df in dfs]
  avg_dfs = pd.DataFrame(avgs, columns=['Avg Error', 'Model'])
  avg_dfs = avg_dfs.drop_duplicates()
  return avg_dfs


def visualize_average_error(avg_dfs, display_names=None, baseline_name='Baseline', category_orders={},
                             width=550, height=500, text_size=19, y_range=None):
  show_dfs = avg_dfs[avg_dfs['Model']!=baseline_name]
  if display_names:
    show_dfs = show_dfs.replace({'Model': display_names})
  baseline = avg_dfs.loc[avg_dfs['Model']==baseline_name, 'Avg Error'].item()
  fig = px.bar(show_dfs, x='Model', y='Avg Error', width=width, height=height, category_orders=category_orders)
  fig.update_layout(shapes=[go.layout.Shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line={'width': 1, 'color': 'black', 'dash':'solid'})])
  fig.update_xaxes(
    ticks="outside",
    tickson="labels")
  fig.update_yaxes(
      ticks="outside",
      tickson="labels")
  fig.add_hline(y=baseline, line_dash='dash', annotation_text='Baseline', annotation_position='bottom right')
  fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', yaxis_title='RMSE (dB)')
  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
  fig.update_layout(font=dict(size=text_size))
  if y_range:
    fig.update_yaxes(range=y_range)
  fig.update_xaxes(title=None)

  fig.show()
  return fig


def visualize_hist(results_folder, display_names=None, baseline_name='Baseline',
                   text_size=19, width=800, height=700, y_range=None):
  filenames = glob.glob(os.path.join(results_folder, '*.pickle'))
  df_percentage = []
  for filename in filenames:
    df = get_sample_error(results_folder, filename, model_name = None)
    model_name = df['Model'].iloc[0]
    if model_name == baseline_name:
      continue
    bin_splits = np.arange(0.1, 0.5, 0.1)
    less_ten, ten_twenty = df.query('Percentages <= 0.1')['Error'].pow(2).mean()**0.5, df.query('0.1 < Percentages <= 0.2')['Error'].pow(2).mean()**0.5
    twenty_thirty, greater_thirty = df.query('0.2 < Percentages <= 0.3')['Error'].pow(2).mean()**0.5, df.query('0.3 < Percentages <= 0.4')['Error'].pow(2).mean()**0.5
    df_model_p = pd.DataFrame.from_dict({'Error':[less_ten, ten_twenty, twenty_thirty, greater_thirty],
                                        'Category':['1%-10%', '11%-20%', '21%-30%', '31%-40%'],
                                        'Model Name':[model_name, model_name, model_name, model_name]})
    df_percentage.append(df_model_p)

  df_percentage = pd.concat(df_percentage)
  df_percentage = df_percentage.drop_duplicates()
  if display_names:
    df_percentage = df_percentage.replace({'Model Name': display_names})

  order = {'Category':['1%-10%', '11%-20%', '21%-30%', '31%-40%']}
  df_top_percent = df_percentage[df_percentage['Category'] == '31%-40%']
  top_percent_order = df_top_percent.sort_values('Error', ascending=False)['Model Name'].tolist()
  order['Model Name'] = top_percent_order

  fig = px.bar(df_percentage, x="Category", y="Error", color="Model Name", category_orders=order, barmode="group")
  fig.update_layout(shapes=[go.layout.Shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line={'width': 1, 'color': 'black', 'dash':'solid'})])
  fig.update_xaxes(
    ticks="outside",
    tickson="labels")
  fig.update_yaxes(
    ticks="outside",
    tickson="labels")
  fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)', yaxis_title='RMSE (dB)')
  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
  fig.update_layout(width=width, height=height)
  fig.update_layout(font=dict(size=text_size))
  fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
  fig.update_layout(legend_font=dict(size=20))
  fig.update_layout(legend_title=None)
  fig.update_layout(legend_borderwidth=1)
  fig.update_layout(legend_bgcolor='hsla(1,1,1,0.5)')
  fig.update_xaxes(title=dict(text='Sampling Rate'))
  if y_range:
    fig.update_yaxes(range=y_range)
  fig.show()
  return fig