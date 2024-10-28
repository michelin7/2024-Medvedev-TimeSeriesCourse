import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions: dict, cutoff, house_idx: list) -> (dict, dict):
    """
    Split time series into two parts: Head and Tail

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses

    Returns
    --------
    heads: heads of time series
    tails: tails of time series
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails


def meter_swapping_detection(heads: dict, tails: dict, house_idx: dict, m: int) -> dict:
    """
    Find the swapped time series pair

    Parameters
    ---------
    heads: heads of time series
    tails: tails of time series
    house_idx: indices of houses
    m: subsequence length

    Returns
    --------
    min_score: time series pair with minimum swap-score
    """

    eps = 0.001

    min_score = {'score': float('inf'), 'pair': (None, None)}

    for i in house_idx:
        for j in house_idx:
            if i != j:  # Избегаем самосравнения
                # Извлекаем ряды для текущих домов
                ts1 = heads.get(f'H_{i}')
                ts2 = tails.get(f'T_{j}')
                
                if ts1 is not None and ts2 is not None:
                    # Вычисляем матричный профиль между ts1 и ts2
                    result = compute_mp(ts1=ts1, m=m, ts2=ts2)
                    swap_score = np.min(result['mp'])  # Минимум из матричного профиля
                    
                    # Проверяем, меньше ли текущий swap_score минимального найденного значения
                    if swap_score < min_score['score'] - eps:
                        min_score['score'] = swap_score
                        min_score['pair'] = (i, j)
    
    return min_score


def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="vscode")
