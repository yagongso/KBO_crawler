# coding: utf-8

import sys, csv, json, collections, datetime, os, numbers, importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.random
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib import font_manager as fm, rc
from IPython.display import HTML
from IPython.display import display
import pandas as pd
import seaborn as sns
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
import ipywidgets as widgets
from IPython.display import clear_output
import scipy as sp
from scipy import stats
import pylab as pl

if importlib.util.find_spec('pygam') is not None:
    from pygam import LogisticGAM

Results = Enum('Results', '볼 스트라이크 헛스윙 파울 타격 번트파울 번트헛스윙')
Stuffs = Enum('Stuffs', '직구 슬라이더 포크 체인지업 커브 투심 싱커 커터 너클볼')
Colors = {'볼': '#3245ef', '스트라이크': '#ef2926', '헛스윙':'#1a1b1c', '파울':'#edf72c', '타격':'#8348d1', '번트파울':'#edf72c', '번트헛스윙':'#1a1b1c' }

def set_fonts(name=None):
    if os.name == 'posix':
        fm.get_fontconfig_fonts()
        font_location = '/Library/Fonts/NanumSquareOTFRegular.otf'
        font_name = fm.FontProperties(fname=font_location).get_name()
        rc('font', family=font_name)
    else:
        if name is not None:
            rc('font', family=name)
            if fm.FontProperties().get_name() == 'DejaVu Sans':
                rc('font', family='NanumSquare')
        else:
            rc('font', family='NanumSquare')
        

def clean_debug(df):
    return df[['pitcher', 'batter', 'inning', 'inning_topbot', 'outs', 'balls', 'strikes',
               'pitch_result', 'pa_result', 'pitch_type', 'pa_number', 'pitch_number',
               'score_away', 'score_home',
               'on_1b', 'on_2b', 'on_3b']]
    
        
def read_light(fname):
    import warnings
    warnings.filterwarnings("ignore")
    # 포지션 데이터, 선수 ID 빼고 read

    df = pd.read_csv(fname,
                     usecols=['pitch_type', 'pitcher', 'batter', 'speed', 'pitch_result', 'pa_result',
                             'balls', 'strikes', 'outs', 'inning', 'inning_topbot',
                             'score_away', 'score_home', 'stands', 'throws',
                             'on_1b', 'on_2b', 'on_3b', 'px', 'pz', 'pfx_x', 'pfx_z',
                             'x0', 'z0', 'sz_top', 'sz_bot', 'game_date', 'home', 'away',
                             'stadium', 'referee', 'pos_2']
                       )
    df = clean_data(df)
    
    warnings.filterwarnings("default")
    return df

        
def clean_data(df):
    df = df.assign(speed = pd.to_numeric(df.speed, errors='coerce'))
    df = df.assign(px = pd.to_numeric(df.px, errors='coerce'))
    df = df.assign(pz = pd.to_numeric(df.pz, errors='coerce'))
    df = df.assign(sz_top = pd.to_numeric(df.sz_top, errors='coerce'))
    df = df.assign(sz_bot = pd.to_numeric(df.sz_bot, errors='coerce'))
    df = df.assign(pfx_x = pd.to_numeric(df.pfx_x, errors='coerce'))
    df = df.assign(pfx_z = pd.to_numeric(df.pfx_z, errors='coerce'))
    df = df.assign(x0 = pd.to_numeric(df.x0, errors='coerce'))
    df = df.assign(z0 = pd.to_numeric(df.z0, errors='coerce'))
    
    df = df.drop(df.loc[df.px.isnull()].index)
    df = df.drop(df.loc[df.pz.isnull()].index)
    df = df.drop(df.loc[df.pitch_type.isnull()].index)
    df = df.drop(df.loc[df.pitch_type == 'None'].index)
    df = df.drop(df.loc[df.sz_bot.isnull()].index)
    df = df.drop(df.loc[df.sz_top.isnull()].index)
    
    return df


def preprocess_data(df):
    df = df.assign(pit_team=np.where(df.inning_topbot == '말', df.away, df.home))
    df = df.assign(hit_team=np.where(df.inning_topbot == '초', df.away, df.home))

    df = df.assign(calls=np.where(df.pitch_result=='스트라이크', 1, 0))
    df = df.assign(stands_cat=np.where(df.stands=='양',
                                         np.where(df.throws=='좌', 1, 0),
                                         np.where(df.stands=='우', 0, 1)))

    df.stadium = pd.Categorical(df.stadium)
    df = df.assign(venue=df.stadium.cat.codes)
    return df


def plot_strike_calls(df, title=None, show_pitch_number=False, print_std=False):
    return plot_by_call(df, title, calls=['스트라이크', '볼'], legends=True, show_pitch_number=show_pitch_number, print_std=print_std)


def plot_by_call(df, title=None, calls=None, legends=True, show_pitch_number=False, is_cm=False, dpi=80, ax=None):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)

    # 단위: 피트; 좌우폭=17인치=17/24피트, 공1개 지름=약 3인치=1/4피트; 공반개=1/8피트    
    lb = -1.5 # leftBorder
    rb = +1.5  # rightBorder
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24

    bl = 1.59
    tl = 3.44
    obl = bl-3/24
    otl = tl+3/24
    bb = (bl+tl)/2 - (tl-bl)*15/16  # bottomBorder
    tb = (bl+tl)/2 + (tl-bl)*15/16  # topBorder
    
    if is_cm is True:
        lb = lb * 30.48
        rb = rb * 30.48
        bb = bb * 30.48
        tb = tb * 30.48
        ll = ll * 30.48
        rl = rl * 30.48
        oll = oll * 30.48
        orl = orl * 30.48
        bl = bl * 30.48
        tl = tl * 30.48
        obl = obl * 30.48
        otl = otl * 30.48
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5), dpi=dpi, facecolor='grey')
    else:
        fig = None
    
    if calls is None:
        calls_ = df.pitch_result.drop_duplicates()
    elif type(calls) is list:
        calls_ = calls
    elif type(calls) is str:
        calls_ = calls
    else:
        print()
        print( 'ERROR: call option must be either string or list' )
        exit(1)
        
    for c in calls_:
        f = df.loc[df.pitch_result == c]
        ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*fig.dpi, label=c, cmap='set1', zorder=0)

        if show_pitch_number is True:
            for i in f.index:
                if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                    ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                            color='white', fontsize='medium', weight='bold', horizontalalignment='center')
    
    ax.plot( [ll, ll], [bl, tl], color='white', linestyle='solid', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color='white', linestyle='solid', lw=1 )
    ax.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='white', linestyle= 'solid', lw=.5 )
    ax.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='white', linestyle= 'solid', lw=.5 )

    ax.plot( [ll, rl], [bl, bl], color='white', linestyle='solid', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color='white', linestyle='solid', lw=1 )
    ax.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='white', linestyle= 'solid', lw=.5 )
    ax.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='white', linestyle= 'solid', lw=.5 )

    ax.plot( [oll, oll], [obl, otl], color='white', linestyle='solid', lw=1 )
    ax.plot( [orl, orl], [obl, otl], color='white', linestyle='solid', lw=1 )

    ax.plot( [oll, orl], [obl, obl], color='white', linestyle='solid', lw=1 )
    ax.plot( [oll, orl], [otl, otl], color='white', linestyle='solid', lw=1 )
    ax.axis( [lb, rb, bb, tb] )

    if title is not None:
        plt.title(title, fontsize='xx-large', color='white', weight='bold', horizontalalignment='center')
    
    plt.axis('off')
    ax.autoscale_view('tight')
    
    if legends is True:
        ax.legend(loc='lower center', ncol=2, fontsize='medium')
        
    return fig, ax


def plot_by_pitch_type(df, title=None, pitch_types=None, legends=True, show_pitch_number=False, is_cm=False, dpi=80, ax=None):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)

    # 단위: 피트; 좌우폭=17인치=17/24피트, 공1개 지름=약 3인치=1/4피트; 공반개=1/8피트    
    lb = -1.5 # leftBorder
    rb = +1.5  # rightBorder
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24

    bl = 1.59
    tl = 3.44
    obl = bl-3/24
    otl = tl+3/24
    bb = (bl+tl)/2 - (tl-bl)*15/16  # bottomBorder
    tb = (bl+tl)/2 + (tl-bl)*15/16  # topBorder
    
    if is_cm is True:
        lb = lb * 30.48
        rb = rb * 30.48
        bb = bb * 30.48
        tb = tb * 30.48
        ll = ll * 30.48
        rl = rl * 30.48
        oll = oll * 30.48
        orl = orl * 30.48
        bl = bl * 30.48
        tl = tl * 30.48
        obl = obl * 30.48
        otl = otl * 30.48
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5), dpi=dpi, facecolor='grey')
    else:
        fig = None
    
    if pitch_types is None:
        pitch_types_ = df.pitch_type.drop_duplicates()
    elif type(pitch_types) is list:
        pitch_types_ = pitch_types
    elif type(pitch_types) is str:
        pitch_types_ = pitch_types
    else:
        print()
        print( 'ERROR: call option must be either string or list' )
        exit(1)
        
    for p in pitch_types_:
        f = df.loc[df.pitch_type == p]
        ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*fig.dpi, label=p, cmap='set1', zorder=0)

        if show_pitch_number is True:
            for i in f.index:
                if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                    ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                            color='white', fontsize='medium', weight='bold', horizontalalignment='center')
    
    ax.plot( [ll, ll], [bl, tl], color='white', linestyle='solid', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color='white', linestyle='solid', lw=1 )
    ax.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='white', linestyle= 'solid', lw=.5 )
    ax.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='white', linestyle= 'solid', lw=.5 )

    ax.plot( [ll, rl], [bl, bl], color='white', linestyle='solid', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color='white', linestyle='solid', lw=1 )
    ax.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='white', linestyle= 'solid', lw=.5 )
    ax.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='white', linestyle= 'solid', lw=.5 )

    ax.plot( [oll, oll], [obl, otl], color='white', linestyle='solid', lw=1 )
    ax.plot( [orl, orl], [obl, otl], color='white', linestyle='solid', lw=1 )

    ax.plot( [oll, orl], [obl, obl], color='white', linestyle='solid', lw=1 )
    ax.plot( [oll, orl], [otl, otl], color='white', linestyle='solid', lw=1 )
    ax.axis( [lb, rb, bb, tb] )

    if title is not None:
        plt.title(title, fontsize='xx-large', color='white', weight='bold', horizontalalignment='center')
    
    plt.axis('off')
    ax.autoscale_view('tight')
    
    if legends is True:
        ax.legend(loc='lower center', ncol=2, fontsize='medium')
        
    return fig, ax


# 경기 전체 call
def plot_match_calls(df, title=None):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)
    
    lb = -1.5  # leftBorder
    rb = +1.5  # rightBorder
    tb = +4.0  # topBorder
    bb = +1.0  # bottomBorder
    
    ll = -17/24  # leftLine
    rl = +17/24  # rightLine
    tl = +3.325  # topLine
    bl = +1.579  # bototmLine
    
    oll = -17/24-1/8  # outerLeftLine
    orl = +17/24+1/8  # outerRightLine
    otl = +3.325+1/8  # outerTopLine
    obl = +1.579-1/8  # outerBottomLine
    
    fig = plt.figure(figsize=(12,7), dpi=160, facecolor='#898f99')
    
    if title is not None:
        st = fig.suptitle(title, fontsize=20)
        st.set_color('white')
        st.set_weight('bold')
        st.set_horizontalalignment('center')

    # NaN, None 값들을 제거
    sub_df = df
    if df.px.isnull().any():
        sub_df = df.drop(df.loc[df.px.isnull()].index)
    
    strikes = sub_df.loc[sub_df.pitch_result == '스트라이크']
    balls = sub_df.loc[sub_df.pitch_result == '볼']
    whiffs = sub_df.loc[(sub_df.pitch_result == '헛스윙') | (sub_df.pitch_result == '번트헛스윙')]
    fouls = sub_df.loc[(sub_df.pitch_result == '파울') | (sub_df.pitch_result == '번트파울')]
    inplays = sub_df.loc[sub_df.pitch_result == '타격']
    
    ############
    # 1/6 : 스트라이크+볼
    ############
    ax = fig.add_subplot(231, facecolor='#313133')
    ax.tick_params(axis='x', colors='white')
    
    plt.scatter(strikes.px, strikes.pz, color='#ef2926', alpha=.5, s=np.pi*50, label='스트라이크')
    plt.scatter(balls.px, balls.pz, color='#3245ef', alpha=.5, s=np.pi*50, label='볼')
    
    plt.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.axis( [lb, rb, bb, tb] )

    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.autoscale_view('tight')
    ax.text(0, 3.8, '스트라이크+볼', color='white', fontsize=14, horizontalalignment='center', weight='bold')

    ############
    # 2/6 : 스트라이크
    ############
    ax = fig.add_subplot(232, facecolor='#313133')
    ax.tick_params(axis='x', colors='white')

    plt.scatter(strikes.px, strikes.pz, color='#ef2926', alpha=.5, s=np.pi*50, label='스트라이크')
    
    plt.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.axis( [lb, rb, bb, tb] )

    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.autoscale_view('tight')
    ax.text(0, 3.8, '스트라이크', color='white', fontsize=14, horizontalalignment='center', weight='bold')
    
    ############
    # 3/6 : 볼
    ############
    ax = fig.add_subplot(233, facecolor='#313133')
    ax.tick_params(axis='x', colors='white')

    plt.scatter(balls.px, balls.pz, color='#3245ef', alpha=.5, s=np.pi*50, label='볼')

    plt.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.axis( [lb, rb, bb, tb] )
    
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.text(0, 3.8, '볼', color='white', fontsize=14, horizontalalignment='center', weight='bold')
    ax.autoscale_view('tight')
    
    ############
    # 4/6 : 헛스윙
    ############
    ax = fig.add_subplot(2,3,4, facecolor='#313133')
    ax.tick_params(axis='x', colors='white')

    plt.scatter(whiffs.px, whiffs.pz, color='#1a1b1c', alpha=.5, s=np.pi*50, label='헛스윙')

    plt.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.axis( [lb, rb, bb, tb] )

    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.text(0, 3.8, '헛스윙', color='white', fontsize=14, horizontalalignment='center', weight='bold')
    ax.autoscale_view('tight')

    ############
    # 5/6 : 파울
    ############
    ax = fig.add_subplot(235, facecolor='#313133')
    ax.tick_params(axis='x', colors='white')

    plt.scatter(fouls.px, fouls.pz, color='#edf72c', alpha=.5, s=np.pi*50, label='파울')

    plt.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.axis( [lb, rb, bb, tb] )

    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.text(0, 3.8, '파울', color='white', fontsize=14, horizontalalignment='center', weight='bold')
    ax.autoscale_view('tight')

    ############
    # 6/6 : 인플레이(타격)
    ############
    ax = fig.add_subplot(236, facecolor='#d19c49')
    ax.tick_params(axis='x', colors='white')

    plt.scatter(inplays.px, inplays.pz, color='#8348d1', alpha=.5, s=np.pi*50, label='인플레이')

    plt.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
    plt.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    plt.axis( [lb, rb, bb, tb] )

    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.text(0, 3.8, '인플레이', color='white', fontsize=14, horizontalalignment='center', weight='bold')
    ax.autoscale_view('tight')
    
    plt.show()


def fmt(x, pos):
    return r'{}%'.format(int(x*100))


def plot_contour_balls(df, title=None, dpi=144, is_cm=False, cmap=None, ax=None):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)
    
    lb = -1.5
    rb = +1.5
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24

    bl = 1.59
    tl = 3.44
    obl = bl-3/24
    otl = tl+3/24
    bb = (bl+tl)/2 - (tl-bl)*15/16
    tb = (bl+tl)/2 + (tl-bl)*15/16
    
    if is_cm is True:
        lb = lb * 30.48
        rb = rb * 30.48
        bb = bb * 30.48
        tb = tb * 30.48
        ll = ll * 30.48
        rl = rl * 30.48
        oll = oll * 30.48
        orl = orl * 30.48
        bl = bl * 30.48
        tl = tl * 30.48
        obl = obl * 30.48
        otl = otl * 30.48
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4), dpi=dpi, facecolor='white')
    else:
        fig = None
    
    if is_cm is False:
        major_xtick_step = major_ytick_step = 1/2
        minor_xtick_step = minor_ytick_step = 1/10
    else:
        major_xtick_step = major_ytick_step = 20
        minor_xtick_step = minor_ytick_step = 5
    
    major_xticks = np.arange(lb, rb+major_xtick_step, major_xtick_step)
    minor_xticks = np.arange(lb, rb+minor_xtick_step, minor_xtick_step)
    
    major_yticks = np.arange(bb, tb+major_ytick_step, major_ytick_step)
    minor_yticks = np.arange(bb, tb+minor_ytick_step, minor_ytick_step)
            
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    
    plt.rcParams['axes.unicode_minus'] = False
    
    if cmap is None:
        cmap='Reds'
    
    sns.kdeplot(df.px, df.pz, shade=True, clip=((lb, rb), (bb, tb)), legend=False,
                cbar=True, cmap=cmap, cbar_kws={'format': ticker.FuncFormatter(fmt)},
                ax=ax, zorder=0)

    sns.kdeplot(df.px, df.pz, clip=((lb, rb), (bb, tb)), legend=False,
                cmap=cmap, linewidths=2.5,
                ax=ax, zorder=1)
    
    ax.plot( [ll, ll], [bl, tl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [ll, rl], [bl, bl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [oll, oll], [obl, otl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [orl, orl], [obl, otl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [oll, orl], [obl, obl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [oll, orl], [otl, otl], color='black', linestyle='dashed', lw=1 )

    ax.grid(which='minor', alpha=1.0, color='white', linewidth=0.2, zorder=10)
    ax.grid(which='major', alpha=1.0, color='white', linewidth=0.2, zorder=10)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    
    ax.autoscale_view('tight')

    if title is not None:
        ax.set_title(title, fontsize='medium')
    
    return fig, ax


def plot_heatmap(df, title=None, dpi=144, is_cm=False, cmap=None, ax=None, show_full=False, color=None):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)
    
    lb = -1.5
    rb = +1.5
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24

    bl = 1.59
    tl = 3.44
    obl = bl-3/24
    otl = tl+3/24
    bb = (bl+tl)/2 - (tl-bl)*15/16
    tb = (bl+tl)/2 + (tl-bl)*15/16
    
    if is_cm is True:
        lb = lb * 30.48
        rb = rb * 30.48
        bb = bb * 30.48
        tb = tb * 30.48
        ll = ll * 30.48
        rl = rl * 30.48
        oll = oll * 30.48
        orl = orl * 30.48
        bl = bl * 30.48
        tl = tl * 30.48
        obl = obl * 30.48
        otl = otl * 30.48
    
    strikes = df.loc[df.pitch_result == '스트라이크']
    balls = df.loc[df.pitch_result == '볼']

    bins = 36

    c1, x, y, _ = plt.hist2d(strikes.px, strikes.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    c2, x, y, _ = plt.hist2d(balls.px, balls.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    plt.close()

    np.seterr(divide='ignore', invalid='ignore')
    r = np.nan_to_num(c1 / (c1+c2))
    np.seterr(divide=None, invalid=None)
    rg = gaussian_filter(r, sigma=1.5, truncate=1, mode='constant')
    
    x, y = np.mgrid[lb:rb:bins*1j, bb:tb:bins*1j]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4), dpi=dpi, facecolor='white')
    else:
        fig = None
    ax.cla()
    
    if cmap is None:
        cmap='Reds'
    
    if cmap is not None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('mycmap', [color]*6)

    if show_full is True:
        levels = np.asarray([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
        cmap = LinearSegmentedColormap.from_list('mycmap', [color]*10)
    else:
        levels = np.asarray([.5, .6, .7, .8, .9, 1.])

    cs = ax.contourf(x, y, rg, levels=levels, cmap=cmap, zorder=1)
    ax.contour(x, y, rg, levels=levels, cmap=cmap, linewidths=2, zorder=1)
    
    ax.set_facecolor('#cccccc')
    plt.colorbar(cs, format=ticker.FuncFormatter(fmt), ax=ax)
    
    if is_cm is False:
        major_xtick_step = major_ytick_step = 1/2
        minor_xtick_step = minor_ytick_step = 1/12
    else:
        major_xtick_step = major_ytick_step = 20
        minor_xtick_step = minor_ytick_step = 20/6
    
    major_xticks = np.arange(lb, rb+major_xtick_step, major_xtick_step)
    minor_xticks = np.arange(lb, rb+minor_xtick_step, minor_xtick_step)
    
    major_yticks = np.arange(0, 5+major_ytick_step, major_ytick_step)
    minor_yticks = np.arange(0, 5+minor_ytick_step, minor_ytick_step)

    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    
    ax.plot( [ll, ll], [bl, tl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [ll, rl], [bl, bl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [oll, oll], [obl, otl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [orl, orl], [obl, otl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [oll, orl], [obl, obl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [oll, orl], [otl, otl], color='black', linestyle='dashed', lw=1 )

    ax.grid(which='minor', color='white', linewidth=0.1, zorder=10)
    ax.grid(which='major', color='white', linewidth=0.2, zorder=10)

    ax.autoscale_view('tight')
    plt.rcParams['axes.unicode_minus'] = False
    
    ax.set_xbound(lb, rb)
    ax.set_ybound(bb, tb)
    
    if title is not None:
        ax.set_title(title, fontsize='x-large')
    
    return fig, ax


def plot_szone(df, title=None, dpi=144, is_cm=False, show_area=False, ax=None):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)
    
    lb = -1.5
    rb = +1.5
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24

    bl = 1.59
    tl = 3.44
    obl = bl-3/24
    otl = tl+3/24
    bb = 1.0
    tb = 4.0
    
    if is_cm is True:
        lb = lb * 30.48
        rb = rb * 30.48
        bb = bb * 30.48
        tb = tb * 30.48
        ll = ll * 30.48
        rl = rl * 30.48
        oll = oll * 30.48
        orl = orl * 30.48
        bl = bl * 30.48
        tl = tl * 30.48
        obl = obl * 30.48
        otl = otl * 30.48
    
    strikes = df.loc[df.pitch_result == '스트라이크']
    balls = df.loc[df.pitch_result == '볼']

    bins = 36

    c1, x, y, i = plt.hist2d(strikes.px, strikes.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    c2, x, y, i = plt.hist2d(balls.px, balls.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    plt.close()

    np.seterr(divide='ignore', invalid='ignore')
    r = np.nan_to_num(c1 / (c1+c2))
    np.seterr(divide=None, invalid=None)
    rg = gaussian_filter(r, sigma=1.5, truncate=1, mode='constant')
    
    x, y = np.meshgrid(np.arange(lb, rb, (rb-lb)/bins), np.arange(bb, tb, (tb-bb)/bins))

    x = x.T
    y = y.T
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5), dpi=dpi, facecolor='white')
    else:
        fig = None
    
    cmap = matplotlib.colors.ListedColormap(['white', '#ccffcc'])
    plt.pcolor(x, y, rg, cmap=cmap)
    
    if is_cm is False:
        major_xtick_step = major_ytick_step = 1/2
        minor_xtick_step = minor_ytick_step = 1/10
    else:
        major_xtick_step = major_ytick_step = 20
        minor_xtick_step = minor_ytick_step = 5
    
    major_xticks = np.arange(lb, rb+major_xtick_step, major_xtick_step)
    minor_xticks = np.arange(lb, rb+minor_xtick_step, minor_xtick_step)
    
    major_yticks = np.arange(0, 5+major_ytick_step, major_ytick_step)
    minor_yticks = np.arange(0, 5+minor_ytick_step, minor_ytick_step)

    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    
    ax.plot( [ll, ll], [bl, tl], color='black', linestyle='solid', lw=0.5 )
    ax.plot( [rl, rl], [bl, tl], color='black', linestyle='solid', lw=0.5 )

    ax.plot( [ll, rl], [bl, bl], color='black', linestyle='solid', lw=0.5 )
    ax.plot( [ll, rl], [tl, tl], color='black', linestyle='solid', lw=0.5 )

    ax.plot( [oll, oll], [obl, otl], color='black', linestyle='solid', lw=0.5 )
    ax.plot( [orl, orl], [obl, otl], color='black', linestyle='solid', lw=0.5 )

    ax.plot( [oll, orl], [obl, obl], color='black', linestyle='solid', lw=0.5 )
    ax.plot( [oll, orl], [otl, otl], color='black', linestyle='solid', lw=0.5 )

    ax.grid(which='minor', alpha=1.0, color='grey', linewidth=0.2)
    ax.grid(which='major', alpha=1.0, color='grey', linewidth=0.2)

    bb = (bl+tl)/2 - (tl-bl)*15/16
    tb = (bl+tl)/2 + (tl-bl)*15/16
    
    ax.set_ybound(bb, tb)
    
    ax.autoscale_view('tight')
    plt.rcParams['axes.unicode_minus'] = False
    
    if title is not None:
        plt.title(title, fontsize=14, horizontalalignment='center')
    
    area = np.sum(rg >= 0.5)
    print('S-Zone size: {} sq.inch'.format(area))
    
    if show_area is True:
        ax.text( 0, (tb+bb)/2, '{} sq. inch'.format(str(area)), color='black',
                 fontsize=16, horizontalalignment='center' )
    
    return fig, ax


def release_point(df, title=None, pitcher=None, xlim=None, ylim=None, square=True, ax=None):
    if pitcher is not None:
        sub_df = df.loc[df.pitcher == pitcher]
    else:
        sub_df = df
        
    if sub_df.px.dtypes == np.object:
        sub_df = clean_data(sub_df)
    
    pitches = sub_df.pitch_type.drop_duplicates()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=100, facecolor='white')
    else:
        fig = None
    
    for p in pitches:
        ax.scatter(sub_df.loc[sub_df.pitch_type == p].x0, sub_df.loc[sub_df.pitch_type == p].z0, s=np.pi*20, label=p,cmap='set1')
    
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_xbound(-3, 3)
    ax.set_ybound(0, 10)
    if xlim is not None:
        if type(xlim) is list:
            if (len(xlim) == 2) & all(isinstance(x, numbers.Number) for x in xlim):
                ax.set_xbound(min(xlim), max(xlim))
    if ylim is not None:
        if type(ylim) is list:
            if (len(ylim) == 2) & all(isinstance(y, numbers.Number) for y in ylim):
                ax.set_ybound(min(ylim), max(ylim))
    
    ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    if square is True:
        ax.axis('square')
        xmin, xmax = ax.get_xbound()
        ymin, ymax = ax.get_ybound()
        ax.set_xbound((xmax+xmin)/2 - 1, (xmax+xmin)/2 + 1)
        ax.set_ybound((ymax+ymin)/2 - 1, (ymax+ymin)/2 + 1)
        
    if title is not None:
        ax.set_title(title, fontsize='medium')
    
    return fig, ax


def pitcher_info(df, pitcher=None):
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    if pitcher is not None:
        sub_df = df.loc[df.pitcher == pitcher]
    else:
        sub_df = df
    
    if sub_df.px.dtypes == np.object:
        sub_df = clean_data(sub_df)
        
    groupped = sub_df.groupby('pitch_type').mean().loc[:, ['speed', 'pfx_x', 'pfx_z']]
    
    groupped['count'] = sub_df.groupby('pitch_type').count().speed
    groupped['max'] = sub_df.groupby('pitch_type').max().speed
    groupped['min'] = sub_df.groupby('pitch_type').min().speed
    
    groupped['pct'] = groupped['count'] / groupped['count'].sum() * 100
    
    #display(groupped)
    return groupped


def pitcher_plate_discipline(df, pitcher=None, by_pitch=False):
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    if pitcher is not None:
        if isinstance(pitcher, str):
            sub_df = df.loc[df.pitcher == pitcher]
        elif isinstance(pitcher, list):
            sub_df = df.loc[df.pitcher.isin(pitcher)]
        elif isinstance(pitcher, pd.Series):
            if pitcher.dtypes == np.object:
                sub_df = df.loc[df.pitcher.isin(pitcher)]
            else:
                sub_df = df.loc[df.pitcher.isin(pitcher.index)]
        else:
            return False
    else:
        sub_df = df

    if sub_df.px.dtypes == np.object:
        sub_df = clean_data(sub_df)
    
    sub_df = sub_df.assign(swing=np.where(sub_df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울']), 1, 0))
    sub_df = sub_df.assign(miss=np.where(sub_df.pitch_result.isin(['번트헛스윙', '헛스윙']), 1, 0))

    izmask = sub_df.px.between(-10/12, 10/12) & sub_df.pz.between(1.6, 3.6)
    ozmask = ~izmask

    sub_df = sub_df.assign(iz_swing=
                           np.where(sub_df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                                    & izmask, 1, 0))
    sub_df = sub_df.assign(iz_miss=
                           np.where(sub_df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                                    & izmask, 1, 0))
    sub_df = sub_df.assign(oz_swing=
                           np.where(sub_df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                                    & ozmask, 1, 0))
    sub_df = sub_df.assign(oz_miss=np.where(sub_df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                                            & ozmask, 1, 0))

    if isinstance(pitcher, str):
        if by_pitch is True:
            tab =  pd.DataFrame({'raw_num': sub_df.groupby('pitch_type').count().speed,
                                 'swing': sub_df.groupby('pitch_type').sum().swing,
                                 'miss': sub_df.groupby('pitch_type').sum().miss,
                                 'iz_raw_num': sub_df.loc[izmask].groupby('pitch_type').count().speed,
                                 'iz_swing': sub_df.groupby('pitch_type').sum().iz_swing,
                                 'iz_miss': sub_df.groupby('pitch_type').sum().iz_miss,
                                 'oz_raw_num': sub_df.loc[ozmask].groupby('pitch_type').count().speed,
                                 'oz_swing': sub_df.groupby('pitch_type').sum().oz_swing,
                                 'oz_miss': sub_df.groupby('pitch_type').sum().oz_miss
                                })
        else:
            d = {'raw_num': sub_df.count().speed,
                 'swing': sub_df.sum().swing,
                 'miss': sub_df.sum().miss,
                 'iz_raw_num': sub_df.loc[izmask].count().speed,
                 'iz_swing': sub_df.sum().iz_swing,
                 'iz_miss': sub_df.sum().iz_miss,
                 'oz_raw_num': sub_df.loc[ozmask].count().speed,
                 'oz_swing': sub_df.sum().oz_swing,
                 'oz_miss': sub_df.sum().oz_miss
                }

            tab = pd.DataFrame(data=d, index=[pitcher])
    else:
        if by_pitch is True:
            tab =  pd.DataFrame({'raw_num': sub_df.groupby(['pitcher', 'pitch_type']).count().speed,
                                 'swing': sub_df.groupby(['pitcher', 'pitch_type']).sum().swing,
                                 'miss': sub_df.groupby(['pitcher', 'pitch_type']).sum().miss,
                                 'iz_raw_num': sub_df.loc[izmask].groupby(['pitcher', 'pitch_type']).count().speed,
                                 'iz_swing': sub_df.groupby(['pitcher', 'pitch_type']).sum().iz_swing,
                                 'iz_miss': sub_df.groupby(['pitcher', 'pitch_type']).sum().iz_miss,
                                 'oz_raw_num': sub_df.loc[ozmask].groupby(['pitcher', 'pitch_type']).count().speed,
                                 'oz_swing': sub_df.groupby(['pitcher', 'pitch_type']).sum().oz_swing,
                                 'oz_miss': sub_df.groupby(['pitcher', 'pitch_type']).sum().oz_miss
                                })
        else:
            tab =  pd.DataFrame({'raw_num': sub_df.groupby(['pitcher']).count().speed,
                                 'swing': sub_df.groupby(['pitcher']).sum().swing,
                                 'miss': sub_df.groupby(['pitcher']).sum().miss,
                                 'iz_raw_num': sub_df.loc[izmask].groupby(['pitcher']).count().speed,
                                 'iz_swing': sub_df.groupby(['pitcher']).sum().iz_swing,
                                 'iz_miss': sub_df.groupby(['pitcher']).sum().iz_miss,
                                 'oz_raw_num': sub_df.loc[ozmask].groupby(['pitcher']).count().speed,
                                 'oz_swing': sub_df.groupby(['pitcher']).sum().oz_swing,
                                 'oz_miss': sub_df.groupby(['pitcher']).sum().oz_miss
                                })
        
    tab = tab.assign(swing_p = tab.swing / tab.raw_num*100,
                     swstr_p = tab.miss / tab.raw_num*100,
                     iz_swing_p = tab.iz_swing / tab.iz_raw_num*100,
                     iz_con_p = (1 - tab.iz_miss / tab.iz_swing)*100,
                     oz_swing_p = tab.oz_swing / tab.oz_raw_num*100,
                     oz_con_p = (1 - tab.oz_miss / tab.oz_swing)*100,
                    )
    
    return tab[['swing_p', 'swstr_p', 'iz_swing_p', 'iz_con_p', 'oz_swing_p', 'oz_con_p']]


def batter_plate_discipline(df, batter=None, by_pitch=False):
    pd.set_option('display.float_format', lambda x: '%.1f' % x)
    if batter is not None:
        if isinstance(batter, str):
            sub_df = df.loc[df.batter == batter]
        elif isinstance(batter, list):
            sub_df = df.loc[df.batter.isin(batter)]
        elif isinstance(batter, pd.Series):
            if batter.dtypes == np.object:
                sub_df = df.loc[df.batter.isin(batter)]
            else:
                sub_df = df.loc[df.batter.isin(batter.index)]
        else:
            return False
    else:
        sub_df = df
    
    if sub_df.px.dtypes == np.object:
        sub_df = clean_data(sub_df)
    
    sub_df = sub_df.assign(swing=np.where(sub_df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울']), 1, 0))
    sub_df = sub_df.assign(miss=np.where(sub_df.pitch_result.isin(['번트헛스윙', '헛스윙']), 1, 0))

    izmask = sub_df.px.between(-10/12, 10/12) & sub_df.pz.between(1.6, 3.6)
    ozmask = ~izmask

    sub_df = sub_df.assign(iz_swing=
                           np.where(sub_df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                                    & izmask, 1, 0))
    sub_df = sub_df.assign(iz_miss=
                           np.where(sub_df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                                    & izmask, 1, 0))
    sub_df = sub_df.assign(oz_swing=
                           np.where(sub_df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                                    & ozmask, 1, 0))
    sub_df = sub_df.assign(oz_miss=np.where(sub_df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                                            & ozmask, 1, 0))

    
    if isinstance(batter, str):
        if by_pitch is True:
            tab =  pd.DataFrame({'raw_num': sub_df.groupby('pitch_type').count().speed,
                                 'swing': sub_df.groupby('pitch_type').sum().swing,
                                 'miss': sub_df.groupby('pitch_type').sum().miss,
                                 'iz_raw_num': sub_df.loc[izmask].groupby('pitch_type').count().speed,
                                 'iz_swing': sub_df.groupby('pitch_type').sum().iz_swing,
                                 'iz_miss': sub_df.groupby('pitch_type').sum().iz_miss,
                                 'oz_raw_num': sub_df.loc[ozmask].groupby('pitch_type').count().speed,
                                 'oz_swing': sub_df.groupby('pitch_type').sum().oz_swing,
                                 'oz_miss': sub_df.groupby('pitch_type').sum().oz_miss
                                })
        else:
            d = {'raw_num': sub_df.count().speed,
                 'swing': sub_df.sum().swing,
                 'miss': sub_df.sum().miss,
                 'iz_raw_num': sub_df.loc[izmask].count().speed,
                 'iz_swing': sub_df.sum().iz_swing,
                 'iz_miss': sub_df.sum().iz_miss,
                 'oz_raw_num': sub_df.loc[ozmask].count().speed,
                 'oz_swing': sub_df.sum().oz_swing,
                 'oz_miss': sub_df.sum().oz_miss
                }

            tab = pd.DataFrame(data=d, index=[batter])

        tab = tab.assign(swing_p = tab.swing / tab.raw_num*100,
                         swstr_p = tab.miss / tab.raw_num*100,
                         iz_swing_p = tab.iz_swing / tab.iz_raw_num*100,
                         iz_con_p = (1 - tab.iz_miss / tab.iz_swing)*100,
                         oz_swing_p = tab.oz_swing / tab.oz_raw_num*100,
                         oz_con_p = (1 - tab.oz_miss / tab.oz_swing)*100,
                        )
    else:
        if by_pitch is True:
            tab =  pd.DataFrame({'raw_num': sub_df.groupby(['batter', 'pitch_type']).count().speed,
                                 'swing': sub_df.groupby(['batter', 'pitch_type']).sum().swing,
                                 'miss': sub_df.groupby(['batter', 'pitch_type']).sum().miss,
                                 'iz_raw_num': sub_df.loc[izmask].groupby(['batter', 'pitch_type']).count().speed,
                                 'iz_swing': sub_df.groupby(['batter', 'pitch_type']).sum().iz_swing,
                                 'iz_miss': sub_df.groupby(['batter', 'pitch_type']).sum().iz_miss,
                                 'oz_raw_num': sub_df.loc[ozmask].groupby(['batter', 'pitch_type']).count().speed,
                                 'oz_swing': sub_df.groupby(['batter', 'pitch_type']).sum().oz_swing,
                                 'oz_miss': sub_df.groupby(['batter', 'pitch_type']).sum().oz_miss
                                })
        else:
            tab =  pd.DataFrame({'raw_num': sub_df.groupby(['batter']).count().speed,
                                 'swing': sub_df.groupby(['batter']).sum().swing,
                                 'miss': sub_df.groupby(['batter']).sum().miss,
                                 'iz_raw_num': sub_df.loc[izmask].groupby(['batter']).count().speed,
                                 'iz_swing': sub_df.groupby(['batter']).sum().iz_swing,
                                 'iz_miss': sub_df.groupby(['batter']).sum().iz_miss,
                                 'oz_raw_num': sub_df.loc[ozmask].groupby(['batter']).count().speed,
                                 'oz_swing': sub_df.groupby(['batter']).sum().oz_swing,
                                 'oz_miss': sub_df.groupby(['batter']).sum().oz_miss
                                })

        tab = tab.assign(swing_p = tab.swing / tab.raw_num*100,
                         swstr_p = tab.miss / tab.raw_num*100,
                         iz_swing_p = tab.iz_swing / tab.iz_raw_num*100,
                         iz_con_p = (1 - tab.iz_miss / tab.iz_swing)*100,
                         oz_swing_p = tab.oz_swing / tab.oz_raw_num*100,
                         oz_con_p = (1 - tab.oz_miss / tab.oz_swing)*100,
                        )
            
    
    return tab[['swing_p', 'swstr_p', 'iz_swing_p', 'iz_con_p', 'oz_swing_p', 'oz_con_p']]


RV = np.asarray([
    0.087, 0.124, 0.177, 0.149,
    0.104, 0.136, 0.210, 0.281,
    0.248, 0.294, 0.402, 0.689
])

def plot_by_proba(df, title=None, dpi=144, is_cm=False, cmap=None, ax=None):
    set_fonts()
    if 'proba' not in df.keys():
        print('Key "proba" not in dataframe')
        return
    if df.px.dtypes == np.object:
        df = clean_data(df)
    
    lb = -1.5
    rb = +1.5
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24

    bl = 1.59
    tl = 3.44
    obl = bl-3/24
    otl = tl+3/24
    bb = (bl+tl)/2 - (tl-bl)*15/16
    tb = (bl+tl)/2 + (tl-bl)*15/16
    
    if is_cm is True:
        lb = lb * 30.48
        rb = rb * 30.48
        bb = bb * 30.48
        tb = tb * 30.48
        ll = ll * 30.48
        rl = rl * 30.48
        oll = oll * 30.48
        orl = orl * 30.48
        bl = bl * 30.48
        tl = tl * 30.48
        obl = obl * 30.48
        otl = otl * 30.48

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4), dpi=dpi, facecolor='white')
    else:
        fig = None
    
    if cmap is None:
        cmap='Reds'
    ax.set_facecolor('#cccccc')
    cs = ax.scatter(df.px, df.pz, alpha=.5, s=np.pi/2*dpi, c=df.proba, cmap=cmap, zorder=0, vmin=0, vmax=1)
    plt.colorbar(cs, format=ticker.FuncFormatter(fmt), spacing='proportional', ax=ax)
    
    if is_cm is False:
        major_xtick_step = major_ytick_step = 1/2
        minor_xtick_step = minor_ytick_step = 1/12
    else:
        major_xtick_step = major_ytick_step = 20
        minor_xtick_step = minor_ytick_step = 20/6
    
    major_xticks = np.arange(lb, rb+major_xtick_step, major_xtick_step)
    minor_xticks = np.arange(lb, rb+minor_xtick_step, minor_xtick_step)
    
    major_yticks = np.arange(0, 5+major_ytick_step, major_ytick_step)
    minor_yticks = np.arange(0, 5+minor_ytick_step, minor_ytick_step)

    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor=True)
    
    ax.plot( [ll, ll], [bl, tl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [ll, rl], [bl, bl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [oll, oll], [obl, otl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [orl, orl], [obl, otl], color='black', linestyle='dashed', lw=1 )

    ax.plot( [oll, orl], [obl, obl], color='black', linestyle='dashed', lw=1 )
    ax.plot( [oll, orl], [otl, otl], color='black', linestyle='dashed', lw=1 )

    ax.grid(which='minor', color='white', linewidth=0.1, zorder=10)
    ax.grid(which='major', color='white', linewidth=0.2, zorder=10)

    ax.set_xbound(lb, rb)
    ax.set_ybound(bb, tb)
    
    ax.autoscale_view('tight')
    plt.rcParams['axes.unicode_minus'] = False
    
    if title is not None:
        ax.set_title(title, fontsize='xx-large')
    
    return fig, ax


def calc_framing_gam(df):
    # 10-80% 구간 측정이 mean을 0에 가깝게 맞출 수 있음.
    sub_df = df.loc[df.pitch_result.isin(['스트라이크', '볼']) & (df.stands != 'None') & (df.throws != 'None')]
    sub_df = sub_df.assign(stands = np.where(sub_df.stands == '양',
                                     np.where(sub_df.throws == '좌', '우', '좌'),
                                     sub_df.stands))
    sub_df = sub_df.assign(stands = np.where(sub_df.stands=='우', 0, 1)) # 우=0, 좌=1
    sub_df = sub_df.assign(throws = np.where(sub_df.throws=='우', 0, 1)) # 우=0, 좌=1
    sub_df.stadium = pd.Categorical(sub_df.stadium)
    sub_df = sub_df.assign(venue = sub_df.stadium.cat.codes)

    sub_df = sub_df.assign(pitch_result=np.where(sub_df.pitch_result=='스트라이크', 1, 0))

    features = ['px', 'pz', 'stands', 'throws', 'venue']
    label = ['pitch_result']

    x = sub_df[features]
    y = sub_df[label]

    gam = LogisticGAM().fit(x, y)
    predictions = gam.predict(x)
    proba = gam.predict_proba(x)

    logs = sub_df[features + label + ['pos_1', 'pos_2', 'balls', 'strikes', 'stadium']]
    logs = logs.rename(index=str, columns={'pos_1': 'pitcher', 'pos_2':'catcher'})

    logs = logs.assign(prediction = predictions)
    logs = logs.assign(proba = proba)
    logs = logs.assign(rv = RV[logs.balls + logs.strikes*4])
    logs = logs.assign(excall=np.where(logs.prediction!=logs.pitch_result,
                                       np.where(logs.pitch_result==1, 1, -1),0))
    logs = logs.assign(exstr=np.where(logs.excall==1, 1, 0))
    logs = logs.assign(exball=np.where(logs.excall==-1, 1, 0))
    logs = logs.assign(exrv=logs.excall * logs.rv)
    logs = logs.assign(exrv_prob=np.where(logs.excall==1, (1-logs.proba)*logs.rv,
                                              np.where(logs.excall==-1, -logs.proba*logs.rv, 0)))

    # 포수, catch 개수, extra strike, extra ball, RV sum
    tab = logs.pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum', 'exball': 'sum', 'excall': 'sum', 'exrv': 'sum', 'exrv_prob':'sum', 'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)
    
    return logs, tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)

    
def calc_framing_cell(df, is_cm=False):
    # 20-80% 구간 측정이 mean을 0에 가깝게 맞출 수 있음.
    is_cm=False
    features = ['px', 'pz', 'pitch_result', 'stands', 'throws', 'pitcher', 'catcher',
                'stadium', 'referee', 'balls', 'strikes']

    sub_df = df.loc[df.pitch_result.isin(['스트라이크', '볼']) & (df.stands != 'None') & (df.throws != 'None')]
    sub_df = sub_df.assign(stands = np.where(sub_df.stands == '양',
                                     np.where(sub_df.throws == '좌', '우', '좌'),
                                     sub_df.stands))
    sub_df = sub_df.rename(index=str, columns={'pos_1': 'pitcher', 'pos_2':'catcher'})
    logs = sub_df[features]

    logs = logs.assign(pitch_result=np.where(logs.pitch_result=='스트라이크', 1, 0))
    strikes = logs.loc[logs.pitch_result == 1]
    balls = logs.loc[logs.pitch_result == 0]

    lb = -1.5
    rb = +1.5
    bb = 1.0
    tb = 4.0

    if is_cm is True:
        lb = lb * 30.48
        rb = rb * 30.48
        bb = bb * 30.48
        tb = tb * 30.48

    bins = 36

    c1, x, y, i = plt.hist2d(strikes.px, strikes.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    c2, x, y, i = plt.hist2d(balls.px, balls.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    plt.close()

    np.seterr(divide='ignore', invalid='ignore')
    r = np.nan_to_num(c1 / (c1+c2))
    np.seterr(divide=None, invalid=None)
    probs = gaussian_filter(r, sigma=1.5, truncate=1, mode='constant')

    logs = logs.assign(x_ind = ((logs.px+1.5)*12).astype(np.int8))
    logs = logs.assign(y_ind = ((logs.pz-1)*12).astype(np.int8))
    logs = logs.assign(proba = -1)
    for i in range(0, bins):
        for j in range(0, bins):
            logs = logs.assign(proba = np.where((logs.x_ind == i) & (logs.y_ind == j),
                                                probs[i][j], logs.proba))

    logs = logs.drop(columns='x_ind')
    logs = logs.drop(columns='y_ind')
    logs = logs.assign(prediction = np.where(logs.proba >=0.5, 1, 0))
    logs = logs.assign(rv= RV[logs.balls + logs.strikes*4])
    logs = logs.assign(excall=np.where(logs.prediction!=logs.pitch_result,
                                  np.where(logs.pitch_result==1, 1, -1),0))
    logs = logs.assign(exstr=np.where(logs.excall==1, 1, 0))
    logs = logs.assign(exball=np.where(logs.excall==-1, 1, 0))
    logs = logs.assign(exrv=logs.excall * logs.rv)
    logs = logs.assign(exrv_prob=np.where(logs.excall==1, (1-logs.proba)*logs.rv,
                                          np.where(logs.excall==-1, -logs.proba*logs.rv, 0)))

    # 포수, catch 개수, extra strike, extra ball, RV sum
    tab = logs.pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum', 'exball': 'sum', 'excall': 'sum', 'exrv': 'sum', 'exrv_prob':'sum', 'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)
    
    return logs, tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)


def calc_framing_gam_adv(df, max_dist=0.25):
    lb = -1.5
    rb = +1.5
    bb = 1.0
    tb = 4.0

    strikes = df.loc[df.pitch_result == '스트라이크']
    balls = df.loc[df.pitch_result == '볼']

    bins = 36

    c1, x, y, _ = plt.hist2d(strikes.px, strikes.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    c2, x, y, _ = plt.hist2d(balls.px, balls.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    plt.close()

    np.seterr(divide='ignore', invalid='ignore')
    r = np.nan_to_num(c1 / (c1+c2))
    np.seterr(divide=None, invalid=None)
    rg = gaussian_filter(r, sigma=1.5, truncate=1, mode='constant')

    x, y = np.mgrid[lb:rb:bins*1j, bb:tb:bins*1j]

    CS = plt.contour(x, y, rg, levels=np.asarray([.5]), linewidths=2, zorder=1)
    path = CS.collections[0].get_paths()[0]
    cts = path.vertices
    d = sp.spatial.distance.cdist(cts, cts)
    plt.close()
    
    logs, _ = calc_framing_gam(df)
    logs = logs.assign(dist = np.min(sp.spatial.distance.cdist(logs[['px', 'pz']], cts), axis=1))

    tab = logs.loc[logs.dist < max_dist].pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum', 'exball': 'sum', 'excall': 'sum', 'exrv': 'sum', 'exrv_prob':'sum', 'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)
    
    return logs.loc[logs.dist < max_dist], tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)


def calc_framing_cell_adv(df, max_dist=0.25):
    lb = -1.5
    rb = +1.5
    bb = 1.0
    tb = 4.0

    strikes = df.loc[df.pitch_result == '스트라이크']
    balls = df.loc[df.pitch_result == '볼']

    bins = 36

    c1, x, y, _ = plt.hist2d(strikes.px, strikes.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    c2, x, y, _ = plt.hist2d(balls.px, balls.pz, range=[[lb, rb], [bb, tb]], bins=bins)
    plt.close()

    np.seterr(divide='ignore', invalid='ignore')
    r = np.nan_to_num(c1 / (c1+c2))
    np.seterr(divide=None, invalid=None)
    rg = gaussian_filter(r, sigma=1.5, truncate=1, mode='constant')

    x, y = np.mgrid[lb:rb:bins*1j, bb:tb:bins*1j]

    CS = plt.contour(x, y, rg, levels=np.asarray([.5]), linewidths=2, zorder=1)
    path = CS.collections[0].get_paths()[0]
    cts = path.vertices
    d = sp.spatial.distance.cdist(cts, cts)
    plt.close()
    
    logs, _ = calc_framing_cell(df)
    logs = logs.assign(dist = np.min(sp.spatial.distance.cdist(logs[['px', 'pz']], cts), axis=1))

    tab = logs.loc[logs.dist < max_dist].pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum', 'exball': 'sum', 'excall': 'sum', 'exrv': 'sum', 'exrv_prob':'sum', 'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)
    
    return logs.loc[logs.dist < max_dist], tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)


def graph_batting_result(df, batter, ma_term=0, options=[True, True, True, True, True]):
    datesFmt = mdates.DateFormatter('%m-%d')
    
    if df.game_date.dtype == np.int64:
        df = df.assign(game_date = pd.to_datetime(df.game_date, format='%Y%m%d').dt.date)
    
    df = df.assign(pa=np.where(df.pa_result != 'None', 1, 0))
    df = df.assign(ab=np.where(df.pa_result.isin(['1루타', '내야안타', '안타', '2루타', '3루타', '홈런', '희생번트 야수선택',
                                                  '삼진', '실책', '필드 아웃', '포스 아웃', '병살타', '타구맞음 아웃',
                                                  '낫아웃 폭투', '희생번트 실책', '야수선택', '낫아웃 포일', '삼중살']),
                               1, 0))
    
    df = df.assign(single=np.where(df.pa_result.isin(['1루타', '내야안타', '안타']), 1, 0))
    df = df.assign(double=np.where(df.pa_result == '2루타', 1, 0))
    df = df.assign(triple=np.where(df.pa_result == '3루타', 1, 0))
    df = df.assign(hr=np.where(df.pa_result == '홈런', 1, 0))
    
    df = df.assign(sh=np.where(df.pa_result == '희생번트', 1, 0))
    df = df.assign(sf=np.where(df.pa_result == '희생플라이', 1, 0))
    df = df.assign(so=np.where(df.pa_result.isin(['삼진', '낫아웃 폭투', '낫아웃 포일']), 1, 0))
    df = df.assign(bb=np.where(df.pa_result.isin(['볼넷', '고의4구']), 1, 0))
    df = df.assign(hbp=np.where(df.pa_result == '몸에 맞는 볼', 1, 0))
    
    df = df.assign(hit=df.single + df.double + df.triple + df.hr)
    df = df.assign(tb=df.single + df.double*2 + df.triple*3 + df.hr*4)
    
    sub_df = df.loc[df.batter == batter]
    tab = sub_df.pivot_table(index='game_date',
                             values=['hit', 'tb', 'ab', 'pa', 'hr', 'sh', 'sf', 'so', 'bb', 'hbp'],
                             aggfunc='sum'
                            )
    
    #babip = (hit-hr) / (ab-so-hr+sf)
    tab = tab.fillna(0)
    result_sum = []
    
    hit_sum = tb_sum = 0
    ab_sum = pa_sum = 0
    hr_sum = sh_sum = sf_sum = 0
    so_sum = bb_sum = hbp_sum = 0

    fig, a1 = plt.subplots(figsize=(3, 3), dpi=144, facecolor='white')
    if (ma_term <= 1) or (ma_term >= len(tab.index)):
        for i in tab.index:
            hit_sum += tab.loc[i].hit
            tb_sum += tab.loc[i].tb
            ab_sum += tab.loc[i].ab
            pa_sum += tab.loc[i].pa
            hr_sum += tab.loc[i].hr
            sh_sum += tab.loc[i].sh
            sf_sum += tab.loc[i].sf
            so_sum += tab.loc[i].so
            bb_sum += tab.loc[i].bb
            hbp_sum += tab.loc[i].hbp
            
            # avg, obp, slg, ops, babip
            
            result_sum.append([hit_sum/ab_sum, # avg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum), # obp
                               tb_sum/ab_sum, # slg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum) + tb_sum/ab_sum, #ops
                               (hit_sum-hr_sum)/(ab_sum-so_sum-hr_sum+sf_sum) # babip
                              ])
            
        fig.suptitle(f'{batter}\'s Cumulative Average', fontsize=10)
    else:
        term = ma_term - 1

        for i in range(0, len(tab.index) - term):
            hit_sum = tab.loc[tab.index[i]:tab.index[i+term]].hit.sum()
            tb_sum = tab.loc[tab.index[i]:tab.index[i+term]].tb.sum()
            ab_sum = tab.loc[tab.index[i]:tab.index[i+term]].ab.sum()
            pa_sum = tab.loc[tab.index[i]:tab.index[i+term]].pa.sum()
            hr_sum = tab.loc[tab.index[i]:tab.index[i+term]].hr.sum()
            sh_sum = tab.loc[tab.index[i]:tab.index[i+term]].sh.sum()
            sf_sum = tab.loc[tab.index[i]:tab.index[i+term]].sf.sum()
            so_sum = tab.loc[tab.index[i]:tab.index[i+term]].so.sum()
            bb_sum = tab.loc[tab.index[i]:tab.index[i+term]].bb.sum()
            hbp_sum = tab.loc[tab.index[i]:tab.index[i+term]].hbp.sum()
            
            # avg, obp, slg, ops, babip
            
            result_sum.append([hit_sum/ab_sum, # avg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum), # obp
                               tb_sum/ab_sum, # slg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum) + tb_sum/ab_sum, #ops
                               (hit_sum-hr_sum)/(ab_sum-so_sum-hr_sum+sf_sum) # babip
                              ])
            
        fig.suptitle(f'{batter}\'s {ma_term}-Game Rolling Average')

    rs = np.array(result_sum)

    a1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    min_y = 10000
    max_y = -10000
    a2 = a3 = a4 = a5 = None

    if options[0] is True:
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a1.plot(tab.index, rs[:, 0], label='AVG', color='k')
        else:
            a1.plot(tab.index[term:], rs[:, 0], label='AVG', color='k')
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_yticks([])
        max_y = max(max_y, rs[:, 0].max())
        min_y = min(min_y, rs[:, 0].min())

    if options[1] is True:
        a2 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a2.plot(tab.index, rs[:, 1], label='OBP', color='blueviolet', linestyle=':', linewidth=2)
        else:
            a2.plot(tab.index[term:], rs[:, 1], label='OBP', color='blueviolet', linestyle=':', linewidth=2)
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_yticks([])
        if max_y < rs[:, 1].max():
            max_y = rs[:, 1].max()
        if min_y > rs[:, 1].min():
            min_y = rs[:, 1].min()

    if options[2] is True:
        a3 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a3.plot(tab.index, rs[:, 2], label='SLG', color='orangered', marker='o', markersize='2.5')
        else:
            a3.plot(tab.index[term:], rs[:, 2], label='SLG', color='orangered', marker='o', markersize='2.5')
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_yticks([])
        if max_y < rs[:, 2].max():
            max_y = rs[:, 2].max()
        if min_y > rs[:, 2].min():
            min_y = rs[:, 2].min()

    if options[3] is True:
        a4 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a4.plot(tab.index, rs[:, 3], label='OPS', color='royalblue')
        else:
            a4.plot(tab.index[term:], rs[:, 3], label='OPS', color='royalblue')
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_yticks([])
        if max_y < rs[:, 3].max():
            max_y = rs[:, 3].max()
        if min_y > rs[:, 3].min():
            min_y = rs[:, 3].min()

    if options[4] is True:
        a5 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a5.plot(tab.index, rs[:, 4], label='BABIP', color='gold', linestyle='--')
        else:
            a5.plot(tab.index[term:], rs[:, 4], label='BABIP', color='gold', linestyle='--')
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_yticks([])
        if max_y < rs[:, 4].max():
            max_y = rs[:, 4].max()
        if min_y > rs[:, 4].min():
            min_y = rs[:, 4].min()

    a1.set_ylim([max(0, min_y-0.025), max_y+0.025])
    a1.set_ylabel('')
    if a2 is not None:
        a2.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a2.set_ylabel('')
    if a3 is not None:
        a3.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a3.set_ylabel('')
    if a4 is not None:
        a4.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a4.set_ylabel('')
    if a5 is not None:
        a5.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a5.set_ylabel('')
        
    fig.legend(loc='lower center', ncol=np.array(options).sum(), fontsize='xx-small', bbox_to_anchor=(0.5, -0.005))

    a1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    a1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    a1.grid(True)
    a1.tick_params(labelsize='x-small')

    return fig


def graph_batter_plate_discipline(df, batter, ma_term=0, options=[True, True, True, True, True, True]):
    datesFmt = mdates.DateFormatter('%m-%d')
    
    if df.game_date.dtype == np.int64:
        df = df.assign(game_date = pd.to_datetime(df.game_date, format='%Y%m%d').dt.date)
    
    if df.px.dtypes == np.object:
        df.loc[:, 'px'] = pd.to_numeric(df.px, errors='coerce')
    if df.isnull().any().px is False:
        df = df.drop(df.loc[df.px.isnull()].index)

    if df.pz.dtypes == np.object:
        df.loc[:, 'pz'] = pd.to_numeric(df.pz, errors='coerce')
    if df.isnull().any().pz is False:
        df = df.drop(df.loc[df.pz.isnull()].index)
        
    df = df.assign(swing=np.where(df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울']), 1, 0))
    df = df.assign(miss=np.where(df.pitch_result.isin(['번트헛스윙', '헛스윙']), 1, 0))

    izmask = df.px.between(-10/12, 10/12) & df.pz.between(1.6, 3.6)
    ozmask = ~izmask

    df = df.assign(iz_swing=
                   np.where(df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                            & izmask, 1, 0))
    df = df.assign(iz_miss=
                   np.where(df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                            & izmask, 1, 0))
    df = df.assign(oz_swing=
                   np.where(df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                            & ozmask, 1, 0))
    df = df.assign(oz_miss=
                   np.where(df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                                            & ozmask, 1, 0))
    sub_df = df.loc[df.batter == batter]
    tab = sub_df.pivot_table(index='game_date',
                             values=['swing', 'miss', 'iz_swing', 'iz_miss', 'oz_swing', 'oz_miss'],
                             aggfunc='sum'
                            )
    tab = tab.assign(raw_num = sub_df.groupby('game_date').count().speed)
    tab = tab.assign(iz_raw_num = sub_df.loc[izmask].groupby('game_date').count().speed)
    tab = tab.assign(oz_raw_num = sub_df.loc[ozmask].groupby('game_date').count().speed)
    
    tab = tab.fillna(0)
    result_sum = []
    
    raw_num_sum = iz_raw_num_sum = oz_raw_num_sum = 0
    swing_sum = iz_swing_sum = oz_swing_sum = 0
    miss_sum = iz_miss_sum = oz_miss_sum = 0
    
    fig, a1 = plt.subplots(figsize=(3, 3), dpi=144, facecolor='white')
    if (ma_term <= 1) or (ma_term >= len(tab.index)):
        for i in tab.index:
            raw_num_sum += tab.loc[i].raw_num
            swing_sum += tab.loc[i].swing
            miss_sum += tab.loc[i].miss
            iz_raw_num_sum += tab.loc[i].iz_raw_num
            iz_swing_sum += tab.loc[i].iz_swing
            iz_miss_sum += tab.loc[i].iz_miss
            oz_raw_num_sum += tab.loc[i].oz_raw_num
            oz_swing_sum += tab.loc[i].oz_swing
            oz_miss_sum += tab.loc[i].oz_miss
            
            #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
            
            result_sum.append([swing_sum/raw_num_sum*100,
                               (1-miss_sum/swing_sum)*100,
                               iz_swing_sum/iz_raw_num_sum*100,
                               (1-iz_miss_sum/iz_swing_sum)*100,
                               oz_swing_sum/oz_raw_num_sum*100,
                               (1-oz_miss_sum/oz_swing_sum)*100])
            
        fig.suptitle(f'{batter}\'s Cumulative Average', fontsize=10)
    else:
        term = ma_term - 1

        for i in range(0, len(tab.index) - term):
            raw_num_sum = tab.loc[tab.index[i]:tab.index[i+term]].raw_num.sum()
            swing_sum = tab.loc[tab.index[i]:tab.index[i+term]].swing.sum()
            miss_sum = tab.loc[tab.index[i]:tab.index[i+term]].miss.sum()
            iz_raw_num_sum = tab.loc[tab.index[i]:tab.index[i+term]].iz_raw_num.sum()
            iz_swing_sum = tab.loc[tab.index[i]:tab.index[i+term]].iz_swing.sum()
            iz_miss_sum = tab.loc[tab.index[i]:tab.index[i+term]].iz_miss.sum()
            oz_raw_num_sum = tab.loc[tab.index[i]:tab.index[i+term]].oz_raw_num.sum()
            oz_swing_sum = tab.loc[tab.index[i]:tab.index[i+term]].oz_swing.sum()
            oz_miss_sum = tab.loc[tab.index[i]:tab.index[i+term]].oz_miss.sum()
            
            #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
            
            result_sum.append([swing_sum/raw_num_sum*100,
                               (1-miss_sum/swing_sum)*100,
                               iz_swing_sum/iz_raw_num_sum*100,
                               (1-iz_miss_sum/iz_swing_sum)*100,
                               oz_swing_sum/oz_raw_num_sum*100,
                               (1-oz_miss_sum/oz_swing_sum)*100])
            
        fig.suptitle(f'{batter}\'s {ma_term}-Game Rolling Average')

    rs = np.array(result_sum)

    a1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d%%'))

    min_y = 10000
    max_y = -10000
    a2 = a3 = a4 = a5 = a6 = None

    #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
    
    if options[0] is True:
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a1.plot(tab.index, rs[:, 0], label='Swing%', color='k')
        else:
            a1.plot(tab.index[term:], rs[:, 0], label='Swing%', color='k')
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_yticks([])
        max_y = max(max_y, rs[:, 0].max())
        min_y = min(min_y, rs[:, 0].min())
    
    if options[1] is True:
        a2 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a2.plot(tab.index, rs[:, 1], label='Contact%', color='dimgrey', linestyle='--')
        else:
            a2.plot(tab.index[term:], rs[:, 1], label='Contact%', color='dimgrey', linestyle='--')
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_yticks([])
        if max_y < rs[:, 1].max():
            max_y = rs[:, 1].max()
        if min_y > rs[:, 1].min():
            min_y = rs[:, 1].min()

    if options[2] is True:
        a3 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a3.plot(tab.index, rs[:, 2], label='Z-Swing%', color='blue')
        else:
            a3.plot(tab.index[term:], rs[:, 2], label='Z-Swing%', color='blue')
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_yticks([])
        if max_y < rs[:, 2].max():
            max_y = rs[:, 2].max()
        if min_y > rs[:, 2].min():
            min_y = rs[:, 2].min()
    
    if options[3] is True:
        a4 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a4.plot(tab.index, rs[:, 3], label='Z-Contact%', color='lightskyblue', linestyle='--')
        else:
            a4.plot(tab.index[term:], rs[:, 3], label='Z-Contact%', color='lightskyblue', linestyle='--')
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_yticks([])
        if max_y < rs[:, 3].max():
            max_y = rs[:, 3].max()
        if min_y > rs[:, 3].min():
            min_y = rs[:, 3].min()
    
    if options[4] is True:
        a5 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a5.plot(tab.index, rs[:, 4], label='O-Swing%', color='green')
        else:
            a5.plot(tab.index[term:], rs[:, 4], label='O-Swing%', color='green')
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_yticks([])
        if max_y < rs[:, 4].max():
            max_y = rs[:, 4].max()
        if min_y > rs[:, 4].min():
            min_y = rs[:, 4].min()
    
    if options[5] is True:
        a6 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a6.plot(tab.index, rs[:, 5], label='O-Contact%', color='lime', linestyle='--')
        else:
            a6.plot(tab.index[term:], rs[:, 5], label='O-Contact%', color='lime', linestyle='--')
        a6.xaxis.set_major_formatter(datesFmt)
        a6.grid(True)
        a6.set_yticks([])
        if max_y < rs[:, 5].max():
            max_y = rs[:, 5].max()
        if min_y > rs[:, 5].min():
            min_y = rs[:, 5].min()
    
    a1.set_ylim([max(0, min_y-5), max_y+5])
    a1.set_ylabel('')
    if a2 is not None:
        a2.set_ylim([max(0, min_y-5), max_y+5])
        a2.set_ylabel('')
    if a3 is not None:
        a3.set_ylim([max(0, min_y-5), max_y+5])
        a3.set_ylabel('')
    if a4 is not None:
        a4.set_ylim([max(0, min_y-5), max_y+5])
        a4.set_ylabel('')
    if a5 is not None:
        a5.set_ylim([max(0, min_y-5), max_y+5])
        a5.set_ylabel('')
    if a6 is not None:
        a6.set_ylim([max(0, min_y-5), max_y+5])
        a6.set_ylabel('')
    
    fig.legend(loc='lower center',
               ncol=min(np.array(options).sum(), 3),
               fontsize='xx-small',
               bbox_to_anchor=(0.5, -0.01))

    a1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    a1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    a1.grid(True)
    a1.tick_params(labelsize='x-small')

    return fig


def graph_pitching_result(df, pitcher, ma_term=0, options=[True, True, True, True, True, True, True]):
    # K%, BB%, BABIP, OAVG, OOBP, OSLG, OOPS
    datesFmt = mdates.DateFormatter('%m-%d')
    
    if df.game_date.dtype == np.int64:
        df = df.assign(game_date = pd.to_datetime(df.game_date, format='%Y%m%d').dt.date)
    
    df = df.assign(pa=np.where(df.pa_result != 'None', 1, 0))
    df = df.assign(ab=np.where(df.pa_result.isin(['1루타', '내야안타', '안타', '2루타', '3루타', '홈런', '희생번트 야수선택',
                                                  '삼진', '실책', '필드 아웃', '포스 아웃', '병살타', '타구맞음 아웃',
                                                  '낫아웃 폭투', '희생번트 실책', '야수선택', '낫아웃 포일', '삼중살']),
                               1, 0))
    
    df = df.assign(single=np.where(df.pa_result.isin(['1루타', '내야안타', '안타']), 1, 0))
    df = df.assign(double=np.where(df.pa_result == '2루타', 1, 0))
    df = df.assign(triple=np.where(df.pa_result == '3루타', 1, 0))
    df = df.assign(hr=np.where(df.pa_result == '홈런', 1, 0))
    
    df = df.assign(sh=np.where(df.pa_result == '희생번트', 1, 0))
    df = df.assign(sf=np.where(df.pa_result == '희생플라이', 1, 0))
    df = df.assign(so=np.where(df.pa_result.isin(['삼진', '낫아웃 폭투', '낫아웃 포일']), 1, 0))
    df = df.assign(bb=np.where(df.pa_result.isin(['볼넷', '고의4구']), 1, 0))
    df = df.assign(hbp=np.where(df.pa_result == '몸에 맞는 볼', 1, 0))
    
    df = df.assign(hit=df.single + df.double + df.triple + df.hr)
    df = df.assign(tb=df.single + df.double*2 + df.triple*3 + df.hr*4)
    
    sub_df = df.loc[df.pitcher == pitcher]
    tab = sub_df.pivot_table(index='game_date',
                             values=['hit', 'tb', 'ab', 'pa', 'hr', 'sh', 'sf', 'so', 'bb', 'hbp'],
                             aggfunc='sum'
                            )
    
    #babip = (hit-hr) / (ab-so-hr+sf)
    tab = tab.fillna(0)
    result_sum = []
    
    hit_sum = tb_sum = 0
    ab_sum = pa_sum = 0
    hr_sum = sh_sum = sf_sum = 0
    so_sum = bb_sum = hbp_sum = 0

    fig, a1 = plt.subplots(figsize=(3, 3), dpi=144, facecolor='white')
    if (ma_term <= 1) or (ma_term >= len(tab.index)):
        for i in tab.index:
            hit_sum += tab.loc[i].hit
            tb_sum += tab.loc[i].tb
            ab_sum += tab.loc[i].ab
            pa_sum += tab.loc[i].pa
            hr_sum += tab.loc[i].hr
            sh_sum += tab.loc[i].sh
            sf_sum += tab.loc[i].sf
            so_sum += tab.loc[i].so
            bb_sum += tab.loc[i].bb
            hbp_sum += tab.loc[i].hbp
            
            # avg, obp, slg, ops, babip
            # OAVG, OOBP, OSLG, OOPS, BABIP, K%, BB% 
            
            result_sum.append([hit_sum/ab_sum, # oavg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum), # oobp
                               tb_sum/ab_sum, # oslg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum) + tb_sum/ab_sum, # oops
                               (hit_sum-hr_sum)/(ab_sum-so_sum-hr_sum+sf_sum), # babip
                               so_sum / pa_sum, # K%
                               bb_sum / pa_sum # BB%
                              ])
            
        fig.suptitle(f'{pitcher}\'s Cumulative Average', fontsize=10)
    else:
        term = ma_term - 1

        for i in range(0, len(tab.index) - term):
            hit_sum = tab.loc[tab.index[i]:tab.index[i+term]].hit.sum()
            tb_sum = tab.loc[tab.index[i]:tab.index[i+term]].tb.sum()
            ab_sum = tab.loc[tab.index[i]:tab.index[i+term]].ab.sum()
            pa_sum = tab.loc[tab.index[i]:tab.index[i+term]].pa.sum()
            hr_sum = tab.loc[tab.index[i]:tab.index[i+term]].hr.sum()
            sh_sum = tab.loc[tab.index[i]:tab.index[i+term]].sh.sum()
            sf_sum = tab.loc[tab.index[i]:tab.index[i+term]].sf.sum()
            so_sum = tab.loc[tab.index[i]:tab.index[i+term]].so.sum()
            bb_sum = tab.loc[tab.index[i]:tab.index[i+term]].bb.sum()
            hbp_sum = tab.loc[tab.index[i]:tab.index[i+term]].hbp.sum()
            
            # avg, obp, slg, ops, babip
            
            result_sum.append([hit_sum/ab_sum, # avg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum), # oobp
                               tb_sum/ab_sum, # slg
                               (hit_sum+bb_sum+hbp_sum)/(ab_sum+bb_sum+hbp_sum+sf_sum) + tb_sum/ab_sum, # oops
                               (hit_sum-hr_sum)/(ab_sum-so_sum-hr_sum+sf_sum), # babip
                               so_sum / pa_sum, # K%
                               bb_sum / pa_sum # BB%
                              ])
            
        fig.suptitle(f'{pitcher}\'s {ma_term}-Game Rolling Average')

    rs = np.array(result_sum)

    a1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    min_y = 10000
    max_y = -10000
    a2 = a3 = a4 = a5 = a6 = a7 = None

    if options[0] is True:
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a1.plot(tab.index, rs[:, 0], label='AVG', color='k')
        else:
            a1.plot(tab.index[term:], rs[:, 0], label='AVG', color='k')
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_yticks([])
        max_y = max(max_y, rs[:, 0].max())
        min_y = min(min_y, rs[:, 0].min())

    if options[1] is True:
        a2 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a2.plot(tab.index, rs[:, 1], label='OBP', color='blueviolet', linestyle=':', linewidth=2)
        else:
            a2.plot(tab.index[term:], rs[:, 1], label='OBP', color='blueviolet', linestyle=':', linewidth=2)
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_yticks([])
        if max_y < rs[:, 1].max():
            max_y = rs[:, 1].max()
        if min_y > rs[:, 1].min():
            min_y = rs[:, 1].min()

    if options[2] is True:
        a3 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a3.plot(tab.index, rs[:, 2], label='SLG', color='orangered', marker='o', markersize='2.5')
        else:
            a3.plot(tab.index[term:], rs[:, 2], label='SLG', color='orangered', marker='o', markersize='2.5')
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_yticks([])
        if max_y < rs[:, 2].max():
            max_y = rs[:, 2].max()
        if min_y > rs[:, 2].min():
            min_y = rs[:, 2].min()

    if options[3] is True:
        a4 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a4.plot(tab.index, rs[:, 3], label='OPS', color='royalblue')
        else:
            a4.plot(tab.index[term:], rs[:, 3], label='OPS', color='royalblue')
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_yticks([])
        if max_y < rs[:, 3].max():
            max_y = rs[:, 3].max()
        if min_y > rs[:, 3].min():
            min_y = rs[:, 3].min()

    if options[4] is True:
        a5 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a5.plot(tab.index, rs[:, 4], label='BABIP', color='gold', linestyle='--')
        else:
            a5.plot(tab.index[term:], rs[:, 4], label='BABIP', color='gold', linestyle='--')
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_yticks([])
        if max_y < rs[:, 4].max():
            max_y = rs[:, 4].max()
        if min_y > rs[:, 4].min():
            min_y = rs[:, 4].min()
            
    if options[5] is True:
        a6 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a6.plot(tab.index, rs[:, 5], label='K%', color='green', linestyle='--')
        else:
            a6.plot(tab.index[term:], rs[:, 5], label='K%', color='green', linestyle='--')
        a6.xaxis.set_major_formatter(datesFmt)
        a6.grid(True)
        a6.set_yticks([])
        if max_y < rs[:, 5].max():
            max_y = rs[:, 5].max()
        if min_y > rs[:, 5].min():
            min_y = rs[:, 5].min()
            
    if options[6] is True:
        a7 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a7.plot(tab.index, rs[:, 6], label='BB%', color='lime', linestyle='--')
        else:
            a7.plot(tab.index[term:], rs[:, 6], label='BB%', color='lime', linestyle='--')
        a7.xaxis.set_major_formatter(datesFmt)
        a7.grid(True)
        a7.set_yticks([])
        if max_y < rs[:, 6].max():
            max_y = rs[:, 6].max()
        if min_y > rs[:, 6].min():
            min_y = rs[:, 6].min()

    a1.set_ylim([max(0, min_y-0.025), max_y+0.025])
    a1.set_ylabel('')
    if a2 is not None:
        a2.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a2.set_ylabel('')
    if a3 is not None:
        a3.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a3.set_ylabel('')
    if a4 is not None:
        a4.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a4.set_ylabel('')
    if a5 is not None:
        a5.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a5.set_ylabel('')
    if a6 is not None:
        a6.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a6.set_ylabel('')
    if a7 is not None:
        a7.set_ylim([max(0, min_y-0.025), max_y+0.025])
        a7.set_ylabel('')
        
    fig.legend(loc='lower center', ncol=min(np.array(options).sum(), 4),
               fontsize='xx-small', bbox_to_anchor=(0.5, -0.01))

    a1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    a1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    a1.grid(True)
    a1.tick_params(labelsize='x-small')

    return fig


def graph_pitcher_plate_discipline(df, pitcher, ma_term=0, options=[True, True, True, True, True, True]):
    datesFmt = mdates.DateFormatter('%m-%d')
    
    if df.game_date.dtype == np.int64:
        df = df.assign(game_date = pd.to_datetime(df.game_date, format='%Y%m%d').dt.date)
    
    if df.px.dtypes == np.object:
        df.loc[:, 'px'] = pd.to_numeric(df.px, errors='coerce')
    if df.isnull().any().px is False:
        df = df.drop(df.loc[df.px.isnull()].index)

    if df.pz.dtypes == np.object:
        df.loc[:, 'pz'] = pd.to_numeric(df.pz, errors='coerce')
    if df.isnull().any().pz is False:
        df = df.drop(df.loc[df.pz.isnull()].index)
        
    df = df.assign(swing=np.where(df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울']), 1, 0))
    df = df.assign(miss=np.where(df.pitch_result.isin(['번트헛스윙', '헛스윙']), 1, 0))

    izmask = df.px.between(-10/12, 10/12) & df.pz.between(1.6, 3.6)
    ozmask = ~izmask

    df = df.assign(iz_swing=
                   np.where(df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                            & izmask, 1, 0))
    df = df.assign(iz_miss=
                   np.where(df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                            & izmask, 1, 0))
    df = df.assign(oz_swing=
                   np.where(df.pitch_result.isin(['타격', '번트파울', '번트헛스윙', '헛스윙', '파울'])
                            & ozmask, 1, 0))
    df = df.assign(oz_miss=
                   np.where(df.pitch_result.isin(['번트헛스윙', '헛스윙'])
                                            & ozmask, 1, 0))
    sub_df = df.loc[df.pitcher == pitcher]
    tab = sub_df.pivot_table(index='game_date',
                             values=['swing', 'miss', 'iz_swing', 'iz_miss', 'oz_swing', 'oz_miss'],
                             aggfunc='sum'
                            )
    tab = tab.assign(raw_num = sub_df.groupby('game_date').count().speed)
    tab = tab.assign(iz_raw_num = sub_df.loc[izmask].groupby('game_date').count().speed)
    tab = tab.assign(oz_raw_num = sub_df.loc[ozmask].groupby('game_date').count().speed)
    
    tab = tab.fillna(0)
    result_sum = []
    
    raw_num_sum = iz_raw_num_sum = oz_raw_num_sum = 0
    swing_sum = iz_swing_sum = oz_swing_sum = 0
    miss_sum = iz_miss_sum = oz_miss_sum = 0
    
    fig, a1 = plt.subplots(figsize=(3, 3), dpi=144, facecolor='white')
    if (ma_term <= 1) or (ma_term >= len(tab.index)):
        for i in tab.index:
            raw_num_sum += tab.loc[i].raw_num
            swing_sum += tab.loc[i].swing
            miss_sum += tab.loc[i].miss
            iz_raw_num_sum += tab.loc[i].iz_raw_num
            iz_swing_sum += tab.loc[i].iz_swing
            iz_miss_sum += tab.loc[i].iz_miss
            oz_raw_num_sum += tab.loc[i].oz_raw_num
            oz_swing_sum += tab.loc[i].oz_swing
            oz_miss_sum += tab.loc[i].oz_miss
            
            #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
            
            result_sum.append([swing_sum/raw_num_sum*100,
                               (1-miss_sum/swing_sum)*100,
                               iz_swing_sum/iz_raw_num_sum*100,
                               (1-iz_miss_sum/iz_swing_sum)*100,
                               oz_swing_sum/oz_raw_num_sum*100,
                               (1-oz_miss_sum/oz_swing_sum)*100])
            
        fig.suptitle(f'{pitcher}\'s Cumulative Average', fontsize=10)
    else:
        term = ma_term - 1

        for i in range(0, len(tab.index) - term):
            raw_num_sum = tab.loc[tab.index[i]:tab.index[i+term]].raw_num.sum()
            swing_sum = tab.loc[tab.index[i]:tab.index[i+term]].swing.sum()
            miss_sum = tab.loc[tab.index[i]:tab.index[i+term]].miss.sum()
            iz_raw_num_sum = tab.loc[tab.index[i]:tab.index[i+term]].iz_raw_num.sum()
            iz_swing_sum = tab.loc[tab.index[i]:tab.index[i+term]].iz_swing.sum()
            iz_miss_sum = tab.loc[tab.index[i]:tab.index[i+term]].iz_miss.sum()
            oz_raw_num_sum = tab.loc[tab.index[i]:tab.index[i+term]].oz_raw_num.sum()
            oz_swing_sum = tab.loc[tab.index[i]:tab.index[i+term]].oz_swing.sum()
            oz_miss_sum = tab.loc[tab.index[i]:tab.index[i+term]].oz_miss.sum()
            
            #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
            
            result_sum.append([swing_sum/raw_num_sum*100,
                               (1-miss_sum/swing_sum)*100,
                               iz_swing_sum/iz_raw_num_sum*100,
                               (1-iz_miss_sum/iz_swing_sum)*100,
                               oz_swing_sum/oz_raw_num_sum*100,
                               (1-oz_miss_sum/oz_swing_sum)*100])
            
        fig.suptitle(f'{pitcher}\'s {ma_term}-Game Rolling Average')

    rs = np.array(result_sum)

    a1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d%%'))

    min_y = 10000
    max_y = -10000
    a2 = a3 = a4 = a5 = a6 = None

    #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
    
    if options[0] is True:
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a1.plot(tab.index, rs[:, 0], label='Swing%', color='k')
        else:
            a1.plot(tab.index[term:], rs[:, 0], label='Swing%', color='k')
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_yticks([])
        max_y = max(max_y, rs[:, 0].max())
        min_y = min(min_y, rs[:, 0].min())
    
    if options[1] is True:
        a2 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a2.plot(tab.index, rs[:, 1], label='Contact%', color='dimgrey', linestyle='--')
        else:
            a2.plot(tab.index[term:], rs[:, 1], label='Contact%', color='dimgrey', linestyle='--')
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_yticks([])
        if max_y < rs[:, 1].max():
            max_y = rs[:, 1].max()
        if min_y > rs[:, 1].min():
            min_y = rs[:, 1].min()

    if options[2] is True:
        a3 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a3.plot(tab.index, rs[:, 2], label='Z-Swing%', color='blue')
        else:
            a3.plot(tab.index[term:], rs[:, 2], label='Z-Swing%', color='blue')
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_yticks([])
        if max_y < rs[:, 2].max():
            max_y = rs[:, 2].max()
        if min_y > rs[:, 2].min():
            min_y = rs[:, 2].min()
    
    if options[3] is True:
        a4 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a4.plot(tab.index, rs[:, 3], label='Z-Contact%', color='lightskyblue', linestyle='--')
        else:
            a4.plot(tab.index[term:], rs[:, 3], label='Z-Contact%', color='lightskyblue', linestyle='--')
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_yticks([])
        if max_y < rs[:, 3].max():
            max_y = rs[:, 3].max()
        if min_y > rs[:, 3].min():
            min_y = rs[:, 3].min()
    
    if options[4] is True:
        a5 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a5.plot(tab.index, rs[:, 4], label='O-Swing%', color='green')
        else:
            a5.plot(tab.index[term:], rs[:, 4], label='O-Swing%', color='green')
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_yticks([])
        if max_y < rs[:, 4].max():
            max_y = rs[:, 4].max()
        if min_y > rs[:, 4].min():
            min_y = rs[:, 4].min()
    
    if options[5] is True:
        a6 = a1.twinx()
        if (ma_term <= 1) or (ma_term >= len(tab.index)):
            a6.plot(tab.index, rs[:, 5], label='O-Contact%', color='lime', linestyle='--')
        else:
            a6.plot(tab.index[term:], rs[:, 5], label='O-Contact%', color='lime', linestyle='--')
        a6.xaxis.set_major_formatter(datesFmt)
        a6.grid(True)
        a6.set_yticks([])
        if max_y < rs[:, 5].max():
            max_y = rs[:, 5].max()
        if min_y > rs[:, 5].min():
            min_y = rs[:, 5].min()
    
    a1.set_ylim([max(0, min_y-5), max_y+5])
    a1.set_ylabel('')
    if a2 is not None:
        a2.set_ylim([max(0, min_y-5), max_y+5])
        a2.set_ylabel('')
    if a3 is not None:
        a3.set_ylim([max(0, min_y-5), max_y+5])
        a3.set_ylabel('')
    if a4 is not None:
        a4.set_ylim([max(0, min_y-5), max_y+5])
        a4.set_ylabel('')
    if a5 is not None:
        a5.set_ylim([max(0, min_y-5), max_y+5])
        a5.set_ylabel('')
    if a6 is not None:
        a6.set_ylim([max(0, min_y-5), max_y+5])
        a6.set_ylabel('')
    
    fig.legend(loc='lower center',
               ncol=min(np.array(options).sum(), 3),
               fontsize='xx-small',
               bbox_to_anchor=(0.5, -0.01))

    a1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    a1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    a1.grid(True)
    a1.tick_params(labelsize='x-small')

    return fig


def interactive_batting_graph(df):
    batters = df.batter.drop_duplicates().get_values().tolist()
    batters.sort()
    batter = batters[0]
    term= df.game_date.drop_duplicates().shape[0]

    batterSelect = widgets.Dropdown(
        options=batters,
        index=0,
        description='Batter:'
    )

    termSelectSlider = widgets.IntSlider(
        value=0,
        min=0,
        max=df.loc[df.batter == batterSelect.value].game_date.drop_duplicates().shape[0],
        step=1,
        description='Term:',
        continuous_update=True
    )
    
    def onSelectBatter(b):
        termSelectSlider.max = df.loc[df.batter == batterSelect.value].game_date.drop_duplicates().shape[0]
        
    batterSelect.observe(onSelectBatter, names='value')
    
    words = ['AVG', 'OBP', 'SLG', 'OPS', 'BABIP']
    items = [widgets.ToggleButton(description=w, value=True, layout=widgets.Layout(width='80px')) for w in words]
    statSelectButton = widgets.HBox(items)
    
    options = [True, True, True, True, True]

    interactive_batting_graph.fig = graph_batting_result(df, batterSelect.value, termSelectSlider.value)
    
    updateButton = widgets.Button(
        disabled=False,
        description='Update',
        icon='check',
        layout=widgets.Layout(width='80px')
    )
    
    def onClickUpdateButton(b):
        clear_output()
        display(batterSelect, termSelectSlider, statSelectButton, updateButton)
        
        options = [it.value for it in items]
        
        interactive_batting_graph.fig = graph_batting_result(df, batterSelect.value, termSelectSlider.value, options)
        display(interactive_batting_graph.fig)
        plt.close(interactive_batting_graph.fig)

    updateButton.on_click(onClickUpdateButton)

    display(batterSelect, termSelectSlider, statSelectButton, updateButton)
    display(interactive_batting_graph.fig)
    plt.close(interactive_batting_graph.fig)

    
def interactive_batter_discipline_graph(df):
    batters = df.batter.drop_duplicates().get_values().tolist()
    batters.sort()
    batter = batters[0]
    term= df.game_date.drop_duplicates().shape[0]

    batterSelect = widgets.Dropdown(
        options=batters,
        index=0,
        description='Batter:'
    )

    termSelectSlider = widgets.IntSlider(
        value=0,
        min=0,
        max=df.loc[df.batter == batterSelect.value].game_date.drop_duplicates().shape[0],
        step=1,
        description='Term:',
        continuous_update=True
    )
    
    def onSelectBatter(b):
        termSelectSlider.max = df.loc[df.batter == batterSelect.value].game_date.drop_duplicates().shape[0]
        
    batterSelect.observe(onSelectBatter, names='value')
    
    words = ['Swing%', 'Contact%', 'Z-Swing%', 'Z-Contact%', 'O-Swing%', 'O-Contact%']
    
    items = [widgets.ToggleButton(description=w, value=True, layout=widgets.Layout(width='100px')) for w in words]
    box1 = widgets.HBox([items[0], items[2], items[4]])
    box2 = widgets.HBox([items[1], items[3], items[5]])
    statSelectButton = widgets.VBox([box1, box2])
    
    options = [True, True, True, True, True, True]

    interactive_batter_discipline_graph.fig = graph_batter_plate_discipline(df, batterSelect.value, termSelectSlider.value)
    
    updateButton = widgets.Button(
        disabled=False,
        description='Update',
        icon='check',
        layout=widgets.Layout(width='80px')
    )
    
    def onClickUpdateButton(b):
        clear_output()
        display(batterSelect, termSelectSlider, statSelectButton, updateButton)
        
        options = [it.value for it in items]
        
        interactive_batter_discipline_graph.fig = graph_batter_plate_discipline(df, batterSelect.value, termSelectSlider.value, options)
        display(interactive_batter_discipline_graph.fig)
        plt.close(interactive_batter_discipline_graph.fig)

    updateButton.on_click(onClickUpdateButton)

    display(batterSelect, termSelectSlider, statSelectButton, updateButton)
    display(interactive_batter_discipline_graph.fig)
    plt.close(interactive_batter_discipline_graph.fig)


def interactive_pitching_graph(df):
    pitchers = df.pitcher.drop_duplicates().get_values().tolist()
    pitchers.sort()
    pitcher = pitchers[0]
    term= df.game_date.drop_duplicates().shape[0]

    pitcherSelect = widgets.Dropdown(
        options=pitchers,
        index=0,
        description='pitcher:'
    )

    termSelectSlider = widgets.IntSlider(
        value=0,
        min=0,
        max=df.loc[df.pitcher == pitcherSelect.value].game_date.drop_duplicates().shape[0],
        step=1,
        description='Term:',
        continuous_update=True
    )
    
    def onSelectPitcher(b):
        termSelectSlider.max = df.loc[df.pitcher == pitcherSelect.value].game_date.drop_duplicates().shape[0]
        
    pitcherSelect.observe(onSelectPitcher, names='value')
    
    words = ['AVG', 'OBP', 'SLG', 'OPS', 'BABIP', 'K%', 'BB%']
    items = [widgets.ToggleButton(description=w, value=True, layout=widgets.Layout(width='100px')) for w in words]
    box1 = widgets.HBox([items[0], items[1], items[2], items[3]])
    box2 = widgets.HBox([items[4], items[5], items[6]])
    statSelectButton = widgets.VBox([box1, box2])
    
    options = [True, True, True, True, True, True, True]

    interactive_pitching_graph.fig = graph_pitching_result(df, pitcherSelect.value, termSelectSlider.value)
    
    updateButton = widgets.Button(
        disabled=False,
        description='Update',
        icon='check',
        layout=widgets.Layout(width='80px')
    )
    
    def onClickUpdateButton(b):
        clear_output()
        display(pitcherSelect, termSelectSlider, statSelectButton, updateButton)
        
        options = [it.value for it in items]
        
        interactive_pitching_graph.fig = graph_pitching_result(df, pitcherSelect.value, termSelectSlider.value, options)
        display(interactive_pitching_graph.fig)
        plt.close(interactive_pitching_graph.fig)

    updateButton.on_click(onClickUpdateButton)

    display(pitcherSelect, termSelectSlider, statSelectButton, updateButton)
    display(interactive_pitching_graph.fig)
    plt.close(interactive_pitching_graph.fig)


def interactive_pitcher_discipline_graph(df):
    pitchers = df.pitcher.drop_duplicates().get_values().tolist()
    pitchers.sort()
    pitcher = pitchers[0]
    term= df.game_date.drop_duplicates().shape[0]

    pitcherSelect = widgets.Dropdown(
        options=pitchers,
        index=0,
        description='pitcher:'
    )

    termSelectSlider = widgets.IntSlider(
        value=0,
        min=0,
        max=df.loc[df.pitcher == pitcherSelect.value].game_date.drop_duplicates().shape[0],
        step=1,
        description='Term:',
        continuous_update=True
    )
    
    def onSelectpitcher(b):
        termSelectSlider.max = df.loc[df.pitcher == pitcherSelect.value].game_date.drop_duplicates().shape[0]
        
    pitcherSelect.observe(onSelectpitcher, names='value')
    
    words = ['Swing%', 'Contact%', 'Z-Swing%', 'Z-Contact%', 'O-Swing%', 'O-Contact%']
    
    items = [widgets.ToggleButton(description=w, value=True, layout=widgets.Layout(width='100px')) for w in words]
    box1 = widgets.HBox([items[0], items[2], items[4]])
    box2 = widgets.HBox([items[1], items[3], items[5]])
    statSelectButton = widgets.VBox([box1, box2])
    
    #options = [True, True, True, True, True, True]

    interactive_pitcher_discipline_graph.fig = graph_pitcher_plate_discipline(df, pitcherSelect.value, termSelectSlider.value)
    
    updateButton = widgets.Button(
        disabled=False,
        description='Update',
        icon='check',
        layout=widgets.Layout(width='80px')
    )
    
    def onClickUpdateButton(b):
        clear_output()
        display(pitcherSelect, termSelectSlider, statSelectButton, updateButton)
        
        options = [it.value for it in items]
        
        interactive_pitcher_discipline_graph.fig = graph_pitcher_plate_discipline(df, pitcherSelect.value, termSelectSlider.value, options)
        display(interactive_pitcher_discipline_graph.fig)
        plt.close(interactive_pitcher_discipline_graph.fig)

    updateButton.on_click(onClickUpdateButton)

    display(pitcherSelect, termSelectSlider, statSelectButton, updateButton)
    display(interactive_pitcher_discipline_graph.fig)
    plt.close(interactive_pitcher_discipline_graph.fig)
