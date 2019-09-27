# coding: utf-8

import sys, csv, json, collections, datetime, os, numbers, importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.random
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib import font_manager as fm, rc
from IPython.display import HTML
from IPython.display import display, Audio
import pandas as pd
import seaborn as sns
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
import ipywidgets as widgets
from IPython.display import clear_output
from scipy import stats

if importlib.util.find_spec('pygam') is not None:
    from pygam import LogisticGAM

Results = Enum('Results', '볼 스트라이크 헛스윙 파울 타격 번트파울 번트헛스윙')
Stuffs = Enum('Stuffs', '직구 슬라이더 포크 체인지업 커브 투심 싱커 커터 너클볼')
Colors = {'볼': '#3245ef', '스트라이크': '#ef2926', '헛스윙':'#1a1b1c', '파울':'#edf72c', '타격':'#8348d1', '번트파울':'#edf72c', '번트헛스윙':'#1a1b1c' }


def fmt(x, pos):
    return r'{}%'.format(int(x*100))


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
                rc('font', family='NanumSquareRound')
        else:
            rc('font', family='NanumSquareRound')
        

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
    if ('pfx_x_raw' in df.keys()) is True:
        df = df.assign(pfx_x_raw = pd.to_numeric(df.pfx_x_raw, errors='coerce'))
    if ('pfx_z_raw' in df.keys()) is True:
        df = df.assign(pfx_z_raw = pd.to_numeric(df.pfx_z_raw, errors='coerce'))
    if ('x0' in df.keys()) is True:
        df = df.assign(x0 = pd.to_numeric(df.x0, errors='coerce'))
    if ('z0' in df.keys()) is True:
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


def plot_strike_calls(df, title=None, show_pitch_number=False):
    return plot_by_call(df, title, calls=['스트라이크', '볼'], legends=True, show_pitch_number=show_pitch_number)


def plot_by_call(df, title=None, calls=None, legends=True, show_pitch_number=False, dpi=80, ax=None):
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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5), dpi=dpi, facecolor='#898f99')
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
        ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*dpi, label=c, cmap='set1', zorder=0)

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
        ax.set_title(title, fontsize='xx-large', color='white', weight='bold', horizontalalignment='center')
    
    plt.axis('off')
    ax.autoscale_view('tight')
    
    if legends is True:
        ax.legend(loc='lower center', ncol=2, fontsize='medium')
        
    return fig, ax


def plot_by_pitch_type(df, title=None, pitch_types=None, legends=True, show_pitch_number=False, dpi=80, ax=None):
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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5), dpi=dpi, facecolor='#898f99')
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
    ax = fig.add_subplot(231)
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
    ax = fig.add_subplot(232)
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
    ax = fig.add_subplot(233)
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
    ax = fig.add_subplot(234)
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
    ax = fig.add_subplot(235)
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
    ax = fig.add_subplot(236)
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


def plot_contour_balls(df, title=None, dpi=144, cmap=None, ax=None):
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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4), dpi=dpi, facecolor='white')
    else:
        fig = None

    major_xtick_step = major_ytick_step = 1/2
    minor_xtick_step = minor_ytick_step = 1/10
        
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


def plot_heatmap(df, title=None, dpi=144, cmap=None, ax=None, show_full=False, color=None):
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
    else:
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
    
    major_xtick_step = major_ytick_step = 1/2
    minor_xtick_step = minor_ytick_step = 1/12
    
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


def plot_szone(df, title=None, dpi=144, show_area=False, ax=None):
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
    
    major_xticks = np.linspace(lb, rb, 7)
    minor_xticks = np.linspace(lb, rb, 37)
    
    major_yticks = np.linspace(1, 4, 7)
    minor_yticks = np.linspace(1, 4, 37)

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

def plot_by_proba(df, title=None, dpi=144, cmap=None, ax=None):
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4), dpi=dpi, facecolor='white')
    else:
        fig = None
    
    if cmap is None:
        cmap='Reds'
    ax.set_facecolor('#cccccc')
    cs = ax.scatter(df.px, df.pz, alpha=.5, s=np.pi/2*dpi, c=df.proba, cmap=cmap, zorder=0, vmin=0, vmax=1)
    plt.colorbar(cs, format=ticker.FuncFormatter(fmt), spacing='proportional', ax=ax)
    
    major_xtick_step = major_ytick_step = 1/2
    minor_xtick_step = minor_ytick_step = 1/12

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
