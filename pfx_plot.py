# coding: utf-8

import csv
import json
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.random
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import datetime
from matplotlib import font_manager as fm, rc
from IPython.display import HTML
from IPython.display import display
import pandas as pd
import os
from enum import Enum
import numbers
from scipy.ndimage.filters import gaussian_filter

import importlib
if importlib.util.find_spec('pygam') is not None:
    from pygam import LogisticGAM

Results = Enum('Results', '볼 스트라이크 헛스윙 파울 타격 번트파울 번트헛스윙')
Stuffs = Enum('Stuffs', '직구 슬라이더 포크 체인지업 커브 투심 싱커 커터 너클볼')
Colors = {'볼': '#3245ef', '스트라이크': '#ef2926', '헛스윙':'#1a1b1c', '파울':'#edf72c', '타격':'#8348d1', '번트파울':'#edf72c', '번트헛스윙':'#1a1b1c' }

def set_fonts():
    if os.name == 'posix':
        fm.get_fontconfig_fonts()
        font_location = '/Library/Fonts/NanumSquareOTFRegular.otf'
        font_name = fm.FontProperties(fname=font_location).get_name()
        rc('font', family=font_name)
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
    df.loc[:, 'speed'] = pd.to_numeric(df.speed, errors='coerce')
    df.loc[:, 'px'] = pd.to_numeric(df.px, errors='coerce')
    df.loc[:, 'pz'] = pd.to_numeric(df.pz, errors='coerce')
    df.loc[:, 'sz_top'] = pd.to_numeric(df.sz_top, errors='coerce')
    df.loc[:, 'sz_bot'] = pd.to_numeric(df.sz_bot, errors='coerce')
    df.loc[:, 'pfx_x'] = pd.to_numeric(df.pfx_x, errors='coerce')
    df.loc[:, 'pfx_z'] = pd.to_numeric(df.pfx_z, errors='coerce')
    df.loc[:, 'x0'] = pd.to_numeric(df.x0, errors='coerce')
    df.loc[:, 'z0'] = pd.to_numeric(df.z0, errors='coerce')

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


def plot_by_call(df, title=None, calls=None, legends=True, show_pitch_number=False, print_std=False):
    set_fonts()

    # 단위: 피트; 좌우폭=17인치=17/24피트, 공1개 지름=약 3인치=1/4피트; 공반개=1/8피트
    lb = -1.5  # leftBorder
    rb = +1.5  # rightBorder
    tb = +4.0  # topBorder
    bb = +1.0  # bottomBorder
    
    ll = -17/24  # leftLine
    rl = +17/24  # rightLine
    tl = +3.325  # topLine
    bl = +1.579  # bottomLine
    
    oll = -17/24-1/8  # outerLeftLine
    orl = +17/24+1/8  # outerRightLine
    otl = +3.325+1/8  # outerTopLine
    obl = +1.579-1/8  # outerBottomLine
    
    # 타자 상하 존 경계에 맞춰 표준화하는 경우
    # 타자 상하 존 경계에 맞춰 표준화하는 경우
    # Reference
    # http://tangotiger.com/index.php/site/article/stacast-lab-pitch-zones-heart-shadow-ozone
    
    # 상단 경계 위의 공은 경계선과의 차이를 먼저 계산하고
    # 경계선을 3.5피트로 조정하여 환산.
    # ex) sz_top=3.7, pz=4.3 -> sz_top이 3.5인걸로 가정해 pz_std를 4.1로 보정
    
    # 하단 경계 밑의 공은 존 하단 경계선을 1.5피트로 가정하고
    # 지면(0피트)과 경계선 사이의 상대적인 비율에 맞춰 보정.
    # ex) sz_bot=2, pz=1 -> sz_bot이 1.5인걸로 가정해 pz_std를 0.75로 보정
    
    # 상단-하단 사이, 존 내부의 공은 1.5~3.5 사이의 비율에 맞춰 보정.
    # ex) sz_bot=2, sz_top=5, pz=3.5 -> sz_top=1.5, sz_bot=3.5로 가정해 2.5로 보정
    
    if print_std is True:
        tl = 3.5        # topLine
        bl = 1.5        # bottomLine
        ll = -20/24  # leftLine
        rl = +20/24  # rightLine
        
        df = df.assign(pz_std=np.where(df.pz < df.sz_bot, (df.pz*1.5/df.sz_bot),
                                       np.where(df.pz > df.sz_top, df.pz-(df.sz_top-3.5),
                                                (df.pz - (df.sz_top+df.sz_bot)/2)/(df.sz_top-df.sz_bot)*2+2.5)))

    # 스트라이크, 볼만 표기 -> 서브플롯 1개
    fig, ax = plt.subplots(figsize=(2,2), dpi=150, facecolor='#898f99')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    if title is not None:
        st = fig.suptitle(title, fontsize='medium')
        st.set_color('white')
        st.set_weight('bold')
        st.set_horizontalalignment('center')
    
    if calls is None:
        calls_ = df.pitch_result.drop_duplicates()
        
        for c in calls_:
            f = df.loc[df.pitch_result == c]
            
            if print_std is True:
                ax.scatter(f.px, f.pz_std, color=Colors[c], alpha=.5, s=np.pi*40*72/fig.dpi, label=c)
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, color=Colors[c], alpha=.5, s=np.pi*40*72/fig.dpi, label=c)
                
                if show_pitch_number is True:
                    for i in f.index:
                        if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                            ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
    else:
        if type(calls) is list:
            for call in calls:
                f = df.loc[df.pitch_result == call]
                color = Colors[call]
                
                if print_std is True:
                    ax.scatter(f.px, f.pz_std, color=Colors[call], alpha=.5, s=np.pi*40*72/fig.dpi, label=call)
                    
                    if show_pitch_number is True:
                        for i in f.index:
                            if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                                ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
                else:
                    ax.scatter(f.px, f.pz, color=color, alpha=.5, s=np.pi*40*72/fig.dpi, label=call)

                    if show_pitch_number is True:
                        for i in f.index:
                            if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                                ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
        elif type(calls) is str:
            f = df.loc[df.pitch_result == calls]
            color = Colors[calls]
            
            if print_std is True:
                ax.scatter(f.px, f.pz_std, color=Colors[calls], alpha=.5, s=np.pi*40*72/fig.dpi, label=calls)
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, color=color, alpha=.5, s=np.pi*40*72/fig.dpi, label=calls)

                if show_pitch_number is True:
                    for i in f.index:
                        if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                            ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
        else:
            print()
            print( 'ERROR: call option must be either string or list' )
            exit(1)

    ax.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    ax.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    if print_std is False:
        ax.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
        ax.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

        ax.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
        ax.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    ax.axis( [lb, rb, bb, tb] )

    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.autoscale_view('tight')

    if legends is True:
        ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize='xx-small')
        
    #plt.show()
    return fig, ax


def plot_by_pitch_type(df, title=None, pitch_types=None, legends=True, show_pitch_number=False, print_std=False):
    set_fonts()

    # 단위: 피트; 좌우폭=17인치=17/24피트, 공1개 지름=약 3인치=1/4피트; 공반개=1/8피트
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
    
    # 타자 상하 존 경계에 맞춰 표준화하는 경우
    # Reference
    # http://tangotiger.com/index.php/site/article/stacast-lab-pitch-zones-heart-shadow-ozone
    
    # 상단 경계 위의 공은 경계선과의 차이를 먼저 계산하고
    # 경계선을 3.5피트로 조정하여 환산.
    # ex) sz_top=3.7, pz=4.3 -> sz_top이 3.5인걸로 가정해 pz_std를 4.1로 보정
    
    # 하단 경계 밑의 공은 존 하단 경계선을 1.5피트로 가정하고
    # 지면(0피트)과 경계선 사이의 상대적인 비율에 맞춰 보정.
    # ex) sz_bot=2, pz=1 -> sz_bot이 1.5인걸로 가정해 pz_std를 0.75로 보정
    
    # 상단-하단 사이, 존 내부의 공은 1.5~3.5 사이의 비율에 맞춰 보정.
    # ex) sz_bot=2, sz_top=5, pz=3.5 -> sz_top=1.5, sz_bot=3.5로 가정해 2.5로 보정
    
    if print_std is True:
        tl = 3.5        # topLine
        bl = 1.5        # bottomLine
        ll = -20/24  # leftLine
        rl = +20/24  # rightLine
        
        df = df.assign(pz_std=np.where(df.pz < df.sz_bot, (df.pz*1.5/df.sz_bot),
                                       np.where(df.pz > df.sz_top, df.pz-(df.sz_top-3.5),
                                                (df.pz - (df.sz_top+df.sz_bot)/2)/(df.sz_top-df.sz_bot)*2+2.5)))
    
    fig, ax = plt.subplots(figsize=(2,2), dpi=150, facecolor='#898f99')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    if title is not None:
        st = fig.suptitle(title, fontsize='medium')
        st.set_color('white')
        st.set_weight('bold')
        st.set_horizontalalignment('center')
        
    if pitch_types is None:
        pitch_types_ = df.pitch_type.drop_duplicates()
        
        for p in pitch_types_:
            f = df.loc[df.pitch_type == p]
            
            if print_std is True:
                ax.scatter(f.px, f.pz_std, alpha=.5, s=np.pi*40*72/fig.dpi, label=p, cmap='set1')
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*40*72/fig.dpi, label=p, cmap='set1')
                
                if show_pitch_number is True:
                    for i in f.index:
                        if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                            ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
    else:
        if type(pitch_types) is list:
            for p in pitch_types:
                f = df.loc[df.pitch_type == p]
                
                if print_std is True:
                    ax.scatter(f.px, f.pz_std, alpha=.5, s=np.pi*40*72/fig.dpi, label=p, cmap='set1')
                    
                    if show_pitch_number is True:
                        for i in f.index:
                            if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                                ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
                else:
                    ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*40*72/fig.dpi, label=p, cmap='set1')

                    if show_pitch_number is True:
                        for i in f.index:
                            if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                                ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
        elif type(pitch_types) is str:
            f = df.loc[df.pitch_type == pitch_types]
            
            if print_std is True:
                ax.scatter(f.px, f.pz_std, alpha=.5, s=np.pi*40*72/fig.dpi, label=pitch_types, cmap='set1')
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*40*72/fig.dpi, label=pitch_types, cmap='set1')

                if show_pitch_number is True:
                    for i in f.index:
                        if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                            ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
        else:
            print()
            print( 'ERROR: call option must be either string or list' )
            exit(1)

    ax.plot( [ll, ll], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    ax.plot( [ll, rl], [bl, bl], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color='#f9f9ff', linestyle= '-', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color='#f9f9ff', linestyle= '-', lw=1 )

    if print_std is False:
        ax.plot( [oll, oll], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )
        ax.plot( [orl, orl], [obl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

        ax.plot( [oll, orl], [obl, obl], color='#d0cfd3', linestyle= '-', lw=0.5 )
        ax.plot( [oll, orl], [otl, otl], color='#d0cfd3', linestyle= '-', lw=0.5 )

    ax.axis( [lb, rb, bb, tb] )

    plt.rcParams['axes.unicode_minus'] = False
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    ax.autoscale_view('tight')

    if legends is True:
        ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize='xx-small')
        
    #plt.show()
    return fig, ax

    
# 경기 전체 call
def plot_match_calls(df, title=None):
    set_fonts()
    
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


def plot_contour_balls(df, title=None, print_std=False):
    set_fonts()
    
    lb = -2.0
    rb = +2.0
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24
    
    if print_std is False:
        bb = 0.452
        tb = 4.452
        bl = 1.579
        tl = 3.325
        obl = 1.579-3/24
        otl = 3.325+3/24
    else:
        lb = -2.291
        rb = +2.291
        bb = -2.291
        tb = +2.291
        bl = -1.0
        tl = +1.0
        obl = -1.0-3/24
        otl = +1.0+3/24
        
    from scipy.stats.kde import gaussian_kde
    
    if print_std is False:
        k = gaussian_kde(np.vstack([df.px.values, df.pz.values]))
    else:
        k = gaussian_kde(np.vstack([df.px.values,
                                    (df.pz.values - (df.sz_top.values+df.sz_bot.values)/2) / (df.sz_top.values-df.sz_bot.values)*2]
                                  ))
    
    length = len(df)
    
    xi, yi = np.mgrid[lb:rb:length**0.5*1j,bb:tb:length**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig, ax = plt.subplots(figsize=(4,3), dpi=160)

    if print_std is False:
        cs = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap='YlOrRd' )
    else:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('mycmap', ['#fff6b6', '#fee38b', '#fec561', '#fd9f44', '#fc6c33', '#f03523', '#cf0c1e', '#9f0026'])

        cs = ax.contourf(xi, yi, zi.reshape(xi.shape), cmap=cmap)
    
    cbar = plt.colorbar(cs, format=ticker.FuncFormatter(fmt))

    ax.set_xbound(lb, rb)
    ax.set_ybound(bb, tb)
    
    plt.plot( [ll, ll], [bl, tl], color='#2d2d2d', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#2d2d2d', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#2d2d2d', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#2d2d2d', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#000000', linestyle= '-', lw=0.5 )
    plt.plot( [orl, orl], [obl, otl], color='#000000', linestyle= '-', lw=0.5 )

    plt.plot( [oll, orl], [obl, obl], color='#000000', linestyle= '-', lw=0.5 )
    plt.plot( [oll, orl], [otl, otl], color='#000000', linestyle= '-', lw=0.5 )

    plt.axis( [lb, rb, bb, tb] )

    ax.axis('off')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.autoscale_view('tight')
    fig.set_facecolor('white')

    if title is not None:
        st = fig.suptitle(title, fontsize=16)
        st.set_weight('bold')
        
    #plt.show()
    
    return fig, ax

    
def get_heatmap(df, threshold=0.5, print_std=False, gaussian=True):
    set_fonts()
    
    x = np.arange(-1.5, +1.5, 1/12)
    if print_std is True:
        y = np.arange(-1.5, +1.5, 1/12)
    else:
        y = np.arange(+1.0, +4.0, 1/12)
        
    P = np.zeros((36,36))
    S = np.zeros((36,36))
    
    smask = (df.pitch_result == '스트라이크')
    bmask = (df.pitch_result == '볼')
    
    if ('sz_top' in df.keys()) & ('sz_bot' in df.keys()):
        sub_df = df[['px', 'pz', 'pitch_result', 'sz_top', 'sz_bot']]
    else:
        if print_std is True:
            return False
        sub_df = df[['px', 'pz', 'pitch_result']]
    
    if print_std is True:
        sub_df['pz_std'] = (sub_df.pz-(sub_df.sz_top+sub_df.sz_bot)/2)/(sub_df.sz_top-sub_df.sz_bot)*2
    
    for i in range(len(x)):
        for j in range(len(y)):
            s = 0
            b = 0
            if i == 0:
                if j == 0:
                    if print_std is True:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.pz_std <= y[j])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.pz_std <= y[j])])
                    else:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.pz <= y[j])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.pz <= y[j])])
                else:
                    if print_std is True:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                    else:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
            else:
                if j == 0:
                    if print_std is True:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j])])
                    else:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j])])
                else:
                    if print_std is True:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                    else:
                        s = len(sub_df.loc[smask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        b = len(sub_df.loc[bmask & (sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
            if s+b > 0:
                P[i,j] = s/(s+b)
            else:
                P[i,j] = 0
    P = P.T
    if gaussian is True:
        P = gaussian_filter(P, sigma=0.85, truncate=1, mode='constant', output=np.float32)
    S = (P >= threshold)
    return P, S


def plot_heatmap(df, title=None, print_std=False, gaussian=False):
    set_fonts()
    
    P, S = get_heatmap(df, print_std=print_std, gaussian=gaussian)
    
    lb = -1.5  # leftBorder
    rb = +1.5  # rightBorder
    
    x = np.arange(lb, rb, 1/12)
    
    if print_std is True:
        bb = -1.5  # bottomBorder
        tb = +1.5  # topBorder
        y = np.arange(bb, tb, 1/12)
    else:
        bb = +1.0  # bottomBorder
        tb = +4.0  # topBorder
        y = np.arange(+1.0, +4.0, 1/12)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6,5), dpi=80, facecolor='white')

    plt.rcParams['axes.unicode_minus'] = False
    plt.pcolormesh(X, Y, P)
    plt.colorbar(format=ticker.FuncFormatter(fmt))
    
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24
    bl = 1.579
    tl = 3.325
    obl = 1.579-3/24
    otl = 3.325+3/24
    
    if print_std is True:
        bl = -1.0
        tl = +1.0
        obl = -1.0-3/24
        otl = +1.0+3/24
    
    plt.plot( [ll, ll], [bl, tl], color='#ffffff', linestyle= '-', lw=1 )
    plt.plot( [rl, rl], [bl, tl], color='#ffffff', linestyle= '-', lw=1 )

    plt.plot( [ll, rl], [bl, bl], color='#ffffff', linestyle= '-', lw=1 )
    plt.plot( [ll, rl], [tl, tl], color='#ffffff', linestyle= '-', lw=1 )

    plt.plot( [oll, oll], [obl, otl], color='#ffffff', linestyle= '-', lw=1 )
    plt.plot( [orl, orl], [obl, otl], color='#ffffff', linestyle= '-', lw=1 )

    plt.plot( [oll, orl], [obl, obl], color='#ffffff', linestyle= '-', lw=1 )
    plt.plot( [oll, orl], [otl, otl], color='#ffffff', linestyle= '-', lw=1 )

    if title is not None:
        plt.title(title)

    plt.axis( [lb+1/12, rb-1/12, bb+1/12, tb-1/12])
    
    return fig, ax


def plot_szone(df, threshold=0.5, title=None, show_area=True, print_std=False, gaussian=False):
    set_fonts()
    
    P, S = get_heatmap(df, threshold=threshold, print_std=print_std, gaussian=gaussian)
    
    fig, ax = plt.subplots(figsize=(5,5), dpi=80, facecolor='white')
        
    lb = -1.5  # leftBorder
    rb = +1.5  # rightBorder
    
    x = np.arange(lb, rb, 1/12)
    
    if print_std is True:
        bb = -1.5  # bottomBorder
        tb = +1.5  # topBorder
        y = np.arange(bb, tb, 1/12)
    else:
        bb = +1.0  # bottomBorder
        tb = +4.0  # topBorder
        y = np.arange(+1.0, +4.0, 1/12)
    X, Y = np.meshgrid(x, y)
    
    plt.rcParams['axes.unicode_minus'] = False
    cmap = matplotlib.colors.ListedColormap(['white', '#ccffcc'])
    plt.pcolor(X, Y, S, cmap=cmap)
    ax.set_title('threshold: {}%'.format(round(threshold*100,1)))
    for i in range(len(x)):
        plt.axvline(x=float(x[i]), color='grey', linestyle='--', lw=0.2)

    for j in range(len(y)):
        plt.axhline(y=float(y[j]), color='grey', linestyle='--', lw=0.2)
        
    ll = -17/24
    rl = +17/24
    oll = -20/24
    orl = +20/24
    if print_std is True:
        bl = -1.0
        tl = +1.0
        obl = -1.0-3/24
        otl = +1.0+3/24
    else:
        bl = 1.579
        tl = 3.325
        obl = 1.579-3/24
        otl = 3.325+3/24
        
    plt.plot( [rl, rl], [bl, tl], color='dimgrey', linestyle= '-', lw=0.3 )
    plt.plot( [ll, ll], [bl, tl], color='dimgrey', linestyle= '-', lw=0.3 )
    plt.plot( [ll, rl], [bl, bl], color='dimgrey', linestyle= '-', lw=0.3 )
    plt.plot( [ll, rl], [tl, tl], color='dimgrey', linestyle= '-', lw=0.3 )
    
    plt.plot( [oll, oll], [obl, otl], color='dimgrey', linestyle= '-', lw=0.3 )
    plt.plot( [orl, orl], [obl, otl], color='dimgrey', linestyle= '-', lw=0.3 )
    plt.plot( [oll, orl], [obl, obl], color='dimgrey', linestyle= '-', lw=0.3 )
    plt.plot( [oll, orl], [otl, otl], color='dimgrey', linestyle= '-', lw=0.3 )

    plt.axis( [lb, rb, bb, tb])    
    
    if title is not None:
        ax.text( 0, tl+0.25, title, color='black', fontsize=14, horizontalalignment='center')
        
    area = np.sum(S)
    print('S-Zone size: {} sq.inch'.format(area))
    
    if show_area is True:
        ax.text( 0, (tl+bl)/2, '{} sq. inch'.format(str(area)), color='black', fontsize=16, horizontalalignment='center' )

    return fig, ax


def release_point(df, title=None, pitcher=None, xlim=None, ylim=None, square=True):
    if pitcher is not None:
        sub_df = df.loc[df.pitcher == pitcher]
    else:
        sub_df = df
        
    pitches = sub_df.pitch_type.drop_duplicates()
    
    fig, ax = plt.subplots(figsize=(4,4), dpi=100, facecolor='white')
    
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
        st = fig.suptitle(title, fontsize=12)
        st.set_weight('bold')
    #display(fig)
    
    return fig, ax


def pitcher_info(df, pitcher=None):
    if pitcher is not None:
        sub_df = df.loc[df.pitcher == pitcher]
    else:
        sub_df = df
        
    groupped = sub_df.groupby('pitch_type').mean().loc[:, ['speed', 'pfx_x', 'pfx_z']]
    
    groupped['count'] = sub_df.groupby('pitch_type').count().speed
    groupped['max'] = sub_df.groupby('pitch_type').max().speed
    groupped['min'] = sub_df.groupby('pitch_type').min().speed
    
    groupped['pct'] = groupped['count'] / groupped['count'].sum() * 100
    
    #display(groupped)
    return groupped


def pitcher_plate_discipline(df, pitcher=None, by_pitch=False):
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

def count_extra_strike_balls(df, rmap, lmap, print_std=True, use_RV=False):
    # 36x36 size heatmap
    
    # bin 별로 스트라이크 개수/볼 개수 측정
    # (1-strike확률)*스트라이크 - (strike확률)*볼
    
    es = 0
    eb = 0

    smask = (df.pitch_result == '스트라이크')
    bmask = (df.pitch_result == '볼')
    
    if ('sz_bot' in df.keys()) & ('sz_top' in df.keys()):
        sub_df = df.loc[smask | bmask].loc[:, ['px', 'pz', 'pitch_result', 'sz_top', 'sz_bot', 'stands', 'balls', 'strikes']]
    else:
        if print_std is True:
            return False
        sub_df = df.loc[smask | bmask].loc[:, ['px', 'pz', 'pitch_result', 'stands', 'balls', 'strikes']]
    
    if print_std is True:
        sub_df['pz_std'] = (sub_df.pz-(sub_df.sz_top+sub_df.sz_bot)/2)/(sub_df.sz_top-sub_df.sz_bot)*2
    
    for i in range(len(sub_df)):
        row = sub_df.iloc[i]

        x = row.px
        if not (-13 <= x*12 <= 13):
            continue

        if print_std is True:
            y = row.pz_std
            if not (-13 <= y*12 <= 13):
                continue

            ind1 = int((x+1.5)*12)
            ind2 = int((y+1.5)*12)

        else:
            y = row.pz
            if not (15 <= y*12 <= 45):
                continue
            
            ind1 = int((x+1.5)*12)
            ind2 = int((y-1)*12)
        
        if use_RV is False:
            if row.stands == '우':
                if row.pitch_result == '스트라이크':
                    es += 1-rmap[ind2][ind1]
                else:
                    eb += rmap[ind2][ind1]
            else:
                if row.pitch_result == '스트라이크':
                    es += 1-lmap[ind2][ind1]
                else:
                    eb += lmap[ind2][ind1]
        else:
            if row.stands == '우':
                if row.pitch_result == '스트라이크':
                    es += (1-rmap[ind2][ind1])*RV[row.strikes*4 + row.balls]
                else:
                    eb += (rmap[ind2][ind1])*RV[row.strikes*4 + row.balls]
            else:
                if row.pitch_result == '스트라이크':
                    es += (1-lmap[ind2][ind1])*RV[row.strikes*4 + row.balls]
                else:
                    eb += (lmap[ind2][ind1])*RV[row.strikes*4 + row.balls]
    return es, eb


def get_framing_gam(df, gam, use_RV=False):
    if importlib.util.find_spec('pygam') is None:
        return None
    
    sub_df = df.loc[df.pitch_result.isin(['스트라이크', '볼'])]
    
    if 'venue' not in sub_df.keys():
        X_target = sub_df[['px', 'pz', 'stands_cat']]
    else:
        X_target = sub_df[['px', 'pz', 'stands_cat', 'venue']]

    predictions = gam.predict(X_target)
    
    if use_RV is True:
        y_comp = pd.DataFrame({'calls': sub_df.calls.get_values(),
                               'pred': predictions,
                               'balls': sub_df.balls.get_values(),
                               'strikes': sub_df.strikes.get_values()})
        
        y_comp['extra'] = np.where(y_comp.calls != y_comp.pred,
                                   np.where(y_comp.pred == False,
                                            RV[y_comp.balls+y_comp.strikes*4],
                                            -RV[y_comp.balls+y_comp.strikes*4]),
                                   0)
        
    else:
        y_comp = pd.DataFrame({'calls': sub_df.calls.get_values(), 'pred': predictions})
        
        y_comp['extra'] = np.where(y_comp.calls != y_comp.pred,
                                   np.where(y_comp.pred == False, 1, -1), 0)
        
    return y_comp.loc[y_comp.extra > 0].extra.sum(), y_comp.loc[y_comp.extra < 0].extra.sum()


def get_season_framing_cell(df, use_RV=False, min_catch=0):
    if '양' in df.stands.drop_duplicates():
        df = df.assign(stands_cat=np.where(df.stands=='양',
                                           np.where(df.throws=='좌', '우', '좌'),
                                           np.where(df.stands=='우', '좌', '우'))
                      )
        
    sub_df = df.loc[df.pitch_result.isin(['스트라이크', '볼'])]
    
    Rmap, _ = get_heatmap(sub_df.loc[sub_df.stands == '우'], print_std=False, gaussian=True)
    Lmap, _ = get_heatmap(sub_df.loc[sub_df.stands == '좌'], print_std=False, gaussian=True)

    catchers = sub_df.pos_2.drop_duplicates()
    extras = []

    for c in catchers:
        caughts = sub_df.loc[sub_df.pos_2 == c]
        es, eb = count_extra_strike_balls(caughts, Rmap, Lmap, print_std=False, use_RV=use_RV)
        if use_RV is False:
            es *= 0.198
            eb *= 0.198

        extras.append((c, len(caughts), es, eb))

    extras.sort(key=lambda tup:tup[2]-tup[3], reverse=True)

    print('이름\t판정 횟수\t+Str\t-Ball\t+Run\t+Run/2000')
    for x in extras:
        if x[1] < min_catch:
            continue
        print('{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(x[0], x[1], x[2], x[3], x[2]-x[3],
                                                              (x[2]-x[3])/x[1]*2000))
        
def get_season_framing_gam(df, use_RV=False, min_catch=0, gam=None):
    if importlib.util.find_spec('pygam') is not None:
        return None
    
    if 'calls' not in df.keys():
        # strike call: 1, ball call: 0
        df = df.assign(calls=np.where(df.pitch_result=='스트라이크', 1, 0))
        
    if 'stands_cat' not in df.keys():
        df = df.assign(stands_cat=np.where(df.stands=='양',
                                           np.where(df.throws=='좌', 1, 0),
                                           np.where(df.stands=='우', 1, 0)))
        
    if 'venue' not in df.keys():
        df.stadium = pd.Categorical(df.stadium)
        df['venue'] = df.stadium.cat.codes

    if gam is None:
        if 'pz_adjusted' not in df.keys():
            df = df.assign(pz_adjusted=(df.pz - (df.sz_top+df.sz_bot)/2)/(df.sz_top-df.sz_bot)*2)

        #X = df.loc[df.pitch_result.isin(['스트라이크', '볼'])][['px', 'pz_adjusted', 'stands_cat', 'venue']]
        #X = df.loc[df.pitch_result.isin(['스트라이크', '볼'])][['px', 'pz', 'stands_cat', 'venue']]
        X = df.loc[df.pitch_result.isin(['스트라이크', '볼'])][['px', 'pz', 'stands_cat', 'venue']]
        y = df.loc[df.pitch_result.isin(['스트라이크', '볼'])][['calls']]

        gam = LogisticGAM().fit(X, y)

    if use_RV is True:
        #sub_df = df[['px', 'pz_adjusted', 'stands_cat', 'venue', 'pitch_result', 'calls', 'pos_2', 'balls', 'strikes']]
        sub_df = df[['px', 'pz', 'stands_cat', 'venue', 'pitch_result', 'calls', 'pos_2', 'balls', 'strikes']]
    else:
        #sub_df = df[['px', 'pz_adjusted', 'stands_cat', 'venue', 'pitch_result', 'calls', 'pos_2']]
        sub_df = df[['px', 'pz', 'stands_cat', 'venue', 'pitch_result', 'calls', 'pos_2']]
    
    catchers = sub_df.pos_2.drop_duplicates()

    results = []

    for c in catchers:
        caughts = sub_df.loc[(sub_df.pos_2 == c) & sub_df.pitch_result.isin(['스트라이크', '볼'])]
        if len(caughts) == 0:
            continue
        es, eb = get_framing_gam(sub_df.loc[sub_df.pos_2 == c], gam, use_RV)
        if use_RV is False:
            es *= 0.198
            eb *= 0.198
        results.append((c, len(caughts), es, eb))

    results.sort(key=lambda tup:tup[2]+tup[3], reverse=True)

    print('Framing w/ GAM')
    print('이름\t포구기회\tExStr\tExBall\tExCall\tEC/2000')

    for r in results:
        if r[1] < min_catch:
            continue
        else:
            print('{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}'.format(r[0], r[1], r[2], r[3], r[2]+r[3], (r[2]+r[3])/r[1]*2000))
            
            
def graph_plate_discipline(df, batter, ma_term=0):
    datesFmt = mdates.DateFormatter('%-m-%d')
    
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
    
    if (ma_term <= 1) or (ma_term >= len(tab.index)):
        raw_num_sum = 0
        swing_sum = 0
        miss_sum = 0
        iz_raw_num_sum = 0
        iz_swing_sum = 0
        iz_miss_sum = 0
        oz_raw_num_sum = 0
        oz_swing_sum = 0
        oz_miss_sum = 0

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
                               (1-miss_sum/raw_num_sum)*100,
                               iz_swing_sum/iz_raw_num_sum*100,
                               (1-iz_miss_sum/iz_swing_sum)*100,
                               oz_swing_sum/oz_raw_num_sum*100,
                               (1-oz_miss_sum/oz_swing_sum)*100])
            
        rs = np.array(result_sum)
            
        fig = plt.figure(figsize=(18,8), dpi=72, facecolor='white')
        fig.suptitle(f'{batter}\'s Cumulative Average', fontsize=16)

        #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
        a1 = fig.add_subplot(231)
        a1.plot(tab.index, rs[:, 0])
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_xlabel('date', fontsize=12)
        a1.set_ylabel('swing%', fontsize=12)

        a2 = fig.add_subplot(232)
        a2.plot(tab.index, rs[:, 1])
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_xlabel('date', fontsize=12)
        a2.set_ylabel('contact%', fontsize=12)

        a3 = fig.add_subplot(233)
        a3.plot(tab.index, rs[:, 2])
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_xlabel('date', fontsize=12)
        a3.set_ylabel('iz-swing%', fontsize=12)

        a4 = fig.add_subplot(234)
        a4.plot(tab.index, rs[:, 3])
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_xlabel('date', fontsize=12)
        a4.set_ylabel('iz-con%', fontsize=12)

        a5 = fig.add_subplot(235)
        a5.plot(tab.index, rs[:, 4])
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_xlabel('date', fontsize=12)
        a5.set_ylabel('oz-swing%', fontsize=12)

        a6 = fig.add_subplot(236)
        a6.plot(tab.index, rs[:, 5])
        a6.xaxis.set_major_formatter(datesFmt)
        a6.grid(True)
        a6.set_xlabel('date', fontsize=12)
        a6.set_ylabel('oz-con%', fontsize=12)

        return fig
    else:
        raw_num_sum = 0
        swing_sum = 0
        miss_sum = 0
        iz_raw_num_sum = 0
        iz_swing_sum = 0
        iz_miss_sum = 0
        oz_raw_num_sum = 0
        oz_swing_sum = 0
        oz_miss_sum = 0
        
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
                               (1-miss_sum/raw_num_sum)*100,
                               iz_swing_sum/iz_raw_num_sum*100,
                               (1-iz_miss_sum/iz_swing_sum)*100,
                               oz_swing_sum/oz_raw_num_sum*100,
                               (1-oz_miss_sum/oz_swing_sum)*100])
    
        rs = np.array(result_sum)
    
        fig = plt.figure(figsize=(18,8), dpi=72, facecolor='white')
        fig.suptitle(f'{batter}\'s {ma_term}-Game Rolling Average', fontsize=16)

        #swing%, con%, izswing%, izcon%, ozswing%, ozcon%
        a1 = fig.add_subplot(231)
        a1.plot(tab.index[term:], rs[:, 0])
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_xlabel('date', fontsize=12)
        a1.set_ylabel('swing%', fontsize=12)

        a2 = fig.add_subplot(234)
        a2.plot(tab.index[term:], rs[:, 1])
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_xlabel('date', fontsize=12)
        a2.set_ylabel('contact%', fontsize=12)

        a3 = fig.add_subplot(232)
        a3.plot(tab.index[term:], rs[:, 2])
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_xlabel('date', fontsize=12)
        a3.set_ylabel('iz-swing%', fontsize=12)

        a4 = fig.add_subplot(235)
        a4.plot(tab.index[term:], rs[:, 3])
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_xlabel('date', fontsize=12)
        a4.set_ylabel('iz-con%', fontsize=12)

        a5 = fig.add_subplot(233)
        a5.plot(tab.index[term:], rs[:, 4])
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_xlabel('date', fontsize=12)
        a5.set_ylabel('oz-swing%', fontsize=12)

        a6 = fig.add_subplot(236)
        a6.plot(tab.index[term:], rs[:, 5])
        a6.xaxis.set_major_formatter(datesFmt)
        a6.grid(True)
        a6.set_xlabel('date', fontsize=12)
        a6.set_ylabel('oz-con%', fontsize=12)

        return fig
    
    
def graph_batting_result(df, batter, ma_term=0):
    datesFmt = mdates.DateFormatter('%-m-%d')
    
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
    
    if (ma_term <= 1) or (ma_term >= len(tab.index)):
        hit_sum = tb_sum = 0
        ab_sum = pa_sum = 0
        hr_sum = sh_sum = sf_sum = 0
        so_sum = bb_sum = hbp_sum = 0

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
            
        rs = np.array(result_sum)
            
        fig = plt.figure(figsize=(18,8), dpi=72, facecolor='white')
        fig.suptitle(f'{batter}\'s Cumulative Average', fontsize=16)

        a1 = fig.add_subplot(231)
        a1.plot(tab.index.get_values(), rs[:, 0])
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_xlabel('date', fontsize=12)
        a1.set_ylabel('AVG', fontsize=12)

        a2 = fig.add_subplot(232)
        a2.plot(tab.index, rs[:, 1])
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_xlabel('date', fontsize=12)
        a2.set_ylabel('OBP', fontsize=12)

        a3 = fig.add_subplot(233)
        a3.plot(tab.index, rs[:, 2])
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_xlabel('date', fontsize=12)
        a3.set_ylabel('SLG', fontsize=12)

        a4 = fig.add_subplot(234)
        a4.plot(tab.index, rs[:, 3])
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_xlabel('date', fontsize=12)
        a4.set_ylabel('OPS', fontsize=12)

        a5 = fig.add_subplot(235)
        a5.plot(tab.index, rs[:, 4])
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_xlabel('date', fontsize=12)
        a5.set_ylabel('BABIP', fontsize=12)
        
        a1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a5.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        
        return fig

    else:
        hit_sum = tb_sum = 0
        ab_sum = pa_sum = 0
        hr_sum = sh_sum = sf_sum = 0
        so_sum = bb_sum = hbp_sum = 0
        
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
            
        rs = np.array(result_sum)
            
        fig = plt.figure(figsize=(18, 8), dpi=72, facecolor='white')
        fig.suptitle(f'{batter}\'s {ma_term}-Game Rolling Average', fontsize=16)

        a1 = fig.add_subplot(231)
        a1.plot(tab.index[term:], rs[:, 0])
        a1.xaxis.set_major_formatter(datesFmt)
        a1.grid(True)
        a1.set_xlabel('date', fontsize=12)
        a1.set_ylabel('AVG', fontsize=12)

        a2 = fig.add_subplot(232)
        a2.plot(tab.index[term:], rs[:, 1])
        a2.xaxis.set_major_formatter(datesFmt)
        a2.grid(True)
        a2.set_xlabel('date', fontsize=12)
        a2.set_ylabel('OBP', fontsize=12)

        a3 = fig.add_subplot(233)
        a3.plot(tab.index[term:], rs[:, 2])
        a3.xaxis.set_major_formatter(datesFmt)
        a3.grid(True)
        a3.set_xlabel('date', fontsize=12)
        a3.set_ylabel('SLG', fontsize=12)

        a4 = fig.add_subplot(234)
        a4.plot(tab.index[term:], rs[:, 3])
        a4.xaxis.set_major_formatter(datesFmt)
        a4.grid(True)
        a4.set_xlabel('date', fontsize=12)
        a4.set_ylabel('OPS', fontsize=12)

        a5 = fig.add_subplot(235)
        a5.plot(tab.index[term:], rs[:, 4])
        a5.xaxis.set_major_formatter(datesFmt)
        a5.grid(True)
        a5.set_xlabel('date', fontsize=12)
        a5.set_ylabel('BABIP', fontsize=12)
        
        a1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        a5.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        
        return fig