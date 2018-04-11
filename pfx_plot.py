# coding: utf-8

import csv
import json
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.random
import matplotlib.ticker as ticker
import datetime
from matplotlib import font_manager as fm, rc
from IPython.display import HTML
from IPython.display import display
import pandas as pd
import os
from enum import Enum
import numbers

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
        

def read_light(fname):
    import warnings
    warnings.filterwarnings("ignore")

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
                ax.scatter(f.px, f.pz_std, color=Colors[c], alpha=.5, s=np.pi*20*72/fig.dpi, label=c)
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, color=Colors[c], alpha=.5, s=np.pi*20*72/fig.dpi, label=c)
                
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
                    ax.scatter(f.px, f.pz_std, color=Colors[c], alpha=.5, s=np.pi*20*72/fig.dpi, label=c)
                    
                    if show_pitch_number is True:
                        for i in f.index:
                            if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                                ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
                else:
                    ax.scatter(f.px, f.pz, color=color, alpha=.5, s=np.pi*20*72/fig.dpi, label=call)

                    if show_pitch_number is True:
                        for i in f.index:
                            if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                                ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
        elif type(calls) is str:
            f = df.loc[df.pitch_result == calls]
            color = Colors[calls]
            
            if print_std is True:
                ax.scatter(f.px, f.pz_std, color=Colors[c], alpha=.5, s=np.pi*20*72/fig.dpi, label=c)
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, color=color, alpha=.5, s=np.pi*20*72/fig.dpi, label=call)

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
                ax.scatter(f.px, f.pz_std, alpha=.5, s=np.pi*20*72/fig.dpi, label=p, cmap='set1')
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*20*72/fig.dpi, label=p, cmap='set1')
                
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
                    ax.scatter(f.px, f.pz_std, alpha=.5, s=np.pi*20*72/fig.dpi, label=p, cmap='set1')
                    
                    if show_pitch_number is True:
                        for i in f.index:
                            if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                                ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
                else:
                    ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*20*72/fig.dpi, label=p, cmap='set1')

                    if show_pitch_number is True:
                        for i in f.index:
                            if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                                ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number,
                                        color='white', fontsize='xx-small', horizontalalignment='center')
        elif type(pitch_types) is str:
            f = df.loc[df.pitch_type == pitch_types]
            
            if print_std is True:
                ax.scatter(f.px, f.pz_std, alpha=.5, s=np.pi*20*72/fig.dpi, label=pitch_types, cmap='set1')
                
                if show_pitch_number is True:
                    for i in f.index:
                        if (f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz_std-0.05 < tb) & (f.loc[i].pz_std-0.05 > bb):
                            ax.text(f.loc[i].px, f.loc[i].pz_std - 0.05, f.loc[i].pitch_number,
                                    color='white', fontsize='xx-small', horizontalalignment='center')
            else:
                ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*20*72/fig.dpi, label=pitch_types, cmap='set1')

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
        
    plt.show()
    
    return fig, ax

    
def get_heatmap(df, threshold=0.5, print_std=False):
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
    
    sub_df = df.loc[:, ['px', 'pz', 'pitch_result', 'sz_top', 'sz_bot']]
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
    S = (P >= threshold)
    return P, S


def plot_heatmap(df, title=None, print_std=False):
    set_fonts()
    
    P, S = get_heatmap(df, print_std=print_std )
    
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


def plot_szone(df, threshold=0.5, title=None, show_area=True, print_std=False):
    set_fonts()
    
    P, S = get_heatmap(df, threshold=threshold, print_std=print_std)
    
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
    
    #display(groupped)
    return groupped


def count_extra_strike_balls(df, rmap, lmap, print_std=True):
    # 36x36 size heatmap
    
    # bin 별로 스트라이크 개수/볼 개수 측정
    # (1-strike확률)*스트라이크 - (strike확률)*볼
    
    x = np.arange(-1.5, +1.5, 1/12)
    if print_std is True:
        y = np.arange(-1.5, +1.5, 1/12)
    else:
        y = np.arange(+1.0, +4.0, 1/12)
        
    es = 0
    eb = 0

    smask = (df.pitch_result == '스트라이크')
    bmask = (df.pitch_result == '볼')
    rmask = (df.stands == '우')
    lmask = (df.stands == '좌')
    
    sub_df = df.loc[:, ['px', 'pz', 'pitch_result', 'sz_top', 'sz_bot']]
    if print_std is True:
        sub_df['pz_std'] = (sub_df.pz-(sub_df.sz_top+sub_df.sz_bot)/2)/(sub_df.sz_top-sub_df.sz_bot)*2
    
    rss = sub_df.loc[smask & rmask]
    lss = sub_df.loc[smask & lmask]
    rbs = sub_df.loc[bmask & rmask]
    lbs = sub_df.loc[bmask & lmask]
    
    for i in range(len(x)):
        # 빠른 계산을 위해 좌우 경계에서 공1개 이상(range 3개 길이) 벗어난 경우는 패스
        if (i < 5) | (i > 31):
            continue
        for j in range(len(y)):
            # 빠른 계산을 위해 상하 중심에서 공1개 이상(평균 range 3개 길이) 벗어난 경우는 패스
            if (j < 5) | (j > 31):
                continue
            rs = 0
            rb = 0
            ls = 0
            lb = 0
            
            if i == 0:
                if j == 0:
                    if print_std is True:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j])])
                        lb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j])])
                    else:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j])])
                        lb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j])])
                else:
                    if print_std is True:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        lb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                    else:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        rb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
            else:
                if j == 0:
                    if print_std is True:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j])])
                        lb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j])])
                    else:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j])])
                        lb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j])])
                else:
                    if print_std is True:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                        lb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz_std <= y[j]) & (sub_df.pz_std > y[j-1])])
                    else:
                        rs = len(rss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        rb = len(rbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        ls = len(lss.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
                        lb = len(lbs.loc[(sub_df.px <= x[i]) & (sub_df.px > x[i-1]) & (sub_df.pz <= y[j]) & (sub_df.pz > y[j-1])])
            es += rs * (1-rmap[j][i])
            es += ls * (1-lmap[j][i])
            eb += rb * rmap[j][i]
            eb += lb * lmap[j][i]
            
    return es, eb