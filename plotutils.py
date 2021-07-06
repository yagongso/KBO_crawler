# coding: utf-8

import sys, csv, json, collections, datetime, os, numbers, importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.random
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib import font_manager as fm, rc
from IPython.display import HTML
from IPython.display import display, Audio
import pandas as pd
import seaborn as sns
from enum import Enum
from scipy.ndimage.filters import gaussian_filter
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import colorsys

if importlib.util.find_spec('pygam') is not None:
    from pygam import LogisticGAM

Results = Enum('Results', '볼 스트라이크 헛스윙 파울 타격 번트파울 번트헛스윙')
Stuffs = Enum('Stuffs', '직구 슬라이더 포크 체인지업 커브 투심 싱커 커터 너클볼')
Colors = {'볼': '#3245ef', '스트라이크': '#ef2926', '헛스윙':'#1a1b1c', '파울':'#edf72c', '타격':'#8348d1', '번트파울':'#edf72c', '번트헛스윙':'#1a1b1c' }

BallColors = {
    '직구': 'red',
    '포심': 'red',
    '투심': 'sandybrown',
    '싱커': 'gold',
    '슬라이더': 'green',
    '커터': 'teal',
    '커브': 'violet',
    '체인지업': 'deepskyblue',
    '포크': 'royalblue',
    '너클볼': 'blueviolet'
}

def fmt(x, pos):
    return r'{}%'.format(int(x*100))


def fmt_cnt(x, pos, cnt):
    return r'{}'.format(int(x*cnt))


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


def clean_data(df):
    df = df.assign(speed = pd.to_numeric(df.speed, errors='coerce'),
                   px = pd.to_numeric(df.px, errors='coerce'),
                   pz = pd.to_numeric(df.pz, errors='coerce'),
                   sz_top = pd.to_numeric(df.sz_top, errors='coerce'),
                   sz_bot = pd.to_numeric(df.sz_bot, errors='coerce'),
                   pfx_x = pd.to_numeric(df.pfx_x, errors='coerce'),
                   pfx_z = pd.to_numeric(df.pfx_z, errors='coerce'))
    if ('pfx_x_raw' in df.keys()) == True:
        df = df.assign(pfx_x_raw = pd.to_numeric(df.pfx_x_raw, errors='coerce'))
    if ('pfx_z_raw' in df.keys()) == True:
        df = df.assign(pfx_z_raw = pd.to_numeric(df.pfx_z_raw, errors='coerce'))
    if ('x0' in df.keys()) == True:
        df = df.assign(x0 = pd.to_numeric(df.x0, errors='coerce'))
    if ('z0' in df.keys()) == True:
        df = df.assign(z0 = pd.to_numeric(df.z0, errors='coerce'))

    df = df[df.px.notnull() & df.pz.notnull()]
    df = df[(df.pitch_type != 'None') & df.pitch_type.notnull()]
    df = df[df.sz_bot.notnull() & df.sz_top.notnull()]

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


def plot_by_call(df, title=None, calls=None, legends=True, show_pitch_number=False, dpi=80, ax=None):
    set_fonts()
    if df.px.isnull().any():
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
        ax.scatter(f.px, f.pz, alpha=.5, s=2*np.pi*dpi, label=c, color=Colors[c], zorder=0)

        if show_pitch_number == True:
            for i in f.index:
                if ((f.loc[i].px < rb ) & (f.loc[i].px > lb) & (f.loc[i].pz < tb) & (f.loc[i].pz > bb)):
                    ax.text(f.loc[i].px, f.loc[i].pz-0.05, f.loc[i].pitch_number.astype(int),
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

    ax.axis('off')
    ax.autoscale_view('tight')

    if legends == True:
        ax.legend(loc='lower center', ncol=2, fontsize='medium')

    return fig, ax


def plot_by_pitch_type(df, title=None, pitch_types=None, legends=True, show_pitch_number=False, dpi=80, ax=None):
    set_fonts()
    if df.px.isnull().any():
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
        c = BallColors[p]
        ax.scatter(f.px, f.pz, alpha=.5, s=np.pi*dpi, label=p, color=c, zorder=0)

        if show_pitch_number == True:
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

    ax.axis('off')
    ax.autoscale_view('tight')

    if legends == True:
        ax.legend(loc='lower center', ncol=2, fontsize='medium')

    return fig, ax


def plot_contour_balls(df, title=None, dpi=100, cmap=None, ax=None,
                       color=None, cbar=True, barpct=True, bw_adjust=1):
    set_fonts()
    if df.px.isnull().any():
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
        fig = ax.figure

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
        if color is not None:
            c = hex_to_rgb(color)
            h, s, v = colorsys.rgb_to_hsv(c[0], c[1], c[2])
            colors = []
            for i in range(10):
                r, g, b = colorsys.hsv_to_rgb(h, s*i/10, v)
                colors.append( '#%02x%02x%02x' % (int(r), int(g), int(b)) )

            cmap = LinearSegmentedColormap.from_list('mycmap', colors)
        else:
            cmap='Reds'

    if barpct == True:
        sns.kdeplot(x=df.px, y=df.pz, shade=True, clip=((lb, rb), (bb, tb)), legend=False,
                    cbar=True, cmap=cmap, thresh=0.1, bw_adjust=bw_adjust,
                    cbar_kws={'format': ticker.FuncFormatter(fmt)},
                    ax=ax, zorder=1)
    else:
        sns.kdeplot(x=df.px, y=df.pz, shade=True, clip=((lb, rb), (bb, tb)), legend=False,
                    cbar=True, cmap=cmap, thresh=0.1, bw_adjust=bw_adjust,
                    ax=ax, zorder=1)
        cb = fig.axes[-1]
        cb.set_yticklabels(map(lambda t: int(float(t.get_text()) * len(df)),
                           cb.get_yticklabels()))

    sns.kdeplot(x=df.px, y=df.pz, clip=((lb, rb), (bb, tb)), legend=False,
                cmap=cmap, linewidths=2.5, thresh=0.1, bw_adjust=bw_adjust,
                ax=ax, zorder=2)

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
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.autoscale_view('tight')

    if title is not None:
        ax.set_title(title, fontsize='medium')

    return fig, ax


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+int(hlen/3)], 16) for i in range(0, hlen, int(hlen/3)))


def plot_heatmap(df, title=None, dpi=144, cmap=None, ax=None, show_full=False, color=None, by_inch=True):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)

    if by_inch == True:
        df = df.assign(px = df.px * 12, pz = df.pz * 12)
        lb = -18
        rb = +18
        ll = -17/2
        rl = +17/2
        oll = -10
        orl = +10

        bl = 18
        tl = 42
        obl = bl-3/2
        otl = tl+3/2
        bb = (bl+tl)/2 - (tl-bl)*15/16
        tb = (bl+tl)/2 + (tl-bl)*15/16
    else:
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
        if color is not None:
            c = hex_to_rgb(color)
            h, s, v = colorsys.rgb_to_hsv(c[0], c[1], c[2])
            colors = []
            for i in range(5, 11):
                r, g, b = colorsys.hsv_to_rgb(h, s*i/10, v)
                colors.append( '#%02x%02x%02x' % (int(r), int(g), int(b)) )
            cmap = LinearSegmentedColormap.from_list('mycmap', colors)
        else:
            cmap='Reds'

    if show_full == True:
        levels = np.asarray([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
        if color is not None:
            c = hex_to_rgb(color)
        else:
            c = (255, 0, 0)
        h, s, v = colorsys.rgb_to_hsv(c[0], c[1], c[2])
        colors = []
        for i in range(11):
            r, g, b = colorsys.hsv_to_rgb(h, s*i/10, v)
            colors.append( '#%02x%02x%02x' % (int(r), int(g), int(b)) )

        cmap = LinearSegmentedColormap.from_list('mycmap', colors)
    else:
        levels = np.asarray([.5, .6, .7, .8, .9, 1.])

    cs = ax.contourf(x, y, rg, levels=levels, cmap=cmap, zorder=1)
    ax.contour(x, y, rg, levels=levels, cmap=cmap, linewidths=2, zorder=1)

    ax.set_facecolor('#cccccc')
    plt.colorbar(cs, format=ticker.FuncFormatter(fmt), ax=ax)

    if by_inch == True:
        major_xtick_step = major_ytick_step = 6
        minor_xtick_step = minor_ytick_step = 1
    else:
        major_xtick_step = major_ytick_step = 1/2
        minor_xtick_step = minor_ytick_step = 1/12

    major_xticks = np.arange(lb, rb+major_xtick_step, major_xtick_step)
    minor_xticks = np.arange(lb, rb+minor_xtick_step, minor_xtick_step)

    if by_inch == True:
        major_yticks = np.arange(0, 60+major_ytick_step, major_ytick_step)
        minor_yticks = np.arange(0, 60+minor_ytick_step, minor_ytick_step)
    else:
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


def plot_szone(df, title=None, dpi=144, show_area=False, ax=None, by_inch=True):
    set_fonts()
    if df.px.dtypes == np.object:
        df = clean_data(df)

    if by_inch == True:
        df = df.assign(px = df.px * 12, pz = df.pz * 12)
        lb = -18
        rb = +18
        ll = -17/2
        rl = +17/2
        oll = -10
        orl = +10

        bl = 18
        tl = 42
        obl = bl-3/2
        otl = tl+3/2
        bb = 12
        tb = 48
    else:
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

    major_yticks = np.linspace(bb, tb, 7)
    minor_yticks = np.linspace(bb, tb, 37)

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

    if show_area == True:
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

    if square == True:
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

    groupped['count'] = sub_df.groupby('pitch_type').speed.count()
    groupped['max'] = sub_df.groupby('pitch_type').speed.max()
    groupped['min'] = sub_df.groupby('pitch_type').speed.min()

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
        if by_pitch == True:
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
        if by_pitch == True:
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
        if by_pitch == True:
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
        if by_pitch == True:
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


def break_plot(df, player, mode=0, ax=None, span=.6, show_dots=False):
    # Mode :
    #   0 for yearly
    #   1 for monthly
    # show_dots :
    #   True for show every dot
    #   False for only ellipse
    plt.style.use('fivethirtyeight')
    target = df.loc[df.pitcher == player]
    if target.shape[0] == 0:
        return
    else:
        pitch_types = target.pitch_type.drop_duplicates()

        target = target.assign(month = target.game_date.apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').month))

        dpi = ax.figure.dpi if ax is not None else 100
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5), dpi=100)

        dots_by_type = []
        labels = []

        if mode == 0:
            maxfreq = target.groupby('pitch_type').size().max() / len(target)

            for p in BallColors.keys():
                t = target.loc[target.pitch_type == p]
                freq = len(t) / len(target)
                alpha = freq / maxfreq * .8
                alpha2 = min(1, freq*2+.25)
                s = t.shape[0]
                if s == 0:
                    continue

                width = (t.pfx_x.quantile(.5+span/2) - t.pfx_x.quantile(.5-span/2)) * 2.54
                height = (t.pfx_z_raw.quantile(.5+span/2) - t.pfx_z_raw.quantile(.5-span/2)) * 2.54
                c1, c2 = t.pfx_x.median() * 2.54, -t.pfx_z_raw.median() * 2.54
                color = BallColors[p]

                ellipse1 = Ellipse((c1, c2), width, height,
                                   ec=color, fc=color, lw=1,
                                   alpha=alpha, zorder=2)
                ellipse2 = Ellipse((c1, c2), width, height,
                                   ec=color, fc='#f0f0f0', lw=1,
                                   alpha=.5, zorder=1)
                if show_dots:
                    ax.scatter(t.pfx_x * 2.54, -t.pfx_z_raw * 2.54, alpha=0.25, c=color, s=dpi*.5, zorder=2)

                ax.add_patch(ellipse1)
                ax.add_patch(ellipse2)

                ax.scatter(c1, c2, alpha=alpha2, s=dpi*.5, zorder=2, c=color)
                dots_by_type.append(ax.scatter(-1000, -1000,
                                               s=dpi*2, zorder=-1, c=color))

                if p == '직구':
                    labels.append('포심')
                else:
                    labels.append(p)
            ax.set_title(f'{player} Break Plot')
        else:
            color_added = {p: False for p in pitch_types}
            for m in target.month.drop_duplicates():
                for p in BallColors.keys():
                    t = target.loc[target.pitch_type == p]
                    if t.shape[0] == 0:
                        continue
                    t_part = t.loc[t.month == m]
                    if t_part.shape[0] == 0:
                        continue
                    x = t_part.pfx_x.mean() * 2.54
                    y = -t_part.pfx_z_raw.mean() * 2.54
                    s = t_part.shape[0]

                    if color_added[p] == False:
                        dots_by_type.append(ax.scatter(x, y, s=dpi*2, c=BallColors[p]))
                        if p == '직구':
                            labels.append('포심')
                        else:
                            labels.append(p)
                        color_added[p] = True
                    else:
                        ax.scatter(x, y, s=dpi*2, c=BallColors[p])

                plt.gca().set_prop_cycle(None)
            ax.set_title(f'{player} Monthly Break Plot')

        ax.text(-43, 180, '◀가라앉음', rotation=90, weight='bold')

        ax.text(-35, 180, '◀우타석', weight='bold')
        ax.text(25, 180, '좌타석▶', weight='bold')
        ax.set_xlim(-14 * 2.54, 14 * 2.54)
        ax.set_ylim(0, 180)
        ax.invert_yaxis()

        ax.legend(tuple(dots_by_type), tuple(labels), ncol=3, loc='lower center', fontsize='small', bbox_to_anchor=(0.5, 0.025))

        return ax


def pitchtype_plot(df, pitcher, ax=None):
    plt.style.use('fivethirtyeight')
    target = df.loc[df.pitcher == pitcher]
    if target.shape[0] == 0:
        return
    else:
        target = target.assign(pitch_type = target.pitch_type.apply(lambda x: '포심' if x=='직구' else x))
        g = target.groupby('pitch_type').size().sort_values(ascending=False) / len(target) * 100
        mean = target.groupby('pitch_type').speed.mean()

    dpi = 100

    if (ax is None) or (len(ax) < 2):
        _, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    else:
        dpi = ax[0].figure.dpi

    sns.barplot(x=g.values, y=g.index, palette=BallColors, ax=ax[0])

    sns.boxplot(y=target['pitch_type'],
                x=target['speed'],
                fliersize=0,
                width=0.5,
                palette=BallColors,
                linewidth=1,
                saturation=1,
                order=g.index,
                showmeans=True,
                meanprops=dict(marker='o', markeredgecolor='black', markerfacecolor='white'),
                ax=ax[1])

    ylabels = list(x.get_text() for x in ax[1].get_ymajorticklabels())
    for i in ax[1].get_yticks():
        p = ylabels[i]
        ax[1].text(float(mean[p])-2.5, i-0.3, f'{float(mean[p]):.1f}', fontsize='small')

    ylabels = list(x.get_text() for x in ax[0].get_ymajorticklabels())
    for i in ax[0].get_yticks():
        p = ylabels[i]
        if g[p] < 10:
            ax[0].text(g[p]+4.5, i, f'{float(g[p]):.0f}%')
        else:
            ax[0].text(min(g[p]+6, 56), i, f'{float(g[p]):.0f}%')

    ax[0].set_xlim(0, 50)
    ax[0].yaxis.set_ticks_position('right')
    ax[0].yaxis.set_ticks([])
    ax[0].invert_xaxis()
    ax[0].set_ylabel('')

    newtick = list(str(int(x)) for x in ax[0].get_xticks())
    i = -1
    while True:
        if int(newtick[i]) > 50:
            i = i - 1
            continue
        else:
            newtick[i] = f'{newtick[i]}%'
            break
    newtick[0] = ''
    ax[0].set_xticklabels(newtick)

    ax[1].set_xlim(100, 160)
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')

    newtick = list(str(int(x)) for x in ax[1].get_xticks())
    i = -1
    while True:
        if int(newtick[i]) > 160:
            i = i - 1
            continue
        else:
            newtick[i] = f'{newtick[i]}km/h'
            break
    newtick[0] = ''
    ax[1].set_xticklabels(newtick)

    ax[0].set_title(f'{pitcher} Pitch Type Frequency')
    ax[1].set_title(f'{pitcher} Speed')

    return ax


def draw_zone_line(ax, color='black'):
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


    ax.plot( [ll, ll], [bl, tl], color=color, linestyle='solid', lw=1 )
    ax.plot( [rl, rl], [bl, tl], color=color, linestyle='solid', lw=1 )
    ax.plot( [ll+(rl-ll)/3, ll+(rl-ll)/3], [bl, tl], color=color, linestyle= 'solid', lw=.5 )
    ax.plot( [ll+(rl-ll)*2/3, ll+(rl-ll)*2/3], [bl, tl], color=color, linestyle= 'solid', lw=.5 )

    ax.plot( [ll, rl], [bl, bl], color=color, linestyle='solid', lw=1 )
    ax.plot( [ll, rl], [tl, tl], color=color, linestyle='solid', lw=1 )
    ax.plot( [ll, rl], [bl+(tl-bl)/3, bl+(tl-bl)/3], color=color, linestyle= 'solid', lw=.5 )
    ax.plot( [ll, rl], [bl+(tl-bl)*2/3, bl+(tl-bl)*2/3], color=color, linestyle= 'solid', lw=.5 )

    ax.plot( [oll, oll], [obl, otl], color=color, linestyle='solid', lw=1 )
    ax.plot( [orl, orl], [obl, otl], color=color, linestyle='solid', lw=1 )

    ax.plot( [oll, orl], [obl, obl], color=color, linestyle='solid', lw=1 )
    ax.plot( [oll, orl], [otl, otl], color=color, linestyle='solid', lw=1 )


def pitchmix_breakdown(df, pitcher, ax=None, span=0.6):
    plt.style.use('fivethirtyeight')
    target = df.loc[df.pitcher == pitcher]
    if target.shape[0] == 0:
        return
    else:
        target = target.assign(pitch_type = target.pitch_type.apply(lambda x: '포심' if x=='직구' else x))
        target = target[target.px.notnull()]

    dpi = 72

    if (ax is None) or (len(ax) < 2):
        _, ax = plt.subplots(2, 4, figsize=(20, 10), dpi=72, facecolor='#898f99')
    elif (len(ax[0]) < 4):
        _, ax = plt.subplots(2, 4, figsize=(20, 10), dpi=72, facecolor='#898f99')
    else:
        dpi = ax[0, 0].figure.dpi

    t_left = target[target.stands == '좌']
    t_right = target[target.stands == '우']

    t_left_p3 = t_left[t_left.pitch_number < 3]
    t_left_p4 = t_left[t_left.pitch_number > 4]
    t_left_2s = t_left[t_left.strikes == 2]

    t_right_p3 = t_right[t_right.pitch_number < 3]
    t_right_p4 = t_right[t_right.pitch_number > 4]
    t_right_2s = t_right[t_right.strikes == 2]

    for t, a in zip([t_left, t_left_p3, t_left_p4, t_left_2s,
                     t_right, t_right_p3, t_right_p4, t_right_2s],
                     ax.reshape(-1, )):

        maxfreq = t.groupby('pitch_type').size().max() / len(t)
        dots_by_type = []
        labels = []

        for p in BallColors.keys():
            tp = t[t.pitch_type == p]
            if len(tp) == 0:
                continue

            freq = len(tp) / len(t)

            if freq < 0.05:
                continue
            alpha = freq / maxfreq * .8
            alpha2 = min(1, freq*2+.25)

            w = (tp.px.quantile(.5 + span/2) - tp.px.quantile(.5 - span/2))
            h = (tp.pz.quantile(.5 + span/2) - tp.pz.quantile(.5 - span/2))
            w2 = (tp.px.quantile(.5 + span/4) - tp.px.quantile(.5 - span/4))
            h2 = (tp.pz.quantile(.5 + span/4) - tp.pz.quantile(.5 - span/4))
            c1, c2 = tp.px.median(), tp.pz.median()
            c3 = tp[tp.px.between(tp.px.quantile(.5 - span/4), tp.px.quantile(.5 + span/4))].px.mean()
            c4 = tp[tp.pz.between(tp.pz.quantile(.5 - span/4), tp.pz.quantile(.5 + span/4))].pz.mean()

            color = BallColors[p]

            from matplotlib.patches import Ellipse
            ellipse1 = Ellipse((c1, c2), w, h,
                               ec=color, fc=color, lw=1,
                               alpha=alpha, zorder=2)
            ellipse2 = Ellipse((c1, c2), w, h,
                               ec=color, fc='#f0f0f0', lw=1,
                               alpha=.5, zorder=1)
            ellipse3 = Ellipse((c3, c4), w2, h2,
                               fc=color, lw=1,
                               alpha=min(alpha+0.2, 1), zorder=3,
                               hatch='//', edgecolor='black')
            a.add_patch(ellipse1)
            a.add_patch(ellipse2)
            a.add_patch(ellipse3)
            a.scatter(c1, c2, alpha=alpha2, s=dpi*.5, zorder=2, c=color, label=p)

            dots_by_type.append(a.scatter(-1000, -1000,
                                s=dpi*2, zorder=-1, c=color))

            if p == '직구':
                labels.append('포심')
            else:
                labels.append(p)
        a.legend(tuple(dots_by_type), tuple(labels), ncol=3, loc='lower center', fontsize='small', bbox_to_anchor=(0.5, 0.025))

    ax[0, 0].set_title('vs좌')
    ax[0, 1].set_title('vs좌 P<3')
    ax[0, 2].set_title('vs좌 P>4')
    ax[0, 3].set_title('vs좌 2S')

    ax[1, 0].set_title('vs우')
    ax[1, 1].set_title('vs우 P<3')
    ax[1, 2].set_title('vs우 P>4')
    ax[1, 3].set_title('vs우 2S')

    for a in ax.reshape(-1, ):
        draw_zone_line(a)
        a.set_facecolor('#898f99')
        a.axis( [-1.25, 1.25, 0.50, 3.75] )
        a.axis('off')
        a.autoscale_view('tight')

    plt.show()
    plt.close()
