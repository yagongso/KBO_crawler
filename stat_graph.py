# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

def graph_batting_result(df, batter, ma_term=0, options=[True, True, True, True, True]):
    datesFmt = mdates.DateFormatter('%m-%d')
    
    if df.game_date.dtype == np.int64:
        df = df.assign(game_date = pd.to_datetime(df.game_date, format='%Y%m%d').dt.date)
    
    df = df.assign(pa=np.where(df.pa_result.notnull(), 1, 0))
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
    
    df = df.assign(pa=np.where(df.pa_result.notnull(), 1, 0))
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
    batters = df.batter.drop_duplicates().values.tolist()
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
    batters = df.batter.drop_duplicates().values.tolist()
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
    pitchers = df.pitcher.drop_duplicates().values.tolist()
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
    pitchers = df.pitcher.drop_duplicates().values.tolist()
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

