# coding: utf-8

import numpy as np
import pandas as pd
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from IPython.display import display, Audio
from plotutils import clean_data
from matplotlib.colors import ListedColormap

import importlib
if importlib.util.find_spec('pygam') is not None:
    from pygam import LogisticGAM

def get_re24(df):
    columns_needed = ['game_date', 'away', 'home', 'inning', 'inning_topbot',
                      'outs', 'score_home', 'score_away', 'pitch_result', 'pa_result',
                      'on_1b', 'on_2b', 'on_3b']
    t = df.loc[df.outs<3][columns_needed]
    t = t.assign(play_run_home = t.score_home.shift(-1) - t.score_home,
                 play_run_away = t.score_away.shift(-1) - t.score_away)
    t.play_run_away = t.play_run_away.fillna(0)
    t.play_run_home = t.play_run_home.fillna(0)
    t = t.assign(play_run = np.where((t.inning_topbot == '말') &
                                     (t.pa_result != 'None') &
                                     (~t.pa_result.isnull()),
                                     t.play_run_home,
                                     np.where((t.inning_topbot == '초') &
                                              (t.pa_result != 'None') &
                                              (~t.pa_result.isnull()),
                                              t.play_run_away, 0)))
    t = t.assign(play_run = np.where(t.pa_result == '삼진', 0, t.play_run))
    t = t.assign(play_run = np.where(t.pa_result.isin(['몸에 맞는 공', '몸에 맞는 볼',
                                                       '볼넷', '자동 고의4구', '고의4구',
                                                       '고의 4구', '자동 고의 4구']),
                                     np.where((t.on_1b != 'None') & (~t.on_1b.isnull()) &
                                              (t.on_2b != 'None') & (~t.on_2b.isnull()) &
                                              (t.on_3b != 'None') & (~t.on_3b.isnull()), 1, 0),
                                     t.play_run))
    t.play_run = t.play_run.fillna(0)

    # RE24 계산을 위해 PA result 있는 데이터만 필터
    # 자동 고의4구도 포함 - 이때는 pitch result가 'None'임
    # PA result가 '투구 외 득점'일 때도 pitch result가 'None'인데 이것도 포함
    pa_res = t.loc[(t.pa_result != 'None') &
                   (~t.pa_result.isnull())]

    # base 할당
    pa_res = pa_res.assign(base1 = np.where((pa_res.on_1b != 'None') &
                                            (~pa_res.on_1b.isnull()), '1', '_'),
                           base2 = np.where((pa_res.on_2b != 'None') &
                                            (~pa_res.on_2b.isnull()), '2', '_'),
                           base3 = np.where((pa_res.on_3b != 'None') &
                                            (~pa_res.on_3b.isnull()), '3', '_'))

    pa_res = pa_res.assign(base = pa_res.base1 + pa_res.base2 + pa_res.base3)

    ####################################
    # BASE-OUT 조합에 따른 타석의 숫자 #
    ####################################
    base_out_counts = pa_res.pivot_table('away',
                                         'base', 'outs', 'count', 0).sort_index(ascending=False)

    # 이닝 최대 점수
    g1 = pa_res.groupby(['game_date', 'home', 'away',
                         'inning', 'inning_topbot']).score_away
    g2 = pa_res.groupby(['game_date', 'home', 'away',
                         'inning', 'inning_topbot']).score_home
    pa_res = pa_res.assign(max_score_in_inning = np.where(pa_res.inning_topbot == '초',
                                                          g1.transform(max),
                                                          g2.transform(max)))

    # 해당 플레이 이후 그 이닝에서 발생하는 점수
    pa_res = pa_res.assign(runs_scored_after_play = np.where(pa_res.inning_topbot == '초',
                                                             pa_res.max_score_in_inning - pa_res.score_away,
                                                             pa_res.max_score_in_inning - pa_res.score_home))
    #################################################################
    # BASE-OUT 조합에 따른, 플레이 이후 그 이닝에 발생한 점수의 합 #
    #################################################################
    base_out_run_sum = pa_res.pivot_table('runs_scored_after_play',
                                          'base', 'outs', 'sum', 0).sort_index(ascending=False)

    # RE24 table
    re24 = base_out_run_sum / base_out_counts

    return re24.sort_values(0)


def get_rv_event(df):
    columns_needed = ['game_date', 'away', 'home', 'inning', 'inning_topbot',
                      'outs', 'score_home', 'score_away', 'pitch_result', 'pa_result',
                      'on_1b', 'on_2b', 'on_3b']
    t = df.loc[df.outs<3][columns_needed]
    t = t.assign(play_run_home = t.score_home.shift(-1) - t.score_home,
                 play_run_away = t.score_away.shift(-1) - t.score_away)
    t.play_run_away = t.play_run_away.fillna(0)
    t.play_run_home = t.play_run_home.fillna(0)
    t = t.assign(play_run = np.where((t.inning_topbot == '말') &
                                     (t.pa_result != 'None') &
                                     (~t.pa_result.isnull()),
                                     t.play_run_home,
                                     np.where((t.inning_topbot == '초') &
                                              (t.pa_result != 'None') &
                                              (~t.pa_result.isnull()),
                                              t.play_run_away, 0)))
    t = t.assign(play_run = np.where(t.pa_result == '삼진', 0, t.play_run))
    t = t.assign(play_run = np.where(t.pa_result.isin(['몸에 맞는 공', '몸에 맞는 볼',
                                                       '볼넷', '자동 고의4구', '고의4구',
                                                       '고의 4구', '자동 고의 4구']),
                                     np.where((t.on_1b != 'None') & (~t.on_1b.isnull()) &
                                              (t.on_2b != 'None') & (~t.on_2b.isnull()) &
                                              (t.on_3b != 'None') & (~t.on_3b.isnull()), 1, 0),
                                     t.play_run))
    t.play_run = t.play_run.fillna(0)

    # RE24 계산을 위해 PA result 있는 데이터만 필터
    # 자동 고의4구도 포함 - 이때는 pitch result가 'None'임
    # PA result가 '투구 외 득점'일 때도 pitch result가 'None'인데 이것도 포함
    pa_res = t.loc[(t.pa_result != 'None') & (~t.pa_result.isnull())]

    # base 할당
    pa_res = pa_res.assign(base1 = np.where((pa_res.on_1b != 'None') &
                                            (~pa_res.on_1b.isnull()), '1', '_'),
                           base2 = np.where((pa_res.on_2b != 'None') &
                                            (~pa_res.on_2b.isnull()), '2', '_'),
                           base3 = np.where((pa_res.on_3b != 'None') &
                                            (~pa_res.on_3b.isnull()), '3', '_'))

    pa_res = pa_res.assign(base = pa_res.base1 + pa_res.base2 + pa_res.base3)

    ####################################
    # BASE-OUT 조합에 따른 타석의 숫자 #
    ####################################
    base_out_counts = pa_res.pivot_table('away',
                                         'base', 'outs', 'count', 0).sort_index(ascending=False)

    # 이닝 최대 점수
    g1 = pa_res.groupby(['game_date', 'home', 'away',
                         'inning', 'inning_topbot']).score_away
    g2 = pa_res.groupby(['game_date', 'home', 'away',
                         'inning', 'inning_topbot']).score_home
    pa_res = pa_res.assign(max_score_in_inning = np.where(pa_res.inning_topbot == '초',
                                                          g1.transform(max),
                                                          g2.transform(max)))

    # 해당 플레이 이후 그 이닝에서 발생하는 점수
    pa_res = pa_res.assign(runs_scored_after_play = np.where(pa_res.inning_topbot == '초',
                                                             pa_res.max_score_in_inning - pa_res.score_away,
                                                             pa_res.max_score_in_inning - pa_res.score_home))
    ################################################################
    # BASE-OUT 조합에 따른, 플레이 이후 그 이닝에 발생한 점수의 합 #
    ################################################################
    base_out_run_sum = pa_res.pivot_table('runs_scored_after_play',
                                          'base', 'outs', 'sum', 0).sort_index(ascending=False)

    # RE24 table
    re24 = base_out_run_sum / base_out_counts

    base_out_re = re24.sort_index(axis=1, ascending=False).values.reshape(-1,1)

    ##############################################
    # BASE-OUT 조합에 따른 각 타격 이벤트의 개수 #
    ##############################################
    event_count_by_base_out = pa_res.pivot_table('runs_scored_after_play', ['base', 'outs'],
                                                 'pa_result', 'count', 0).sort_index(ascending=False)

    ################################################
    # BASE-OUT 조합에 따른 각 타석 이벤트의 RE sum #
    ################################################
    event_re_sum_by_base_out = event_count_by_base_out * np.repeat(base_out_re,
                                                                   event_count_by_base_out.shape[1],
                                                                   axis=1)

    ################################################
    # BASE-OUT 조합에 따른 각 타석 이벤트의 RE AVG #
    ################################################
    event_re_avg_by_base_out = event_re_sum_by_base_out.sum(axis=0) / event_count_by_base_out.sum(axis=0)

    ###########################################################
    # 플레이(타석 이벤트) 이후 해당 이닝에 발생한 점수의 평균 #
    ###########################################################
    runs_after_play_sum_by_event = pa_res.pivot_table('runs_scored_after_play',
                                                      'pa_result',
                                                      aggfunc='sum', fill_value=0)
    event_count = pa_res.pivot_table('runs_scored_after_play', 'pa_result', aggfunc='count', fill_value=0)
    runs_after_play_avg_by_event = (runs_after_play_sum_by_event / event_count).fillna(0)

    rv = runs_after_play_avg_by_event.runs_scored_after_play - event_re_avg_by_base_out
    
    rvdf = pd.DataFrame(rv)
    rvdf = rvdf.rename_axis('event')
    rvdf = rvdf.rename(index=str, columns={0:'Run Value'})

    return rvdf


def get_rv_event_simple(df):
    columns_needed = ['game_date', 'away', 'home', 'inning', 'inning_topbot',
                      'outs', 'score_home', 'score_away', 'pitch_result', 'pa_result',
                      'on_1b', 'on_2b', 'on_3b']
    single = ['내야안타', '내야 안타', '1루타', '번트 안타', '번트안타']
    fieldout = ['필드 아웃', '필드아웃', '타구맞음 아웃', '타구 맞음 아웃', '타구맞음아웃']
    outs = ['필드 아웃', '필드아웃', '타구맞음 아웃', '포스 아웃', '포스아웃']

    events_short = ['안타', '내야안타', '내야 안타', '번트안타', '번트 안타', '1루타', '2루타', '3루타', '홈런',
                    '병살타', '자동 고의4구', '고의4구', '고의 4구', '자동 고의 4구',
                    '볼넷', '삼진', '포스 아웃', '필드 아웃', '타구맞음 아웃', '타구 맞음 아웃',
                    '희생플라이', '희생번트', '희생 플라이', '희생 번트']
    t = df.loc[df.outs < 3][columns_needed]

    # 자동고의4구, 고의4구 하나로 합친다
    t = t.assign(pa_result = np.where(t.pa_result == '자동 고의4구',
                                      '고의4구', t.pa_result))
    t = t.assign(pa_result = np.where(t.pa_result == '자동 고의 4구',
                                      '고의4구', t.pa_result))
    
    # 단타 종류는 모두 1루타로 축약, 필드아웃 종류는 필드 아웃으로 축약
    t = t.assign(pa_result2 = np.where(t.pa_result.isin(single), '1루타',
                                       np.where(t.pa_result.isin(fieldout),
                                                '필드 아웃',
                                                t.pa_result)))
    # 필드아웃 포스아웃은 모두 그냥 아웃으로 축약
    t = t.assign(pa_result2 = np.where(t.pa_result2.isin(outs),
                                       '아웃', t.pa_result2))

    # 플레이 단위 별로 홈/어웨이 점수 차이
    t = t.assign(play_run_home = t.score_home.shift(-1) - t.score_home,
                 play_run_away = t.score_away.shift(-1) - t.score_away)
    t.play_run_away = t.play_run_away.fillna(0)
    t.play_run_home = t.play_run_home.fillna(0)

    # 플레이 단위 별로 점수 차이
    t = t.assign(play_run = np.where((t.inning_topbot == '말') &
                                     (t.pa_result != 'None') &
                                     (~t.pa_result.isnull()),
                                     t.play_run_home,
                                     np.where((t.inning_topbot == '초') &
                                              (t.pa_result != 'None') &
                                              (~t.pa_result.isnull()),
                                              t.play_run_away, 0)))

    # 삼진은 플레이 단위 점수차이 0, 볼넷은 밀어내기 제외하면 0(추가 플레이는 고려 X)
    t = t.assign(play_run = np.where(t.pa_result == '삼진', 0, t.play_run))
    t = t.assign(play_run = np.where(t.pa_result.isin(['몸에 맞는 공', '몸에 맞는 볼',
                                                       '볼넷', '자동 고의4구', '고의4구',
                                                       '고의 4구', '자동 고의 4구']),
                                     np.where((t.on_1b != 'None') & (~t.on_1b.isnull()) &
                                              (t.on_2b != 'None') & (~t.on_2b.isnull()) &
                                              (t.on_3b != 'None') & (~t.on_3b.isnull()), 1, 0),
                                     t.play_run))
    t.play_run = t.play_run.fillna(0)

    # RE24 계산을 위해 PA result 있는 데이터만 필터
    # 자동 고의4구도 포함 - 이때는 pitch result가 nan 또는 'None'임(legacy)
    # legacy에서는 PA result가 '투구 외 득점'일 때도 pitch result가 'None'인데 이것도 포함
    pa_res = t.loc[(t.pa_result != 'None') & (~t.pa_result.isnull())]

    # simple 데이터로 필터
    pa_res = pa_res.loc[pa_res.pa_result.isin(events_short)]

    # base 할당 : 주자 있으면 1/2/3으로, 없으면 _으로 표기
    pa_res = pa_res.assign(base1 = np.where((pa_res.on_1b != 'None') & (~pa_res.on_1b.isnull()), '1', '_'),
                           base2 = np.where((pa_res.on_2b != 'None') & (~pa_res.on_2b.isnull()), '2', '_'),
                           base3 = np.where((pa_res.on_3b != 'None') & (~pa_res.on_3b.isnull()), '3', '_'))

    pa_res = pa_res.assign(base = pa_res.base1 + pa_res.base2 + pa_res.base3)

    ####################################
    # BASE-OUT 조합에 따른 타석의 숫자 #
    ####################################
    base_out_counts = pa_res.pivot_table('away',
                                         'base', 'outs', 'count', 0).sort_index(ascending=False)

    # 이닝 최대 점수
    g1 = pa_res.groupby(['game_date', 'home', 'away',
                         'inning', 'inning_topbot']).score_away
    g2 = pa_res.groupby(['game_date', 'home', 'away',
                         'inning', 'inning_topbot']).score_home
    pa_res = pa_res.assign(max_score_in_inning = np.where(pa_res.inning_topbot == '초',
                                                          g1.transform(max),
                                                          g2.transform(max)))

    # 해당 플레이 이후 그 이닝에서 발생하는 점수
    pa_res = pa_res.assign(runs_scored_after_play = np.where(pa_res.inning_topbot == '초',
                                                             pa_res.max_score_in_inning - pa_res.score_away,
                                                             pa_res.max_score_in_inning - pa_res.score_home))
    #################################################################
    # BASE-OUT 조합에 따른, 플레이 이후 그 이닝에 발생한 점수의 합 #
    #################################################################
    base_out_run_sum = pa_res.pivot_table('runs_scored_after_play',
                                          'base', 'outs', 'sum', 0).sort_index(ascending=False)

    # RE24 table
    re24 = base_out_run_sum / base_out_counts
    
    base_out_re = re24.sort_index(axis=1, ascending=False).values.reshape(-1,1)
    
    ##############################################
    # BASE-OUT 조합에 따른 각 타격 이벤트의 개수 #
    ##############################################
    event_count_by_base_out = pa_res.pivot_table('runs_scored_after_play', ['base', 'outs'],
                                                 'pa_result2', 'count', 0).sort_index(ascending=False)

    ################################################
    # BASE-OUT 조합에 따른 각 타석 이벤트의 RE sum #
    ################################################
    event_re_sum_by_base_out = event_count_by_base_out * np.repeat(base_out_re,
                                                                   event_count_by_base_out.shape[1],
                                                                   axis=1)

    ################################################
    # BASE-OUT 조합에 따른 각 타석 이벤트의 RE AVG #
    ################################################
    event_re_avg_by_base_out = event_re_sum_by_base_out.sum(axis=0) / event_count_by_base_out.sum(axis=0)
    
    ###########################################################
    # 플레이(타석 이벤트) 이후 해당 이닝에 발생한 점수의 평균 #
    ###########################################################
    runs_after_play_sum_by_event = pa_res.pivot_table('runs_scored_after_play',
                                                      'pa_result2',
                                                      aggfunc='sum', fill_value=0)
    event_count = pa_res.pivot_table('runs_scored_after_play', 'pa_result2', aggfunc='count', fill_value=0)
    runs_after_play_avg_by_event = (runs_after_play_sum_by_event / event_count).fillna(0)

    rv = runs_after_play_avg_by_event.runs_scored_after_play - event_re_avg_by_base_out

    rvdf = pd.DataFrame(rv)
    rvdf = rvdf.rename_axis('event')
    rvdf = rvdf.rename(index=str, columns={0:'Run Value'})

    return rvdf


def get_rv_of_ball_strike(df):
    """
    볼과 스트라이크의 평균 득점 가치(run value)를 구한다.
    공격(타자) 측 입장에서 계산한 것이므로 플러스가 공격 측 이득이다.

    Parameter
    ---------
    df : pandas.DataFrame
    시즌 pitch by pitch 데이터가 담긴 데이터프레임

    return
    ------
    rvball, rvstrike : float, float
    전자는 볼 1개의 득점 가치, 후자는 스트라이크 1개의 득점 가치.
    """
    columns_needed = ['game_date', 'away', 'home', 'inning', 'inning_topbot',
                      'pa_number', 'pa_result', 'pitch_result']
    t = df.loc[df.outs < 3][columns_needed]
    t = t.assign(pa_code = t.game_date.map(str)+
                           t.away.map(str)+
                           t.inning.map(str)+
                           t.inning_topbot.map(str)+
                           t.pa_number.map(str))
    pa_fin_res = t.loc[(t.pa_result != 'None') &
                       (~t.pa_result.isnull()) &
                       (t.pa_result != '투구 외 득점')][['pa_code', 'pa_result']]
    
    t_join = t.set_index('pa_code').join(pa_fin_res.set_index('pa_code'), how='outer', rsuffix='_callee')
    t_join = t_join.drop(t_join.loc[t_join.pa_result_callee.isnull()].index)
    
    rv = get_rv_event(df)
    dict_rv = dict(rv['Run Value'])

    t_join = t_join.assign(rv = t_join.pa_result_callee.map(dict_rv))

    strikes = t_join.loc[t_join.pitch_result == '스트라이크']
    balls = t_join.loc[t_join.pitch_result == '볼']

    # event(when strike) rv sum
    event_rv_sum_when_strike = strikes.pivot_table('rv',
                                                   columns=['pa_result_callee'],
                                                   aggfunc='sum',
                                                   fill_value=0).sum(axis=1)

    # event(when ball) rv sum
    event_rv_sum_when_ball = balls.pivot_table('rv',
                                               columns=['pa_result_callee'],
                                               aggfunc='sum',
                                               fill_value=0).sum(axis=1)

    # event count(when strike)
    event_count_strike = strikes.pivot_table('rv',
                                             columns=['pa_result_callee'],
                                             aggfunc='count',
                                             fill_value=0).sum(axis=1)
    # event count(when ball)
    event_count_ball = balls.pivot_table('rv',
                                         columns=['pa_result_callee'],
                                         aggfunc='count',
                                         fill_value=0).sum(axis=1)

    return (event_rv_sum_when_ball / event_count_ball)[0], (event_rv_sum_when_strike / event_count_strike)[0]


def get_rv_of_ball_strike_by_count(df):
    """
    볼과 스트라이크의 득점 가치(run value)를 볼카운트 별로 구한다.
    공격(타자) 측 입장에서 계산한 것이므로 플러스가 공격 측 이득이다.

    Parameter
    ---------
    df : pandas.DataFrame
    시즌 pitch by pitch 데이터가 담긴 데이터프레임

    return
    ------
    rvball_series, rvstrike_series : pandas.core.series.Series * 2
    전자는 볼의 득점 가치, 후자는 스트라이크의 득점 가치를 볼카운트 별로 정리한 것이다.
    pandas Series 타입이기 때문에 앞에서 뒤를 빼면 카운트별 '스트와 볼의 득점가치 차이'가 된다.
    """
    columns_needed = ['game_date', 'away', 'home', 'inning', 'inning_topbot',
                      'pa_number', 'pa_result', 'pitch_result', 'balls', 'strikes']
    t = df.loc[df.outs < 3][columns_needed]
    t = t.assign(pa_code = t.game_date.map(str)+
                           t.away.map(str)+
                           t.inning.map(str)+
                           t.inning_topbot.map(str)+
                           t.pa_number.map(str))
    pa_fin_res = t.loc[(t.pa_result != 'None') &
                       (~t.pa_result.isnull()) &
                       (t.pa_result != '투구 외 득점')][['pa_code', 'pa_result']]

    t_join = t.set_index('pa_code').join(pa_fin_res.set_index('pa_code'), how='outer', rsuffix='_callee')
    t_join = t_join.drop(t_join.loc[t_join.pa_result_callee.isnull()].index)

    rv = get_rv_event(df)
    dict_rv = dict(rv['Run Value'])

    t_join = t_join.assign(rv = t_join.pa_result_callee.map(dict_rv))

    strikes = t_join.loc[t_join.pitch_result == '스트라이크']
    balls = t_join.loc[t_join.pitch_result == '볼']

    # event(when strike) rv sum
    event_rv_sum_when_strike = strikes.pivot_table('rv',
                                                   ['strikes', 'balls'],
                                                   columns=['pa_result_callee'],
                                                   aggfunc='sum',
                                                   fill_value=0).sum(axis=1)

    # event(when ball) rv sum
    event_rv_sum_when_ball = balls.pivot_table('rv',
                                               ['strikes', 'balls'],
                                               columns=['pa_result_callee'],
                                               aggfunc='sum',
                                               fill_value=0).sum(axis=1)

    # event count(when strike)
    event_count_strike = strikes.pivot_table('rv',
                                             ['strikes', 'balls'],
                                             columns=['pa_result_callee'],
                                             aggfunc='count',
                                             fill_value=0).sum(axis=1)
    # event count(when ball)
    event_count_ball = balls.pivot_table('rv',
                                         ['strikes', 'balls'],
                                         columns=['pa_result_callee'],
                                         aggfunc='count',
                                         fill_value=0).sum(axis=1)

    return (event_rv_sum_when_ball / event_count_ball), (event_rv_sum_when_strike / event_count_strike)


def calc_framing_cell(df, rv_by_count=False):
    # 20-80% 구간 측정이 mean을 0에 가깝게 맞출 수 있음.
    features = ['px', 'pz', 'pitch_result', 'stands', 'throws', 'pitcher', 'catcher', 'batter',
                'stadium', 'referee', 'balls', 'strikes', 'home', 'away', 'inning_topbot', 'game_date', 'pitch_type']

    sub_df = df.loc[df.pitch_result.isin(['스트라이크', '볼']) &
                    (df.stands != 'None') & (~df.stands.isnull()) &
                    (df.throws != 'None') & (~df.throws.isnull())]
    sub_df = clean_data(sub_df)
    sub_df = sub_df.assign(stands = np.where(sub_df.stands == '양',
                                     np.where(sub_df.throws == '좌', '우', '좌'),
                                     sub_df.stands))
    sub_df = sub_df.rename(index=str, columns={'pos_2':'catcher'})
    logs = sub_df[features]

    logs = logs.assign(pitch_result=np.where(logs.pitch_result=='스트라이크', 1, 0))
    strikes = logs.loc[logs.pitch_result == 1]
    balls = logs.loc[logs.pitch_result == 0]

    lb = -1.5
    rb = +1.5
    bb = 1.0
    tb = 4.0

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
    
    # Run Value
    rv = None
    if rv_by_count is False:
        b, s = get_rv_of_ball_strike(df)
        rv = b - s
    else:
        b, s = get_rv_of_ball_strike_by_count(df)
        rv = b - s

    logs = logs.assign(excall=np.where(logs.prediction!=logs.pitch_result,
                                       np.where(logs.pitch_result==1, 1, -1),0))
    logs = logs.assign(exstr=np.where(logs.excall==1, 1, 0))
    logs = logs.assign(exball=np.where(logs.excall==-1, 1, 0))

    # Run Value
    if rv_by_count is False:
        logs = logs.assign(exrv = logs.excall * rv)
        logs = logs.assign(exrv_prob = np.where(logs.excall==1, (1-logs.proba)*rv,
                                                np.where(logs.excall==-1, -logs.proba*rv, 0)))
    else:
        logs = logs.assign(exrv = rv[logs.strikes*4 + logs.balls].values)
        logs = logs.assign(exrv = logs.excall * logs.exrv)
        logs = logs.assign(exrv_prob_plus = (1-logs.proba)*logs.exrv,
                           exrv_prob_minus = logs.proba*logs.exrv)
        logs = logs.assign(exrv_prob = np.where(logs.excall==1, logs.exrv_prob_plus,
                                                np.where(logs.excall==-1, logs.exrv_prob_minus, 0)))


    # 포수, catch 개수, extra strike, extra ball, RV sum
    tab = logs.pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum',
                                    'exball': 'sum',
                                    'excall': 'sum',
                                    'exrv': 'sum',
                                    'exrv_prob':'sum',
                                    'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)

    return logs, tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)


def calc_framing_cell_adv(df, rv_by_count=False, max_dist=0.25):
    lb = -1.5
    rb = +1.5
    bb = 1.0
    tb = 4.0

    sub_df = clean_data(df)
    strikes = sub_df.loc[sub_df.pitch_result == '스트라이크']
    balls = sub_df.loc[sub_df.pitch_result == '볼']

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

    logs, _ = calc_framing_cell(sub_df, rv_by_count)
    logs = logs.assign(dist = np.min(sp.spatial.distance.cdist(logs[['px', 'pz']], cts), axis=1))

    tab = logs.loc[logs.dist < max_dist].pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum', 'exball': 'sum', 'excall': 'sum', 'exrv': 'sum', 'exrv_prob':'sum', 'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)

    return logs.loc[logs.dist < max_dist], tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)


def calc_framing_gam(df, rv_by_count=False):
    # 10-80% 구간 측정이 mean을 0에 가깝게 맞출 수 있음.
    sub_df = df.loc[df.pitch_result.isin(['스트라이크', '볼']) &
                    (df.stands != 'None') & (~df.stands.isnull()) &
                    (df.throws != 'None') & (~df.throws.isnull())]
    sub_df = clean_data(sub_df)
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

    logs = sub_df[features + label + ['pos_1', 'pos_2', 'balls', 'strikes', 'stadium', 'batter',
                                      'home', 'away', 'inning_topbot', 'game_date', 'pitch_type']]
    logs = logs.rename(index=str, columns={'pos_1': 'pitcher', 'pos_2':'catcher'})

    logs = logs.assign(prediction = predictions)
    logs = logs.assign(proba = proba)

    # Run Value
    rv = None
    if rv_by_count is False:
        b, s = get_rv_of_ball_strike(df)
        rv = b - s
    else:
        b, s = get_rv_of_ball_strike_by_count(df)
        rv = b - s

    logs = logs.assign(excall=np.where(logs.prediction!=logs.pitch_result,
                                       np.where(logs.pitch_result==1, 1, -1),0))
    logs = logs.assign(exstr=np.where(logs.excall==1, 1, 0))
    logs = logs.assign(exball=np.where(logs.excall==-1, 1, 0))

    # Run Value
    if rv_by_count is False:
        logs = logs.assign(exrv = logs.excall * rv)
        logs = logs.assign(exrv_prob = np.where(logs.excall==1, (1-logs.proba)*rv,
                                                np.where(logs.excall==-1, -logs.proba*rv, 0)))
    else:
        logs = logs.assign(exrv = rv[logs.strikes*4 + logs.balls].values)
        logs = logs.assign(exrv = logs.excall * logs.exrv)
        logs = logs.assign(exrv_prob_plus = (1-logs.proba)*logs.exrv,
                           exrv_prob_minus = logs.proba*logs.exrv)
        logs = logs.assign(exrv_prob = np.where(logs.excall==1, logs.exrv_prob_plus,
                                                np.where(logs.excall==-1, logs.exrv_prob_minus, 0)))


    # 포수, catch 개수, extra strike, extra ball, RV sum
    tab = logs.pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum',
                                    'exball': 'sum',
                                    'excall': 'sum',
                                    'exrv': 'sum',
                                    'exrv_prob':'sum',
                                    'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)

    return logs, tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)


def calc_framing_gam_adv(df, rv_by_count=False, max_dist=0.25):
    lb = -1.5
    rb = +1.5
    bb = 1.0
    tb = 4.0

    sub_df = clean_data(df)
    strikes = sub_df.loc[sub_df.pitch_result == '스트라이크']
    balls = sub_df.loc[sub_df.pitch_result == '볼']

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

    logs, _ = calc_framing_gam(sub_df, rv_by_count)
    logs = logs.assign(dist = np.min(sp.spatial.distance.cdist(logs[['px', 'pz']], cts), axis=1))

    tab = logs.loc[logs.dist < max_dist].pivot_table(index='catcher',
                           values=['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'px'],
                           aggfunc={'exstr': 'sum', 'exball': 'sum', 'excall': 'sum', 'exrv': 'sum', 'exrv_prob':'sum', 'px':'count'})
    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('num', ascending=False)

    return logs.loc[logs.dist < max_dist], tab[['num', 'excall', 'exstr', 'exball', 'exrv', 'exrv_prob']].sort_values('excall', ascending=False)


def get_framing_venue_adjustment(df):
    ep_row = []

    for team in df.home.drop_duplicates():
        in_venue = df.loc[df.home == team]
        venue = list(in_venue.stadium.drop_duplicates())[0]
        out_venue = df.loc[df.away == team]

        es_in = in_venue.exstr.sum() / len(in_venue)
        eb_in = in_venue.exball.sum() / len(in_venue)
        v_es_in = es_in * (1-es_in) / len(in_venue)
        v_eb_in = eb_in * (1-eb_in) / len(in_venue)

        es_out = out_venue.exstr.sum() / len(out_venue)
        eb_out = out_venue.exball.sum() / len(out_venue)
        v_es_out = es_out * (1-es_out) / len(out_venue)
        v_eb_out = eb_out * (1-eb_out) / len(out_venue)

        es_adj = (es_in / v_es_in + es_out / v_es_out) / (1/v_es_in + 1/v_es_out)
        eb_adj = (eb_in / v_eb_in + eb_out / v_eb_out) / (1/v_eb_in + 1/v_eb_out)

        ep_row.append([team, es_adj-es_out, eb_adj-eb_out])

    ep_row.sort(key=lambda x:x[0])
    
    return {x[0]: x[1] for x in ep_row}, {x[0]:x[2] for x in ep_row}


def get_framing_pitcher_adjustment(df):
    ep_row = []
    for pit in df.pitcher.drop_duplicates():
        w_p = df.loc[df.pitcher == pit]
        catchers = w_p.catcher.drop_duplicates()
        wo_p = df.loc[df.catcher.isin(catchers) & (df.pitcher != pit)]

        wes = w_p.exstr.sum() / len(w_p)
        v_wes = wes * (1-wes) / len(w_p)
        woes = wo_p.exstr.sum() / len(wo_p)
        v_woes = woes * (1-woes) / len(w_p)

        web = w_p.exball.sum() / len(w_p)
        v_web = web * (1-web) / len(w_p)
        woeb = wo_p.exball.sum() / len(wo_p)
        v_woeb = woeb * (1-woeb) / len(w_p)

        adj_es = (wes/v_wes + woes/v_woes)/(1/v_wes + 1/v_woes)
        adj_eb = (web/v_web + woeb/v_woeb)/(1/v_web + 1/v_woeb)

        ep_row.append([pit, adj_es-woes, adj_eb-woeb])

    ep_row.sort(key=lambda x:x[0])
    
    return {x[0]: x[1] for x in ep_row}, {x[0]:x[2] for x in ep_row}


def adjust_framing(df, rv=None):
    r1, r2 = get_framing_venue_adjustment(df)
    r3, r4 = get_framing_pitcher_adjustment(df)
    df = df.assign(esadj_venue = df.home.map(r1),
                   ebadj_venue = df.home.map(r2),
                   esadj_pitcher = df.pitcher.map(r3),
                   ebadj_pitcher = df.pitcher.map(r4))
    df = df.assign(esadj = df.esadj_venue + df.esadj_pitcher,
                   ebadj = df.ebadj_venue + df.ebadj_pitcher)

    if rv is None:
        rv = get_rv_of_ball_strike(df)
    df = df.assign(exrv_adj = df.exrv - (df.esadj - df.ebadj)*rv)

    return df


def get_framing_run(df):
    _1, _2 = get_rv_of_ball_strike(df)
    rv = _1 - _2

    logs, _ = calc_framing_gam_adv(df)

    logs = adjust_framing(logs, rv)

    logs = logs.assign(exrv_prob_plus_adj = (1-logs.proba)*logs.exrv_adj,
                       exrv_prob_minus_adj = logs.proba*logs.exrv_adj)
    logs = logs.assign(exrv_prob_adj = np.where(logs.excall==1, logs.exrv_prob_plus_adj,
                                       np.where(logs.excall==-1, logs.exrv_prob_minus_adj, 0)))

    tab = logs.pivot_table(['exstr', 'exball', 'excall', 'exrv', 'exrv_prob', 'exrv_adj', 'exrv_prob_adj', 'px'],
                           'catcher',
                           None, # no column
                           {'exstr': 'sum', 'exball': 'sum', 'excall': 'sum',
                            'exrv': 'sum', 'exrv_prob': 'sum', 'exrv_adj': 'sum', 'exrv_prob_adj': 'sum',
                            'px': 'count'}, 0)

    tab = tab.rename(index=str, columns={'px': 'num'}).sort_values('excall', ascending=False)

    return logs, tab


def plot_table_color_scale(df, columns=None):
    N = 128
    top = np.ones((N, 4))
    bottom = np.ones((N, 4))
    top[:, 0] = np.linspace(1, 68/256, N)
    top[:, 1] = np.linspace(1, 189/256, N)
    top[:, 2] = np.linspace(1, 90/256, N)

    bottom[:, 0] = np.linspace(235/256, 1, N)
    bottom[:, 1] = np.linspace(91/256, 1, N)
    bottom[:, 2] = np.linspace(91/256, 1, N)

    newcolors = np.vstack([bottom, top])

    newcmp = ListedColormap(newcolors)

    return df.style.background_gradient(newcmp, subset=columns)


def shadow(df):
    target = df.loc[(df.px.between(-40/3/12, 40/3/12) & df.pz.between(14/12, 22/12)) |
                    (df.px.between(-40/3/12, 40/3/12) & df.pz.between(38/12, 46/12)) |
                    (df.px.between(-40/3/12, -20/3/12) & df.pz.between(14/12, 46/12)) |
                    (df.px.between(20/3/12, 40/3/12) & df.pz.between(14/12, 46/12))]

    strikes = target.loc[target.pitch_result == 1]

    t1 = target.pivot_table(index='catcher', values='exrv_adj', aggfunc='count')
    ts = strikes.pivot_table(index='catcher', values='exrv_adj', aggfunc='count')
    t2 = strikes.pivot_table(index='catcher', values='exrv_adj', aggfunc='count')

    shadow_avg = ts.sum()/t1.sum()*100

    t2['exrv_adj'] = t2.div(t1).mul(100)
    t2['shadow% above avg'] = t2['exrv_adj'].apply(lambda x:x- shadow_avg)
    t2 = t2.rename(index=str, columns={'exrv_adj': 'shadow%'})
    return t2


def frun_style(frun_df, threshold=500, shadow_df=None):
    t = frun_df.loc[frun_df.num > threshold][['num', 'exrv_adj']]
    t = t.assign(이름 = t.index.to_series())
    t = t.assign(fRun2000 = t.exrv_adj / t.num * 2000)
    t = t.sort_values('exrv_adj', ascending=False)
    t = t.rename(columns={'num': '판정 횟수', 'exrv_adj': 'fRun', 'fRun2000': 'fRun/2000'})

    columns = ['fRun', 'fRun/2000']
    if shadow_df is not None:
        t = t.join(shadow(shadow_df))
        t = t[['이름', '판정 횟수', 'fRun', 'fRun/2000', 'shadow%', 'shadow% above avg']]
        columns = ['fRun', 'fRun/2000', 'shadow%', 'shadow% above avg']
    else:
        t = t[['이름', '판정 횟수', 'fRun', 'fRun/2000']]

    styles = [
        dict(selector="th", props=[("font-size", "12pt"),
                                   ("text-align", "center"),
                                   ("background-color", "#616161"),
                                   ("color", "white")]),
        dict(selector="td", props=[("font-size", "10pt"),
                                   ("text-align", "center")])
    ]
    s = plot_table_color_scale(t, columns)

    if shadow_df is not None:
        s = s.hide_index().format({'fRun': "{:.1f}", 'fRun/2000': "{:.1f}",
                                   'shadow%': "{:.1f}", 'shadow% above avg': "{:.1f}"}).set_table_styles(styles)
    else:
        s = s.hide_index().format({'fRun': "{:.1f}", 'fRun/2000': "{:.1f}"}).set_table_styles(styles)
    return s


def soundAlert1():
    display(Audio(url='https://sound.peal.io/ps/audios/000/000/537/original/woo_vu_luvub_dub_dub.wav', autoplay=True))


def soundAlert2():
    display(Audio(url='https://sound.peal.io/ps/audios/000/000/430/original/liljon_3.mp3', autoplay=True))
