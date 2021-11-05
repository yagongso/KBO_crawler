import os, json, regex, csv, sys, traceback, pathlib

import pandas as pd
import numpy as np

header_row = ['pitch_type', 'pitcher', 'batter', 'pitcher_ID', 'batter_ID',
              'speed', 'pitch_result', 'pa_result', 'pa_result_detail',
              'description', 'balls', 'strikes', 'outs',
              'inning', 'inning_topbot', 'score_away', 'score_home',
              'stands', 'throws', 'on_1b', 'on_2b', 'on_3b', 'pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5',
              'pos_6', 'pos_7', 'pos_8', 'pos_9',
              'px', 'pz', 'pfx_x', 'pfx_z', 'pfx_x_raw', 'pfx_z_raw',
              'x0', 'z0', 'sz_top', 'sz_bot',
              'y0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
              'game_date', 'home', 'away', 'home_alias', 'away_alias',
              'stadium', 'referee', 'pa_number', 'pitch_number', 'pitchID', 'gameID']

# 원본 텍스트 / pa_result 기록 텍스트 / 더 디테일한 pa_result_detail 기록 텍스트
batter_result = [
    ['삼진', '삼진', '삼진'],
    ['볼넷', '볼넷', '볼넷'],
    ['자동 고의 4구', '자동 고의4구', '자동 고의4구'],
    ['자동 고의4구', '자동 고의4구', '자동 고의4구'],
    ['고의4구', '고의4구', '고의4구'],
    ['몸에', '몸에 맞는 볼', '몸에 맞는 볼'],
    ['1루타', '안타', '안타'],
    ['내야안타', '내야안타', '내야안타'],
    ['번트안타', '번트 안타', '번트 안타'],
    ['안타', '안타', '안타'],
    ['2루타', '2루타', '2루타'],
    ['3루타', '3루타', '3루타'],
    ['홈런', '홈런', '홈런'],
    ['낫아웃 폭투', '낫아웃 출루', '낫아웃 폭투'],
    ['낫아웃 포일', '낫아웃 출루', '낫아웃 포일'],
    ['낫 아웃', '삼진', '낫아웃 삼진'],
    ['낫아웃 다른주자 수비 실책', '낫아웃 출루', '낫아웃 다른 주자 수비 실책'],
    ['낫아웃 다른주자 수비', '낫아웃 출루', '낫아웃 다른 주자 포스 아웃'],
    ['낫아웃 출루', '낫아웃 다른 주자 수비', '낫아웃 다른 주자 수비'],
    ['땅볼로 출루', '포스 아웃', '땅볼 아웃'],
    ['땅볼 아웃', '포스 아웃', '땅볼 아웃'],
    ['인필드', '필드 아웃', '인필드 플라이'],
    ['파울플라이', '필드 아웃', '파울 플라이 아웃'],
    ['라인드라이브 아웃', '필드 아웃', '라인드라이브 아웃'],
    ['병살타', '병살타', '병살타'],
    ['희생번트 아웃', '희생번트', '희생번트'],
    ['희생플라이 아웃', '희생플라이', '희생플라이'],
    ['희생플라이아웃', '희생플라이', '희생플라이'],
    ['쓰리번트', '삼진', '쓰리번트 삼진'],
    ['타구맞음', '타구맞음 아웃', '타구맞음 아웃'],
    ['희생번트 실책', '실책', '희생번트 실책'],
    ['희생번트 야수선택', '야수 선택', '희생번트 야수선택'],
    [' 플라이 아웃', '필드 아웃', '플라이 아웃'],
    [' 번트 아웃', '필드 아웃', '번트 아웃'],
    ['야수선택', '야수 선택', '야수 선택'],
    ['실책', '실책', '실책'],
    ['타격방해', '타격방해', '타격방해'],
    ['삼중살', '삼중살', '삼중살'],
    ['부정타격', '필드 아웃', '부정 타격 아웃'],
    ['번트', '번트 안타', '안타'],
]


def parse_batter_result(text):
    for tup in batter_result:
        if text.find(tup[0]) >= 0:
            return tup[1], tup[2]


def parse_batter_as_runner(text):
    runner = text.split(' ')[0]

    result = 'o' if text.find('아웃') > 0 else 'a'
    result = 'h' if text.find('홈런') > 0 else result
    # 낫아웃 예외처리
    if result == 'o':
        if text.find('폭투') >= 0:
            result = 'a'
        elif text.find('포일') >= 0:
            result = 'a'
        elif text.find('다른주자') >= 0:
            result = 'a'
        elif text.find('출루') >= 0:
            result = 'a'
        elif text.find('실책') >= 0:
            result = 'a'

    before_base = 0
    after_base = None
    if any([s in text for s in ['안타', '1루타', '4구', '볼넷', '출루',
                                '낫아웃 포일', '낫아웃 폭투', '낫아웃 다른주자',
                                '몸에 맞는', '실책', '타격방해', '야수선택']]):
        after_base = 1
    elif '2루타' in text:
        after_base = 2
    elif '3루타' in text:
        after_base = 3
    elif '홈런' in text:
        after_base = 4
    elif '아웃' in text:
        after_base = None
    
    return [runner, result, before_base, after_base]


def parse_runner_result(text):
    runner = text.split(' ')[1]

    result = 'o' if text.find('아웃') > 0 else 'a' # o : out, a : advance
    result = 'h' if text.find('홈인') > 0 else result # 'h' : home-in

    before_base = int(text[0])
    after_base = None if result != 'a' else int(text[text.find('루까지')-1])
    after_base = 4 if result == 'h' else after_base

    return [runner, result, before_base, after_base]


def get_pitch_location_break(row):
    if np.isnan(row[17]):
        return [None]*6
    else:
        ax = row[16]
        ay = row[18]
        az = row[19]
        vx0 = row[14]
        vy0 = row[12]
        vz0 = row[13]
        x0 = row[17]
        y0 = 50
        z0 = row[15]
        cpy = 1.4167

        # do math
        t = (-vy0 - (vy0 * vy0 - 2 * ay * (y0 - cpy)) ** 0.5) / ay

        t40 = (-vy0 - (vy0 * vy0 - 2 * ay * (y0 - 40)) ** 0.5) / ay
        x40 = x0 + vx0 * t40 + 0.5 * ax * t40 * t40
        vx40 = vx0 + ax * t40
        z40 = z0 + vz0 * t40 + 0.5 * az * t40 * t40
        vz40 = vz0 + az * t40
        th = t - t40
        x_no_air = x40 + vx40 * th
        z_no_air = z40 + vz40 * th - 0.5 * 32.174 * th * th
        z_no_induced = z0 + vz0 * t

        px = x0 + vx0 * t + ax * t * t * 0.5
        pz = z0 + vz0 * t + az * t * t * 0.5

        pfx_x = (px - x_no_air) * 12
        pfx_z = (pz - z_no_air) * 12
        pfx_x_raw = px * 12
        pfx_z_raw = (pz - z_no_induced) * 12

        return px, pz, pfx_x, pfx_z, pfx_x_raw, pfx_z_raw


class game_status:
    def __init__ (self):
        # game data frame
        self.pitching_df = None
        self.batting_df = None
        self.relay_array = None

        self.home_batter_list = None
        self.home_pitcher_list = None
        self.away_batter_list = None
        self.away_pitcher_list = None

        # game meta data
        self.game_date = None
        self.away = None
        self.home = None
        self.away_alias = None
        self.home_alias = None
        self.stadium = None
        self.referee = None
        self.game_id = None

        # current game numbers, names, codes, etc.
        self.score = [0, 0]
        self.batter_name = None
        self.batter_code = -1
        self.pitcher_name = None
        self.pitcher_code = -1
        self.stands = None
        self.throws = None
        self.DH_exist = [True, True]
        self.DH_exist_after = [True, True]

        # lineup(order), fielding position, base
        self.lineups = [[], []] # 초 공격 / 말 공격
        self.fields = [{}, {}] # 초 수비 / 말 수비
        self.runner_bases = []

        # status change at pa, inning
        self.pa_number = 0
        self.inn = 0
        self.top_bot = 1 # 0 : 초, 1 : 말
        self.cur_order = 1

        # status by pitch, etc.
        self.pitch_number = 0
        self.score = [0, 0] # 초공(어웨이), 말공(홈)
        self.balls, self.outs, self.strikes = 0, 0, 0
        self.last_pitch = None

        # pitch result, pa result
        self.pitch_result = None
        self.pa_result = None
        self.pa_result_detail = None
        self.description = ''        

        # print rows
        self.print_rows = []

        # log text
        self.log_file = None
        self.log_text = []

        # error status
        self.change_error = False

        # debug
        self.ind = 0
        self.cur_row = None
        self.cur_text = ''
        self.text_stack = None

    def load(self, game_id, pdf, bdf, rdf, log_file=None):
        self.pitching_df = pdf
        self.batting_df = bdf
        self.game_id = game_id
        self.log_file = log_file

        self.game_date = game_id[:8]
        self.away = game_id[8:10]
        self.home = game_id[10:12]
        self.stadium = rdf.stadium.unique()[0]
        self.referee = rdf.referee.unique()[0]
        self.away_alias = pdf.loc[pdf.homeaway == 'a'].team_name.unique()[0]
        self.home_alias = pdf.loc[pdf.homeaway == 'h'].team_name.unique()[0]
        if 'pCode' in pdf.columns:
            pdf = pdf.rename(index=str, columns={'pCode': 'pcode'})
        if 'pCode' in bdf.columns:
            bdf = bdf.rename(index=str, columns={'pCode': 'pcode'})

        ########################
        # 라인업 & 필드 채우기 #
        ########################

        bats = [bdf.loc[bdf.homeaway == 'a'],
                bdf.loc[bdf.homeaway == 'h']]
        pits = [pdf.loc[pdf.homeaway == 'a'],
                pdf.loc[pdf.homeaway == 'h']]
        abat_seqno_min = bats[0].loc[bats[0].batOrder.between(1, 9)].groupby('batOrder').seqno.min().tolist()
        hbat_seqno_min = bats[1].loc[bats[1].batOrder.between(1, 9)].groupby('batOrder').seqno.min().tolist()

        for i in range(9):
            away_code = bats[0].loc[(bats[0].batOrder == i+1) &
                                    (bats[0].seqno == abat_seqno_min[i])].pcode.values[0]
            away_name = bats[0].loc[(bats[0].batOrder == i+1) &
                                    (bats[0].seqno == abat_seqno_min[i])].name.values[0]
            home_code = bats[1].loc[(bats[1].batOrder == i+1) &
                                    (bats[1].seqno == hbat_seqno_min[i])].pcode.values[0]
            home_name = bats[1].loc[(bats[1].batOrder == i+1) &
                                    (bats[1].seqno == hbat_seqno_min[i])].name.values[0]
            away_hittype = bats[0].loc[bats[0].pcode == away_code].hitType.values[0]
            home_hittype = bats[1].loc[bats[1].pcode == home_code].hitType.values[0]
            away_player = {'name': away_name, 'code': away_code, 'hitType': away_hittype}
            home_player = {'name': home_name, 'code': home_code, 'hitType': home_hittype}

            away_pos = bats[0].loc[bats[0].pcode == away_code].posName.values[0]
            home_pos = bats[1].loc[bats[1].pcode == home_code].posName.values[0]
            self.fields[1][away_pos] = away_player
            self.fields[0][home_pos] = home_player

            away_lineup = {'name': away_name, 'code': int(away_code), 'pos': away_pos, 'hitType': away_hittype}
            home_lineup = {'name': home_name, 'code': int(home_code), 'pos': home_pos, 'hitType': home_hittype}
            self.lineups[0].append(away_lineup)
            self.lineups[1].append(home_lineup)

        away_pitcher = {'name': pits[0].iloc[0]['name'],
                        'code': int(pits[0].iloc[0].pcode),
                        'hitType': pits[0].iloc[0].hitType}
        home_pitcher = {'name': pits[1].iloc[0]['name'],
                        'code': int(pits[1].iloc[0].pcode),
                        'hitType': pits[1].iloc[0].hitType}
        self.fields[1]['투수'] = away_pitcher
        self.fields[0]['투수'] = home_pitcher

        home_bdf = bdf.loc[bdf.homeaway == 'h']
        away_bdf = bdf.loc[bdf.homeaway == 'a']
        home_pdf = pdf.loc[pdf.homeaway == 'h']
        away_pdf = pdf.loc[pdf.homeaway == 'a']

        batter_list_cols = ['name', 'pcode', 'posName', 'hitType', 'batOrder', 'seqno']
        pitcher_list_cols = ['name', 'pcode', 'hitType', 'seqno']

        self.home_batter_list = home_bdf[batter_list_cols].values.tolist()
        self.home_pitcher_list = home_pdf[pitcher_list_cols].values.tolist()
        self.away_batter_list = away_bdf[batter_list_cols].values.tolist()
        self.away_pitcher_list = away_pdf[pitcher_list_cols].values.tolist()

        rdf_cols = ['textOrder', 'seqno', 'text', 'type',
                    'stuff', 'pitchId', 'speed', 'referee', 'stadium']
        if 'x0' in rdf.columns:
            rdf_cols += ['crossPlateX', 'topSz', 'bottomSz',
                         'vy0', 'vz0', 'vx0', 'z0', 'ax', 'x0', 'ay', 'az']

        for col in ['outPlayer', 'inPlayer', 'shiftPlayer']:
            if col in rdf.columns:
                rdf_cols.append(col)

        self.relay_array = rdf.loc[rdf[['textOrder', 'seqno']].drop_duplicates().index][rdf_cols].sort_values(['textOrder', 'seqno']).values


    def convert_row_to_save_format(self, row,
                                   pa_result_details=None):
        # row: pandas Series
        save_row = {k: None for k in header_row}
        save_row['pitcher'] = self.pitcher_name
        save_row['batter'] = self.batter_name
        save_row['pitcher_ID'] = self.pitcher_code
        save_row['batter_ID'] = self.batter_code
        save_row['balls'] = self.balls
        save_row['strikes'] = self.strikes
        save_row['outs'] = self.outs
        save_row['inning'] = self.inn
        save_row['inning_topbot'] = '초' if self.top_bot == 0 else '말'
        save_row['score_away'] = self.score[0]
        save_row['score_home'] = self.score[1]
        if (self.stands is not None) and (type(self.stands) == str):
            if len(self.stands) > 2:
                save_row['stands'] = self.stands[2]
        if (self.throws is not None) and (type(self.throws) == str):
            if len(self.throws) > 2:
                save_row['throws'] = self.throws[0]
        save_row['pa_number'] = self.pa_number

        save_row['game_date'] = self.game_date
        save_row['home'] = self.home
        save_row['away'] = self.away
        save_row['home_alias'] = self.home_alias
        save_row['away_alias'] = self.away_alias
        save_row['stadium'] = self.stadium
        save_row['referee'] = self.referee
        save_row['gameID'] = self.game_id

        if self.change_error == False:
            for runner in self.runner_bases:
                if runner[2] > 0:
                    save_row[f'on_{runner[2]}b'] = runner[0]
        else:
            for i in range(1, 4):
                save_row[f'on_{i}b'] = None

        if self.change_error == False:
            save_row['pos_1'] = self.fields[self.top_bot]['투수'].get('name')
            save_row['pos_2'] = self.fields[self.top_bot]['포수'].get('name')
            save_row['pos_3'] = self.fields[self.top_bot]['1루수'].get('name')
            save_row['pos_4'] = self.fields[self.top_bot]['2루수'].get('name')
            save_row['pos_5'] = self.fields[self.top_bot]['유격수'].get('name')
            save_row['pos_6'] = self.fields[self.top_bot]['3루수'].get('name')
            save_row['pos_7'] = self.fields[self.top_bot]['좌익수'].get('name')
            save_row['pos_8'] = self.fields[self.top_bot]['중견수'].get('name')
            save_row['pos_9'] = self.fields[self.top_bot]['우익수'].get('name')
        else:
            for i in range(1, 10):
                save_row[f'pos_{i}'] = None

        if row is not None:
            save_row['pitch_type'] = row[4]
            save_row['speed'] = row[6]
            save_row['pitch_result'] = row[2].split(' ')[-1]
            save_row['pitch_number'] = self.pitch_number
            save_row['pitchID'] = row[5]

            if len(row) > 13:
                if not np.isnan(row[17]):
                    save_row['x0'] = row[17]
                    save_row['z0'] = row[15]
                    save_row['sz_top'] = row[10]
                    save_row['sz_bot'] = row[11]
                    px, pz, pfx_x, pfx_z, pfx_x_raw, pfx_z_raw = get_pitch_location_break(row)

                    save_row['px'] = px
                    save_row['pz'] = pz
                    save_row['pfx_x'] = pfx_x
                    save_row['pfx_z'] = pfx_z
                    save_row['pfx_x_raw'] = pfx_x_raw
                    save_row['pfx_z_raw'] = pfx_z_raw

                    save_row['y0'] = 50
                    save_row['vx0'] = row[14]
                    save_row['vy0'] = row[12]
                    save_row['vz0'] = row[13]
                    save_row['ax'] = row[16]
                    save_row['ay'] = row[18]
                    save_row['az'] = row[19]

        if pa_result_details is not None:
            save_row['description'] = pa_result_details[0]
            save_row['pa_result'] = pa_result_details[1]
            save_row['pa_result_detail'] = pa_result_details[2]

        return save_row

    def handle_runner_stack(self, runner_stack, debug_mode=False):
        if len(self.runner_bases) == 0:
            if debug_mode == True:
                if self.change_error == False:
                    self.log_text.append('base error: No runners but calling handle_runner_stack')
                    for row in runner_stack:
                        self.log_text.append(f'text - {row}')
                    self.change_error = True
            cur_runner = None
        else:
            cur_runner = self.runner_bases[0]
        cur_runner_ind = 0

        batter_runner = False
        for row in runner_stack[::-1]:
            if (row[3] == 13) or (row[3] == 23):
                runner_name, run_result, runner_before_base, runner_after_base = parse_batter_as_runner(row[2])
                batter_runner = True
            else:
                runner_name, run_result, runner_before_base, runner_after_base = parse_runner_result(row[2])

            if self.change_error == False:
                base_loop_num = 0
                while not ((cur_runner[0] == runner_name) &\
                            (cur_runner[2] <= runner_before_base) &\
                            (cur_runner[3][-1] >= runner_before_base)):
                    cur_runner_ind = cur_runner_ind + 1
                    if cur_runner_ind >= len(self.runner_bases):
                        cur_runner_ind = 0
                    cur_runner = self.runner_bases[cur_runner_ind]
                    base_loop_num += 1
                    if base_loop_num > 4:
                        if debug_mode == True:
                            if self.change_error == False:
                                self.log_text.append('base error: cannot find runner in bases')
                                for row in runner_stack:
                                    self.log_text.append(f'text - {row}')
                                self.change_error = True
                        break

                if cur_runner[3][-1] == 5:
                    if runner_after_base is not None:
                        cur_runner[3] = [runner_after_base, runner_before_base]
                    elif runner_before_base != 0:
                        cur_runner[3] = [runner_after_base, runner_before_base]
                else:
                    cur_runner[3].append(runner_before_base)

            if run_result == 'h':
                self.score[self.top_bot] += 1
            elif run_result == 'o':
                self.outs += 1

        if self.change_error == False:
            after_runner_bases = []

            for runner in self.runner_bases:
                # name, code, src, route
                if (batter_runner == True) & (runner[2] == 0) & (runner[3][0] == 5):
                    continue
                elif runner[3][0] == 5:
                    after_runner_bases.append(runner[:])
                elif (runner[3][0] is not None) & (runner[3][0] != 4):
                    after_runner_bases.append([runner[0], runner[1], runner[3][0], [5]])
            self.runner_bases = after_runner_bases[:]


    def handle_change_stack(self, change_stack):
        for i in range(len(change_stack)):
            change = change_stack[i]
            before_pos, after_pos, order, after_name, after_code, after_hittype = change
 
            # 투수, 야수(수비중) 교체
            if (after_pos != '대타') & (after_pos != '대주자'):
                self.fields[self.top_bot][after_pos]['code'] = after_code
                self.fields[self.top_bot][after_pos]['name'] = after_name
                self.fields[self.top_bot][after_pos]['hitType'] = after_hittype

            # 타자 교체
            if order is not None:
                # 대타
                if (after_pos != '대타') & (after_pos != '대주자'):
                    tb = 1 - self.top_bot
                # 대주자 또는 대수비
                else:
                    tb = self.top_bot

                self.lineups[tb][order - 1]['code'] = after_code
                self.lineups[tb][order - 1]['name'] = after_name
                self.lineups[tb][order - 1]['pos'] = after_pos
                self.lineups[tb][order - 1]['hitType'] = after_hittype

            if after_pos == '투수':
                self.pitcher_code = after_code
                self.pitcher_name = after_name
                self.throws = after_hittype
            elif after_pos == '대주자':
                before_base = int(before_pos[0])
                for j in range(len(self.runner_bases)):
                    if self.runner_bases[j][2] == before_base:
                        self.runner_bases[j][0] = after_name
                        self.runner_bases[j][1] = after_code
                        break
            elif after_pos == '대타':
                self.runner_bases[-1][0] = after_name
                self.runner_bases[-1][1] = after_code
                self.stands = after_hittype
        self.DH_exist[self.top_bot] = self.DH_exist_after[self.top_bot]


    def handle_change(self, text_stack, debug_mode=False):
        change_stack = []
        for text in text_stack:
            order = None
            before_pos = text.split(' ')[0]

            if text.find('변경') > 0:
                after_pos = text.split('(')[0].strip().split(' ')[-1]
                shift_name = text.split(' ')[1].strip()
                shift_code = None
                shift_hittype = None

                for i in range(9):
                    if (self.lineups[1 - self.top_bot][i].get('name') == shift_name) &\
                       (self.lineups[1 - self.top_bot][i].get('pos') == before_pos):
                        shift_code = self.lineups[1 - self.top_bot][i].get('code')
                        shift_hittype = self.lineups[1 - self.top_bot][i].get('hitType')
                        order = i + 1
                        break
                if before_pos == '지명타자':
                    self.DH_exist_after[self.top_bot] = False
                if after_pos == '투수':
                    self.DH_exist_after[self.top_bot] = False

                ########################################################################
                ##### 대타 출장 후 같은 이닝에 타순 1바퀴 돌면서 포지션 변경하는 경우
                ########################################################################
                if (before_pos == '대타') & (order == None):
                    for i in range(9):
                        if (self.lineups[self.top_bot][i].get('name') == shift_name) &\
                           (self.lineups[self.top_bot][i].get('pos') == before_pos):
                            self.lineups[self.top_bot][i]['pos'] = after_pos
                            break
                    continue

                change = [before_pos, after_pos, order, shift_name, shift_code, shift_hittype]
                change_stack.append(change)
            else:
                after_pos = text.split('(')[0].strip().split(' ')[3]
                after_name = text.split('(')[0].strip().split(' ')[-1]
                before_name = text.split(' ')[1].strip()
                before_code = None
                after_code = None
                after_hittype = None
                order = None

                if (before_pos == '지명타자') & (after_pos !='지명타자') & (after_pos !='대타'):
                    self.DH_exist_after[self.top_bot] = False
                elif (before_pos == '투수') & (after_pos != '투수'):
                    self.DH_exist_after[self.top_bot] = False
                elif (before_pos != '투수') & (after_pos == '투수'):
                    self.DH_exist_after[self.top_bot] = False

                if before_pos.find('번타자') > 0:
                    order = int(before_pos[0])
                    before_code = self.lineups[self.top_bot][order - 1].get('code')

                    # 대타 텍스트가 먼저 나오는 경우, 교체 선반영된 버그
                    # 이미 라인업/batter name에 교체된 선수 이름 들어가 있음
                    # 교체하는 대신, after_name이 현재 batter name과 같으면 continue한다
                    if (before_name != after_name) & (after_name == self.batter_name):
                        after_code = before_code
                        after_hittype = self.stands
                    else:
                        if self.top_bot == 0:
                            for i in range(len(self.away_batter_list) - 1):
                                if self.away_batter_list[i][1] == before_code:
                                    after_code = self.away_batter_list[i+1][1]
                                    after_hittype = self.away_batter_list[i+1][3]
                                    break
                        else:
                            for i in range(len(self.home_batter_list) - 1):
                                if self.home_batter_list[i][1] == before_code:
                                    after_code = self.home_batter_list[i+1][1]
                                    after_hittype = self.home_batter_list[i+1][3]
                                    break
                    if after_code is None:
                        if debug_mode == True:
                            self.log_text.append('cant find player code in batter list')
                            self.log_text.append(f'text - {text}')
                        if self.change_error == False:
                            self.change_error = True
                elif after_pos == '투수':
                    # 예외처리
                    # 기존 투수가 야수 자리에 들어가는 경우(지명타자 소멸)
                    # ex) 지명타자 OOO : 투수 XXX(으)로 교체
                    if (after_name == self.pitcher_name) & (before_pos != '투수'):
                        after_code = self.pitcher_code
                        after_hittype = self.throws
                    # 야수 자리에 새로 투수가 들어오는 경우(지명타자 소멸, 투수 교체)
                    # ex) 1루수 OOO : 투수 XXX(으)로 교체
                    elif (after_name != self.pitcher_name) & (before_pos != '투수'):
                        if self.top_bot == 0:
                            for i in range(len(self.home_pitcher_list) - 1):
                                if self.home_pitcher_list[i][1] == self.pitcher_code:
                                    after_code = self.home_pitcher_list[i+1][1]
                                    after_hittype = self.home_pitcher_list[i+1][2]
                                    break
                        else:
                            for i in range(len(self.away_pitcher_list) - 1):
                                if self.away_pitcher_list[i][1] == self.pitcher_code:
                                    after_code = self.away_pitcher_list[i+1][1]
                                    after_hittype = self.away_pitcher_list[i+1][2]
                                    break
                    else:
                        before_code = self.fields[self.top_bot][before_pos].get('code')
                        if self.top_bot == 0:
                            for i in range(len(self.home_pitcher_list) - 1):
                                if self.home_pitcher_list[i][1] == before_code:
                                    after_code = self.home_pitcher_list[i+1][1]
                                    after_hittype = self.home_pitcher_list[i+1][2]
                                    break
                        else:
                            for i in range(len(self.away_pitcher_list) - 1):
                                if self.away_pitcher_list[i][1] == before_code:
                                    after_code = self.away_pitcher_list[i+1][1]
                                    after_hittype = self.away_pitcher_list[i+1][2]
                                    break
                    if after_code is None:
                        if debug_mode == True:
                            self.log_text.append('cant find player code in pitcher list')
                            self.log_text.append(f'text - {text}')
                        if self.change_error == False:
                            self.change_error = True
                    if ((self.DH_exist[self.top_bot] is False) or\
                        (self.DH_exist_after[self.top_bot] != self.DH_exist[self.top_bot])):
                        for i in range(9):
                            if (self.lineups[1 - self.top_bot][i].get('name') == before_name) &\
                               (self.lineups[1 - self.top_bot][i].get('pos') == before_pos):
                                order = i + 1
                                break
                elif before_pos.find('루주자') > 0:
                    # 대주자 교체
                    before_base = int(before_pos[0])
                    for runner in self.runner_bases:
                        if (runner[0] == before_name) & (runner[2] == before_base):
                            before_code = runner[1]
                            break
                    for i in range(9):
                        if self.lineups[self.top_bot][i].get('code') == before_code:
                            order = i + 1
                            break
                    if self.top_bot == 0:
                        for i in range(len(self.away_batter_list) - 1):
                            if self.away_batter_list[i][1] == before_code:
                                after_code = self.away_batter_list[i+1][1]
                                after_hittype = self.away_batter_list[i+1][3]
                                break
                    else:
                        for i in range(len(self.home_batter_list) - 1):
                            if self.home_batter_list[i][1] == before_code:
                                after_code = self.home_batter_list[i+1][1]
                                after_hittype = self.home_batter_list[i+1][3]
                                break
                    if after_code is None:
                        if debug_mode == True:
                            self.log_text.append('cant find player code in batter list')
                            self.log_text.append(f'text - {text}')
                        if self.change_error == False:
                            self.change_error = True
                else:
                    # 대타 교체
                    for i in range(9):
                        if (self.lineups[1 - self.top_bot][i].get('name') == before_name) &\
                           (self.lineups[1 - self.top_bot][i].get('pos') == before_pos):
                            before_code = self.lineups[1 - self.top_bot][i].get('code')
                            order = i + 1
                            break
                    if order is None:
                        for i in range(9):
                            if (self.lineups[self.top_bot][i].get('name') == before_name) &\
                               (self.lineups[self.top_bot][i].get('pos') == before_pos):
                                before_code = self.lineups[self.top_bot][i].get('code')
                                order = i + 1
                                break
                    # 대타 교체가 이어지는 경우
                    # 선입력된 change stack을 처리하고(lineup에 반영)
                    # 다시 대타 text 처리를 시도함
                    if order is None:
                        try:
                            self.handle_change_stack(change_stack)
                            change_stack = []

                            for i in range(9):
                                if (self.lineups[1 - self.top_bot][i].get('name') == before_name) &\
                                   (self.lineups[1 - self.top_bot][i].get('pos') == before_pos):
                                    before_code = self.lineups[1 - self.top_bot][i].get('code')
                                    order = i + 1
                                    break
                            if order is None:
                                for i in range(9):
                                    if (self.lineups[self.top_bot][i].get('name') == before_name) &\
                                       (self.lineups[self.top_bot][i].get('pos') == before_pos):
                                        before_code = self.lineups[self.top_bot][i].get('code')
                                        order = i + 1
                                        break
                            # 대타 텍스트가 먼저 나오는 경우, 교체 선반영된 버그
                            # 이미 라인업/batter name에 교체된 선수 이름 들어가 있음
                            # 교체하는 대신, 현재 batter name과 같은 이름이 나오면 continue한다
                            if order is None:
                                already_change = False
                                for i in range(9):
                                    if (self.lineups[1 - self.top_bot][i].get('name') == after_name) &\
                                       (self.batter_name == after_name):
                                        already_change = True
                                        break
                                if already_change == True:
                                    continue
                                else:
                                    for i in range(9):
                                        if (self.lineups[self.top_bot][i].get('name') == after_name) &\
                                           (self.batter_name == after_name):
                                            already_change = True
                                            break
                                    if already_change == True:
                                        continue
                            if before_code is None:
                                if debug_mode == True:
                                    self.log_text.append('cant find player name/position in lineup')
                                    self.log_text.append(f'text - {text}')
                                if self.change_error == False:
                                    self.change_error = True
                        except:
                            if self.change_error == False:
                                self.change_error = True

                    # 라인업(타순)에서 교체
                    if self.top_bot == 0:
                        for i in range(len(self.home_batter_list) - 1):
                            if self.home_batter_list[i][1] == before_code:
                                after_code = self.home_batter_list[i+1][1]
                                after_hittype = self.home_batter_list[i+1][3]
                                break
                        if after_code is None:
                            for i in range(len(self.away_batter_list) - 1):
                                if self.away_batter_list[i][1] == before_code:
                                    after_code = self.away_batter_list[i+1][1]
                                    after_hittype = self.away_batter_list[i+1][3]
                                    break
                    else:
                        for i in range(len(self.away_batter_list) - 1):
                            if self.away_batter_list[i][1] == before_code:
                                after_code = self.away_batter_list[i+1][1]
                                after_hittype = self.away_batter_list[i+1][3]
                                break
                        if after_code is None:
                            for i in range(len(self.home_batter_list) - 1):
                                if self.home_batter_list[i][1] == before_code:
                                    after_code = self.home_batter_list[i+1][1]
                                    after_hittype = self.home_batter_list[i+1][3]
                                    break

                    # 주자인 경우(루상 주자와 대조) 베이스 교체
                    if after_code is None:
                        if debug_mode == True:
                            self.log_text.append('cant find player code in lineup')
                        if self.change_error == False:
                            self.change_error = True
                    for i in range(len(self.runner_bases)):
                        if self.runner_bases[i][1] == before_code:
                            self.runner_bases[i][0] = after_name
                            self.runner_bases[i][1] = after_code
                            break
                change = [before_pos, after_pos, order, after_name, after_code, after_hittype]
                change_stack.append(change)

        self.handle_change_stack(change_stack)


    def parse_game(self, debug_mode=False):
        try:
            self.ind = 0
            self.inn = 0
            self.top_bot = 1
            while self.ind < self.relay_array.shape[0]:
                row = self.relay_array[self.ind]
                cur_type = row[3]
                self.cur_text = row[2]
                cur_to = row[0]

                ##############################
                ###### type에 따라 파싱 ######
                ##############################

                if cur_type == 1:
                    self.last_pitch = row
                    self.pitch_number += 1
                    res = self.cur_text.split(' ')[-1]
                    self.pitch_result = res
                    # 인플레이/삼진/볼넷 이외에는 여기서 row print
                    # 몸에맞는공은 타석 결과에서 수정
                    if res != '타격':
                        save_row = self.convert_row_to_save_format(row)
                        self.print_rows.append(save_row)

                    if res == '볼':
                        self.balls += 1
                    elif res == '스트라이크':
                        self.strikes += 1
                    elif res.find('헛스윙') > -1:
                        self.strikes += 1
                    elif res.find('파울') > -1:
                        self.strikes = self.strikes+1 if self.strikes < 2 else 2
                    if (self.strikes > 3) or (self.balls > 4):
                        if debug_mode == True:
                            self.log_text.append('3S/4B')
                        assert False
                    self.ind = self.ind + 1

                elif cur_type == 8:
                    # 타석 시작(x번타자 / 대타)
                    position = self.cur_text.split(' ')[0]
                    self.last_pitch = None
                    self.pa_result = None
                    self.pitch_result = None
                    self.pa_result_detail = None
                    self.description = ''
                    if len(position) > 2:
                        self.cur_order = int(position[0])
                        if self.change_error == False:
                            self.batter_name, self.batter_code,\
                            _pos, self.stands = self.lineups[self.top_bot][self.cur_order - 1].values()
                        else:
                            self.batter_name = self.cur_text.split(' ')[1]
                            self.batter_code = None
                            self.stands = None
                        self.pitch_number = 0
                        self.pa_number += 1
                        self.balls, self.strikes = 0, 0

                        # 버그 : 대타 교체 텍스트가 누락된 경우. 20170902HTWO02017
                        # '대타 OOO' 텍스트가 들어왔는데, batter_name과 다름
                        # 교체된 것으로 간주, batter_name과 lineups의 이름과 code 등을 바꾼다
                        # 임시조치
                        if self.batter_name != self.cur_text.split(' ')[1]:
                            self.batter_name = self.cur_text.split(' ')[1]
                            self.lineups[self.top_bot][self.cur_order - 1]['name'] = self.batter_name

                            if self.top_bot == 0:
                                for p in self.away_batter_list:
                                    if (p[0] == self.batter_name) & (p[4] == self.cur_order):
                                        self.batter_code = p[1]
                                        self.stands = p[3]
                                        break
                            else:
                                for p in self.home_batter_list:
                                    if (p[0] == self.batter_name) & (p[4] == self.cur_order):
                                        self.batter_code = p[1]
                                        self.stands = p[3]
                                        break
                            self.lineups[self.top_bot][self.cur_order - 1]['code'] = self.batter_code
                            self.lineups[self.top_bot][self.cur_order - 1]['hitType'] = self.stands

                        self.runner_bases.append([self.batter_name, self.batter_code, 0, [5]])
                    else:
                        if self.change_error == False:
                            self.batter_name, self.batter_code,\
                            _pos, self.stands = self.lineups[self.top_bot][self.cur_order - 1].values()
                        else:
                            self.batter_name = self.cur_text.split(' ')[1]
                            self.batter_code = None
                            self.stands = None
                        # 버그 : 대타 교체 텍스트가 누락된 경우. 20170902HTWO02017
                        # '대타 OOO' 텍스트가 들어왔는데, batter_name과 다름
                        # 교체된 것으로 간주, batter_name과 lineups의 이름과 code 등을 바꾼다
                        # 임시조치
                        if self.batter_name != self.cur_text.split(' ')[1]:
                            if len(self.cur_text.strip().split(' ')) > 1:
                                self.batter_name = self.cur_text.strip().split(' ')[1]
                                self.lineups[self.top_bot][self.cur_order - 1]['name'] =self.batter_name
                                if self.top_bot == 0:
                                    for p in self.away_batter_list:
                                        if (p[0] == self.batter_name) & (p[4] == self.cur_order):
                                            self.batter_code = p[1]
                                            self.stands = p[3]
                                            break
                                else:
                                    for p in self.home_batter_list:
                                        if (p[0] == self.batter_name) & (p[4] == self.cur_order):
                                            self.batter_code = p[1]
                                            self.stands = p[3]
                                            break
                                self.lineups[self.top_bot][self.cur_order - 1]['code'] = self.batter_code
                                self.lineups[self.top_bot][self.cur_order - 1]['hitType'] = self.stands

                                for runner in self.runner_bases:
                                    if (runner[2] == 0) & (runner[0] != self.batter_name):
                                        runner[0] = self.batter_name
                                        runner[1] = self.batter_code
                                        break
                    self.ind = self.ind + 1
                elif (cur_type == 13) or (cur_type == 23):
                    # 타자주자(비득점/득점)
                    # 주자 처리 텍스트 row를 쭉 text_stack에 쌓는다
                    # 쌓은 row를 한번에 처리(handle_runner_stack)
                    self.text_stack = []
                    cur_ind = self.ind
                    self.cur_row = self.relay_array[cur_ind]
                    self.description = ''

                    while ((cur_type == 13) or (cur_type == 23) or\
                           (cur_type == 14) or (cur_type == 24)):
                        self.text_stack.append(self.cur_row)
                        self.description += self.cur_row[2].strip() + '; '
                        cur_ind += 1

                        if self.relay_array[cur_ind][0] != cur_to:
                            break
                        self.cur_row = self.relay_array[cur_ind]
                        cur_type = self.cur_row[3]
                    self.description = self.description.strip()

                    result = parse_batter_result(self.cur_text.strip())
                    self.pa_result = result[0]
                    self.pa_result_detail = result[1]

                    # 타격 결과가 '타격'인 경우에는 새 row를 추가, 여기에 description등을 넣음
                    # 자동 고의4구 경우도 새 row를 추가
                    # 아닌 경우에는 직전에 print_rows에 추가된 row(tail)에 description, pa_result 등을 추가
                    if (self.pitch_result != '타격') & (self.pa_result != '자동 고의4구'):
                        self.print_rows[-1]['description'] = self.description
                        self.print_rows[-1]['pa_result'] = self.pa_result
                        self.print_rows[-1]['pa_result_detail'] = self.pa_result_detail
                    else:
                        save_row = self.convert_row_to_save_format(self.last_pitch,
                                                                   [self.description, self.pa_result, self.pa_result_detail])
                        self.print_rows.append(save_row)

                    self.ind = cur_ind
                    self.handle_runner_stack(self.text_stack, debug_mode)
                    if self.outs > 3:
                        if debug_mode == True:
                            self.log_text.append('Outs > 3')
                        assert False
                elif (cur_type == 14) or (cur_type == 24):
                    # 주자(비득점/득점)
                    # 주자 처리 텍스트 row를 쭉 text_stack에 쌓는다
                    # 쌓은 row를 한번에 처리(handle_runner_stack)
                    self.text_stack = []
                    cur_ind = self.ind
                    self.cur_row = self.relay_array[cur_ind]
                    self.description = ''
                    while (cur_type == 14) or (cur_type == 24):
                        self.text_stack.append(self.cur_row)
                        self.description += self.cur_row[2].strip() + '; '
                        cur_ind = cur_ind + 1
                        if self.relay_array[cur_ind][0] != cur_to:
                            break
                        self.cur_row = self.relay_array[cur_ind]
                        cur_type = self.cur_row[3]
                    self.description = self.description.strip()

                    save_row = self.convert_row_to_save_format(None,
                                                               [self.description, None, None])
                    self.print_rows.append(save_row)
                    self.ind = cur_ind

                    self.handle_runner_stack(self.text_stack, debug_mode)
                    if self.outs > 3:
                        if debug_mode == True:
                            self.log_text.append('Outs > 3')
                        assert False
                elif cur_type == 0:
                    # 이닝 시작
                    self.ind = self.ind + 1
                    if (len(self.print_rows) > 0) & (self.outs < 3):
                        if debug_mode == True:
                            self.log_text.append('outs < 3')
                        assert False
                    if self.top_bot == 1:
                        self.inn += 1
                    self.top_bot = (1 - self.top_bot)
                    self.pitcher_name, self.pitcher_code, self.throws = self.fields[self.top_bot].get('투수').values()

                    self.runner_bases = []
                    self.balls, self.outs, self.strikes = 0, 0, 0
                    self.DH_exist_after = self.DH_exist[:]
                    self.last_pitch = None

                    self.pitch_number = 0
                elif cur_type == 2:
                    # 교체/변경
                    # 교체/변경 텍스트 row를 쭉 text_stack에 쌓는다
                    # 쌓은 row를 한번에 처리(handle_change_stack)
                    self.text_stack = []
                    cur_ind = self.ind
                    self.cur_row = self.relay_array[cur_ind]
                    self.description = ''

                    while self.cur_row[3] == 2:
                        self.text_stack.append(self.cur_row[2])
                        self.description += self.cur_row[2].strip() + '; '
                        cur_ind = cur_ind + 1
                        if self.relay_array[cur_ind][0] != cur_to:
                            break
                        self.cur_row = self.relay_array[cur_ind]
                    self.description = self.description.strip()
                    self.ind = cur_ind

                    save_row = self.convert_row_to_save_format(None,
                                                               [self.description, None, None])
                    self.print_rows.append(save_row)

                    self.handle_change(self.text_stack, debug_mode)
                elif cur_type == 7:
                    self.ind = self.ind + 1
                else:
                    self.ind = self.ind + 1
            return True
        except:
            prlen = len(self.print_rows)-1
            while prlen >= 0:
                if self.print_rows[prlen]['pa_number'] == self.pa_number:
                    self.print_rows = self.print_rows[:-1]
                    prlen -= 1
                else:
                    break
            if debug_mode == True:
                self.log_text.append("-"*60)
                self.log_text.append(f"=== gameID : {self.game_id}")
                self.log_text.append("-"*60)

                tb = '말' if self.top_bot == 1 else '초'

                self.log_text.append(f'  GAME STATUS : {self.inn}회{tb} '+
                                     f'{self.outs}사 {self.strikes}S {self.balls}B '
                                     f'{self.cur_order}번타자 {self.batter_name}타석 '+
                                     f'{self.pitch_number}구')

                lines = traceback.format_exc().strip().split('\n')
                rl = [lines[-1]]
                lines = lines[1:-1]
                lines.reverse()
                for i in range(0, len(lines), 2):
                    if i+1 < len(lines):
                        rl.append(f'\t{lines[i].strip()} at {lines[i+1].strip()}')
                    else:
                        rl.append(f'\t{lines[i].strip()}')
                self.log_text += rl
                if self.log_file is not None:
                    if not self.log_file.closed:
                        for row in self.log_text:
                            self.log_file.write(row + '\n')
            return False

    def print_current_status(self):
        tb = '말' if self.top_bot == 1 else '초'

        print(f'  GAME STATUS : {self.inn}회{tb} '+
              f'{self.outs}사 {self.strikes}S {self.balls}B '
              f'{self.cur_order}번타자 {self.batter_name}타석 '+
              f'{self.pitch_number}구')

    def save_game(self, path=None):
        if len(self.print_rows) > 0:
            row_df = pd.DataFrame(self.print_rows)
            try:
                row_df['speed'] = row_df.speed.convert_dtypes()
                row_df['pitch_number'] = row_df.pitch_number.convert_dtypes()
                row_df['pa_number'] = row_df.pa_number.convert_dtypes()
            except:
                pass

            row_df.loc[:, 'px':'az'] = row_df.round({'px': 3, 'pz': 3, 'pfx_x': 3, 'pfx_z': 3,
                                                     'pfx_x_raw': 3, 'pfx_z_raw': 3, 'x0': 3, 'z0': 3,
                                                     'sz_top': 3, 'sz_bot': 3, 'y0': 3,
                                                     'vx0': 3, 'vy0': 3, 'vz0': 3,
                                                     'ax': 3, 'ay': 3, 'az': 3}).loc[:, 'px':'az']
            enc = 'cp949' if sys.platform == 'win32' else 'utf-8'

            if path is None:
                path = pathlib.Path('.')
            save_path = str(path / f'{self.game_id}.csv')

            row_df.to_csv(save_path,
                        encoding=enc,
                        index=False)
