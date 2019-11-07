# pfx_parse.py
#
# parse JSON, make game structure, convert to CSV.
# JSON 데이터를 읽어와 게임 상황을 구현하고
# gameday line 하나에 맞는 text row를 생성, CSV 파일에 저장한다.

import os
import json
from collections import OrderedDict
import regex
import csv

# custom library
from utils import print_progress

position = {
    '지명타자': 0,
    '투수': 1,
    '포수': 2,
    '1루수': 3,
    '2루수': 4,
    '3루수': 5,
    '유격수': 6,
    '좌익수': 7,
    '중견수': 8,
    '우익수': 9,
    '대타': 10,
    '대주자': 11
}

field_home = {
    0: 'home_dh',
    1: 'home_p',
    2: 'home_c',
    3: 'home_1b',
    4: 'home_2b',
    5: 'home_3b',
    6: 'home_ss',
    7: 'home_lf',
    8: 'home_cf',
    9: 'home_rf'
}

field_away = {
    0: 'away_dh',
    1: 'away_p',
    2: 'away_c',
    3: 'away_1b',
    4: 'away_2b',
    5: 'away_3b',
    6: 'away_ss',
    7: 'away_lf',
    8: 'away_cf',
    9: 'away_rf'
}


class BallGame:
    # game status dict
    game_status = {
        'game_date': '00000000',

        # batter & pitcher
        'pitcher': None,
        'batter': None,
        'pitcher_ID': 0,
        'batter_ID': 0,

        # bats/throws; 0 for Left, 1 for Right
        'stand': 0,
        'throws': 0,

        # ball count & inning & score
        'inning': 1,
        # 0 for top, 1 for bot
        'inning_top_bot': 0,
        'balls': 0,
        'strikes': 0,
        'outs': 0,
        'score_home': 0,
        'score_away': 0,

        # pitch & pa result
        # 결과 나왔을 때만 기록
        'pa_result': None,
        'pitch_result': None,

        # pitch & pa num
        'pa_number': 0,
        'pitch_number': 0,

        # base
        'on_1b': None,
        'on_2b': None,
        'on_3b': None,

        # pfx data
        'pitch_type': None,
        'speed': None,
        'px': None,
        'pz': None,
        'pfx_x': None,
        'pfx_z': None,
        'pfx_x_raw': None,
        'pfx_z_raw': None,
        'x0': None,
        'z0': None,
        'sz_top': None,
        'sz_bot': None,

        # home & away
        'home': None,
        'away': None,
        'stadium': None,
        'referee': None,

        # home field
        'home_p': None,
        'home_c': None,
        'home_1b': None,
        'home_2b': None,
        'home_3b': None,
        'home_ss': None,
        'home_lf': None,
        'home_cf': None,
        'home_rf': None,
        'home_dh': None,

        # away field
        'away_p': None,
        'away_c': None,
        'away_1b': None,
        'away_2b': None,
        'away_3b': None,
        'away_ss': None,
        'away_lf': None,
        'away_cf': None,
        'away_rf': None,
        'away_dh': None,

        # home lineup
        'home_lineup': [
            # dummy 1~9
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0}
        ],

        # away lineup
        'away_lineup': [
            # dummy 1~9
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0}
        ],

        # raw data 추가
        'y0': None,
        'vx0': None,
        'vy0': None,
        'vz0': None,
        'ax': None,
        'ay': None,
        'az': None,
        'pitchId': None
    }

    # True 일 때 다음 타석/투구 전환->스코어, 아웃카운트, 이닝 초/말 등 변경.
    made_runs = False
    runs_how_many = 0
    made_outs = False
    outs_how_many = 0
    made_in_play = False
    runner_change = False
    change_1b = False
    change_2b = False
    change_3b = False
    next_1b = None
    next_2b = None
    next_3b = None
    ball_and_not_hbp = False
    set_hitter_to_base = False
    text_row = []
    prev_pid = None
    made_errors = False

    def reset_pfx(self):
        self.game_status['pitch_type'] = None
        self.game_status['speed'] = None
        self.game_status['px'] = None
        self.game_status['pz'] = None
        self.game_status['pfx_x'] = None
        self.game_status['pfx_z'] = None
        self.game_status['pfx_x_raw'] = None
        self.game_status['pfx_z_raw'] = None
        self.game_status['x0'] = None
        self.game_status['z0'] = None
        self.game_status['sz_top'] = None
        self.game_status['sz_bot'] = None
        self.game_status['y0'] = None
        self.game_status['vx0'] = None
        self.game_status['vy0'] = None
        self.game_status['vz0'] = None
        self.game_status['ax'] = None
        self.game_status['ay'] = None
        self.game_status['az'] = None
        self.game_status['pitchId'] = None

    def __init__(self, game_date=None):
        if game_date is not None:
            self.game_status['game_date'] = game_date

        self.game_status['pitcher'] = None
        self.game_status['batter'] = None
        self.game_status['pitcher_ID'] = 0
        self.game_status['batter_ID'] = 0

        self.game_status['stand'] = 0
        self.game_status['throws'] = 0

        self.game_status['inning'] = 0
        self.game_status['inning_top_bot'] = 1
        self.game_status['balls'] = 0
        self.game_status['strikes'] = 0
        self.game_status['outs'] = 0
        self.game_status['score_home'] = 0
        self.game_status['score_away'] = 0

        self.game_status['pa_result'] = None
        self.game_status['pitch_result'] = None

        self.game_status['pa_number'] = 0
        self.game_status['pitch_number'] = 0

        self.game_status['on_1b'] = None
        self.game_status['on_2b'] = None
        self.game_status['on_3b'] = None

        self.reset_pfx()

        self.game_status['home'] = None
        self.game_status['away'] = None
        self.game_status['stadium'] = None
        self.game_status['referee'] = None

        for i in range(10):
            self.game_status[field_away[i]] = None
            self.game_status[field_home[i]] = None

        self.game_status['home_lineup'] = [
            # dummy 1~9
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0}
        ]
        self.game_status['away_lineup'] = [
            # dummy 1~9
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0},
            {'pos': '', 'name': '', 'seqno': 0}
        ]

        self.text_row = []
        self.made_runs = False
        self.runs_how_many = 0
        self.made_outs = False
        self.outs_how_many = 0
        self.made_in_play = False
        self.runner_change = False
        self.change_1b = False
        self.change_2b = False
        self.change_3b = False
        self.next_1b = None
        self.next_2b = None
        self.next_3b = None
        self.ball_and_not_hbp = False
        self.set_hitter_to_base = False
        self.prev_pid = None
        self.made_errors = False
        self.rain_delay = False

    def print_row(self):
        row = [str(self.game_status['pitch_type']), str(self.game_status['pitcher']), str(self.game_status['batter']),
               str(self.game_status['pitcher_ID']), str(self.game_status['batter_ID']), str(self.game_status['speed']),
               str(self.game_status['pitch_result']), str(self.game_status['pa_result']),
               str(self.game_status['balls']), str(self.game_status['strikes']), str(self.game_status['outs']),
               str(self.game_status['inning'])]
        if self.game_status['inning_top_bot'] == 0:
            row.append('초')
        else:
            row.append('말')
        row.append(str(self.game_status['score_away']))
        row.append(str(self.game_status['score_home']))
        row.append(str(self.game_status['stands']))
        row.append(str(self.game_status['throws']))
        row.append(str(self.game_status['on_1b']))
        row.append(str(self.game_status['on_2b']))
        row.append(str(self.game_status['on_3b']))

        row.append(str(self.game_status['px']))
        row.append(str(self.game_status['pz']))
        row.append(str(self.game_status['pfx_x']))
        row.append(str(self.game_status['pfx_z']))
        row.append(str(self.game_status['pfx_x_raw']))
        row.append(str(self.game_status['pfx_z_raw']))
        row.append(str(self.game_status['x0']))
        row.append(str(self.game_status['z0']))
        row.append(str(self.game_status['sz_top']))
        row.append(str(self.game_status['sz_bot']))

        if self.game_status['inning_top_bot'] == 0:
            row.append(str(self.game_status['home_p']))
            row.append(str(self.game_status['home_c']))
            row.append(str(self.game_status['home_1b']))
            row.append(str(self.game_status['home_2b']))
            row.append(str(self.game_status['home_3b']))
            row.append(str(self.game_status['home_ss']))
            row.append(str(self.game_status['home_lf']))
            row.append(str(self.game_status['home_cf']))
            row.append(str(self.game_status['home_rf']))
        else:
            row.append(str(self.game_status['away_p']))
            row.append(str(self.game_status['away_c']))
            row.append(str(self.game_status['away_1b']))
            row.append(str(self.game_status['away_2b']))
            row.append(str(self.game_status['away_3b']))
            row.append(str(self.game_status['away_ss']))
            row.append(str(self.game_status['away_lf']))
            row.append(str(self.game_status['away_cf']))
            row.append(str(self.game_status['away_rf']))

        row.append(str(self.game_status['game_date']))
        row.append(str(self.game_status['home']))
        row.append(str(self.game_status['away']))
        row.append(str(self.game_status['stadium']))
        row.append(str(self.game_status['referee']))

        row.append(str(self.game_status['pa_number']))
        row.append(str(self.game_status['pitch_number']))

        # raw data 추가
        row.append(str(self.game_status['y0']))
        row.append(str(self.game_status['vx0']))
        row.append(str(self.game_status['vy0']))
        row.append(str(self.game_status['vz0']))
        row.append(str(self.game_status['ax']))
        row.append(str(self.game_status['ay']))
        row.append(str(self.game_status['az']))
        row.append(str(self.game_status['pitchId']))
        self.text_row.append(row)

    def print_row_debug(self):
        # for debug
        print('\n')
        row = str(self.game_status['pitch_result']) + ',PIT'
        row += str(self.game_status['pitcher']) + ',BAT'
        row += str(self.game_status['batter']) + ','
        row += str(self.game_status['balls']) + 'B/'
        row += str(self.game_status['strikes']) + 'S/'
        row += str(self.game_status['outs']) + 'O,'
        row += str(self.game_status['inning'])
        if self.game_status['inning_top_bot'] == 0:
            row += '초,'
        else:
            row += '말,'
        row += str(self.game_status['score_away']) + ':'
        row += str(self.game_status['score_home']) + ', '
        row += str(self.game_status['on_1b']) + '/'
        row += str(self.game_status['on_2b']) + '/'
        row += str(self.game_status['on_3b']) + ',PA_RES:'
        row += str(self.game_status['pa_result']) + '\n'

        if self.game_status['inning_top_bot'] == 0:
            row += str(self.game_status['home_p']) + '/'
            row += str(self.game_status['home_c']) + '/'
            row += str(self.game_status['home_1b']) + '/'
            row += str(self.game_status['home_2b']) + '/'
            row += str(self.game_status['home_3b']) + '/'
            row += str(self.game_status['home_ss']) + '/'
            row += str(self.game_status['home_lf']) + '/'
            row += str(self.game_status['home_cf']) + '/'
            row += str(self.game_status['home_rf'])
        else:
            row += str(self.game_status['away_p']) + '/'
            row += str(self.game_status['away_c']) + '/'
            row += str(self.game_status['away_1b']) + '/'
            row += str(self.game_status['away_2b']) + '/'
            row += str(self.game_status['away_3b']) + '/'
            row += str(self.game_status['away_ss']) + '/'
            row += str(self.game_status['away_lf']) + '/'
            row += str(self.game_status['away_cf']) + '/'
            row += str(self.game_status['away_rf'])
        print(row)

    def set_home_away(self, home, away):
        self.game_status['home'] = home
        self.game_status['away'] = away

    def set_referee(self, referee):
        self.game_status['referee'] = referee

    def set_stadium(self, stadium):
        self.game_status['stadium'] = stadium

    def set_lineup(self, js):
        away_lineup = js['awayTeamLineUp']
        home_lineup = js['homeTeamLineUp']
        for away_bat in away_lineup['batter']:
            order = away_bat['batOrder'] - 1
            if self.game_status['away_lineup'][order]['seqno'] == 0:
                self.game_status['away_lineup'][order]['pos'] = away_bat['posName']
                self.game_status['away_lineup'][order]['name'] = away_bat['name']
                self.game_status['away_lineup'][order]['seqno'] = away_bat['seqno']
                pos_num = position[away_bat['posName']]
                if pos_num < 10:
                    pos_name = field_away[pos_num]
                    if (self.game_status[pos_name] is None) & (away_bat['seqno'] == 1):
                        self.game_status[pos_name] = away_bat['name']
            else:
                cur_seqno = self.game_status['away_lineup'][order]['seqno']
                if away_bat['seqno'] < cur_seqno:
                    self.game_status['away_lineup'][order]['pos'] = away_bat['posName']
                    self.game_status['away_lineup'][order]['name'] = away_bat['name']
                    self.game_status['away_lineup'][order]['seqno'] = away_bat['seqno']
                    pos_num = position[away_bat['posName']]
                    if pos_num < 10:
                        pos_name = field_away[pos_num]
                        if (self.game_status[pos_name] is None) & (away_bat['seqno'] == 1):
                            self.game_status[pos_name] = away_bat['name']

        for home_bat in home_lineup['batter']:
            order = home_bat['batOrder'] - 1
            if self.game_status['home_lineup'][order]['seqno'] == 0:
                self.game_status['home_lineup'][order]['pos'] = home_bat['posName']
                self.game_status['home_lineup'][order]['name'] = home_bat['name']
                self.game_status['home_lineup'][order]['seqno'] = home_bat['seqno']
                pos_num = position[home_bat['posName']]
                if pos_num < 10:
                    pos_name = field_home[pos_num]
                    if (self.game_status[pos_name] is None) & (home_bat['seqno'] == 1):
                        self.game_status[pos_name] = home_bat['name']
            else:
                cur_seqno = self.game_status['home_lineup'][order]['seqno']
                if home_bat['seqno'] < cur_seqno:
                    self.game_status['home_lineup'][order]['pos'] = home_bat['posName']
                    self.game_status['home_lineup'][order]['name'] = home_bat['name']
                    self.game_status['home_lineup'][order]['seqno'] = home_bat['seqno']
                    pos_num = position[home_bat['posName']]
                    if pos_num < 10:
                        pos_name = field_home[pos_num]
                        if (self.game_status[pos_name] is None) & (home_bat['seqno'] == 1):
                            self.game_status[pos_name] = home_bat['name']

        for away_pit in away_lineup['pitcher']:
            if not (away_pit['seqno'] == 1):
                continue
            else:
                self.game_status['away_p'] = away_pit['name']

        for home_pit in home_lineup['pitcher']:
            if not (home_pit['seqno'] == 1):
                continue
            else:
                self.game_status['home_p'] = home_pit['name']

    # 경기 종료 조건 체크
    # 종료 조건
    # 1. 이닝 >= 9
    # 2. 아웃 = 3
    # 3-1. 초 & 점수 홈>어웨이
    # 3-2. 말 & 점수 어웨이>홈

    def check_game_over(self):
        # 종료 조건
        if self.game_status['inning'] >= 9:
            if self.game_status['outs'] == 3:
                if self.game_status['inning_top_bot'] == 0:
                    if self.game_status['score_home'] > self.game_status['score_away']:
                        return True
                else:
                    if self.game_status['score_home'] < self.game_status['score_away']:
                        return True
        return False

    # 이닝 변경

    def go_to_next_inning(self):
        if self.game_status['strikes'] == 3:
            self.game_status['strikes'] = 2
            self.print_row()
            # self.print_row_debug()
        if self.made_in_play is True:
            self.print_row()
            # self.print_row_debug()

        if self.made_outs is True:
            if self.game_status['outs'] + self.outs_how_many != 3:
                rc = 'out != 3\n'
                rc += '{}회 종료 시'.format(
                    self.game_status['inning']
                )
                return rc

        if self.made_runs is True:
            if self.game_status['inning_top_bot'] == 0:
                self.game_status['score_away'] += self.runs_how_many
            else:
                self.game_status['score_home'] += self.runs_how_many

        if self.game_status['inning_top_bot'] == 1:
            self.game_status['inning'] += 1

        self.game_status['inning_top_bot'] = (1 - self.game_status['inning_top_bot'])

        if self.game_status['inning_top_bot'] == 0:
            self.game_status['pitcher'] = self.game_status['home_p']
        else:
            self.game_status['pitcher'] = self.game_status['away_p']

        self.game_status['pa_result'] = None
        self.game_status['pitch_result'] = None

        self.game_status['outs'] = 0
        self.game_status['on_1b'] = None
        self.game_status['on_2b'] = None
        self.game_status['on_3b'] = None
        self.game_status['pitch_number'] = 0

        self.made_runs = False
        self.runs_how_many = 0
        self.made_outs = False
        self.outs_how_many = 0
        self.made_in_play = False
        self.runner_change = False
        self.change_1b = False
        self.change_2b = False
        self.change_3b = False
        self.next_1b = None
        self.next_2b = None
        self.next_3b = None
        self.ball_and_not_hbp = False
        self.set_hitter_to_base = False

        return True

    # 타석 리셋
    def go_to_next_pa(self):
        # 이전 타석의 결과물을 print한다.
        # 타석 관련 데이터를 reset한다. 볼, 스트, 아웃 등
        # 타석 종료된 경우: (1) inplay (2) 3S (3) 4B(IBB 포함)
        #                   (4)HBP (5) 자동 고의4구
        #     이럴 때는 이전 타석 결과를 print한다.
        # 그 밖의 경우: 타석 도중 교체
        #     타석 데이터 reset 없이 true만 반환한다.
        
        # BUGBUG : 자동 고의4구 직전 투구 기록되지 않음
        # 20180720WONC02018 스크럭스 8회말 타석
        # print_row()를 실행하지 않음(ibb()로 game_status에 pa_result만 기록함)
        #   자동 고의4구를 새로운 결과로 추가(기존: 고의4구만 존재)
        #   타석 종료된 경우 (5) 자동 고의4구 추가
        if self.made_errors is True:
            # BUGBUG : 실책 이후 자동 고의4구 처리시, 고의4구 텍스트 출력 처리 안됨
            # - 시작일 : 20191107
            # - 종료일 : 20191107
            #
            # CASE :
            # 20190602OBKT0 8회초 김재환 타석, 자동고의4구
            # 실책 이후 나오는 고의4구는 출력이 안되고 있었음
            #
            # CAUSE :
            # 실책때 self.made_errors를 False로 바꾸지 않아서 에러.
            self.made_errors = False
            # BUGBUG : 실책 출루인데 포스 아웃으로 기록되는 사례
            # - 시작일 : 20180812
            # - 종료일 : 20180812
            #
            # CASE :
            # 20180812HTSK0 1회초 KIA 공격, 3번 최형우 타석
            # 최형우 2루수 땅볼로 출루
            # parse_pa_result에서 force_out 처리되지만
            # 이후 텍스트를 보면 실책으로 무사 출루함.
            #
            # 실제 텍스트(순서대로)
            # 최형우 : 2루수 앞 땅볼로 출루
            # 1루주자 최형우 : 실책으로 2루까지 진루
            # 1루주자 이명기 : 2루수 실책으로 2루까지 진루(2루수 실책->유격수)
            # 2루주자 이명기 : 주자의 재치로 3루까지 진루
            # 2루주자 버나디나 : 3루까지 진루
            # 3루주자 버나디나 : 실책으로 홈인
            #
            # OBJECT : 아웃 없는 경우 실책 출루였던 것으로 판정.
            #
            # 아래 추가 BUG 확인으로 로직 수정.
            if (self.made_outs is False) and (self.game_status['pa_result'] != '실책'):
                # BUGBUG : 실책 아닌데 실책으로 기록되는 경우
                # - 시작일 : 20190706
                # - 종료일 : 20190706
                #
                # CASE :
                # 크게 2가지가 있다.
                # (1) 인플레이/볼넷/삼진 이후 실책 발생
                #  -> 타석 결과가 실책으로 수정
                # (2) 타석 도중 상황 발생, 그 과정에서 실책으로 이닝 종료
                #  -> 타석이 끝나지 않았음에도 타석 결과 기록(실책으로)
                #
                # (1)의 경우 : 케이스가 아주 많음.
                # 20190410SKHH 3회초 SK 공격, 9번 김성현 타석
                # 내야안타 기록, 이후 실책으로 추가 진루
                # -> 내야안타가 아닌 실책으로 기록됨.
                #
                # (2)의 경우: 케이스가 많지만 2가지가 있음.
                # 20190410NCHT 2회말 KIA 공격, 9번 박찬호 타석
                # 포수가 2루 견제, 견제구 빠짐, 2루주자 3루행
                # 뒤늦게 출발한 1루주자 2루에서 태그아웃
                # -> 타석 결과가 없고 '실책'이 아님에도 '실책' 기록
                #
                # 20190410OBLT 1회초 두산 공격, 5번 페르난데스 타석
                # 볼넷이 나왔는데 폭투성 투구
                # 포수가 공 빠트리고 3루주자 홈인, 투수 실책 기록
                # -> 사실상 포일이고 결과는 볼넷이지만 실책으로 정정
                #
                # OBJECT : 다양한 사례 고려해 로직 수정.
                #
                # 20180812 BUG까지 고려해야 한다.
                #
                # 이 분기로 오는 건 made_errors가 True로 기록됐기 때문인데
                # made_errors가 True가 되는 경로는 세 가지다.
                # (1) 타자주자가 각종 타구 실책으로 출루.(roe)
                # (2) 타자주자가 희생번트 실책으로 출루.(sac_hit_error)
                # (3) 주자 처리 중 실책 텍스트 발견.(parse_runner)
                #
                # 여기서 실제 실책은 1, 2고
                # 이 경우 game_status의 pa_result가 이미
                # '실책' 혹은 '희생번트 실책'이 되어 있다.
                #
                # 문제는 3인데,
                # '실책으로 출루'는 2018년쯤부터 2개 라인으로 나눠 처리한다.
                # '땅볼로 출루' -> 다음 텍스트에서 '실책으로 출루'
                # 이런 순서로 말이다.
                # 이 경우 parse_pa_result에서 force_out으로 처리되고
                # pa_result가 '포스 아웃'으로 되어 있다.
                # -> 이것은 실책 출루로 처리해야 한다.
                #
                # 그렇다면 나머지 사례는?
                # 정상적인 안타, 볼넷, 삼진, 희생플라이 등인데
                # 이후 주루 상황에서 실책이 발생한 것이다.
                # -> 이것은 실책으로 처리하면 안된다.
                if (self.game_status['pa_result'] == '포스 아웃'):
                    self.game_status['pa_result'] = '실책'
        if self.made_in_play is True:
            self.print_row()
            # self.print_row_debug()
        elif self.game_status['strikes'] == 3:
            self.game_status['strikes'] = 2
            self.print_row()
            # self.print_row_debug()
        elif self.game_status['balls'] == 4:
            # 자동 아닌 고의4구도 여기에 포함
            self.game_status['balls'] = 3
            self.print_row()
            # self.print_row_debug()
        elif self.game_status['pa_result'] is not None:
            if self.game_status['pa_result'].find('몸에') > -1:
                self.game_status['balls'] -= 1
                self.print_row()
                # self.print_row_debug()
            elif self.game_status['pa_result'].find('자동') > -1:
                # 타석 종료된 경우 (5) 자동 고의4구 추가
                self.print_row()
                # self.print_row_debug()
        elif self.rain_delay is True:
            if self.ball_and_not_hbp is True:
                # 연속으로 볼이 들어왔을 때
                # 사구가 아니라면 앞선 볼 상황을 출력
                self.game_status['balls'] -= 1
                self.print_row()
            # self.print_row_debug()
        
        self.game_status['balls'] = 0
        self.game_status['strikes'] = 0
        if self.made_runs is True:
            if self.game_status['inning_top_bot'] == 0:
                self.game_status['score_away'] += self.runs_how_many
            else:
                self.game_status['score_home'] += self.runs_how_many

        if self.made_outs is True:
            self.game_status['outs'] += self.outs_how_many
            if self.game_status['outs'] > 3:
                rc = 'out > 3\n'
                rc += '{}회 {}:{} 타석'.format(
                    self.game_status['inning'],
                    self.game_status['pitcher'],
                    self.game_status['batter']
                )
                return rc

        if self.runner_change is True:
            if self.change_3b is True:
                self.game_status['on_3b'] = self.next_3b

            if self.change_2b is True:
                self.game_status['on_2b'] = self.next_2b

            if self.change_1b is True:
                self.game_status['on_1b'] = self.next_1b

        self.reset_pfx()
        self.game_status['pa_result'] = None
        self.game_status['pitch_result'] = None
        self.game_status['pitch_number'] = 0
        self.game_status['pa_number'] += 1

        self.made_runs = False
        self.runs_how_many = 0
        self.made_outs = False
        self.outs_how_many = 0
        self.made_in_play = False
        self.runner_change = False
        self.next_1b = None
        self.next_2b = None
        self.next_3b = None
        self.change_1b = False
        self.change_2b = False
        self.change_3b = False
        self.ball_and_not_hbp = False
        self.set_hitter_to_base = False

        return True

    # 득점
    def score(self, runs):
        self.made_runs = True
        self.runs_how_many += runs

    # 아웃
    def out(self, outs):
        self.made_outs = True
        self.outs_how_many += outs

    # --------------------------------
    # pitch result
    # --------------------------------

    def get_ball(self):
        # 몸에맞는공의 경우 아무 카운트에서나 나온다
        # 다음 parse_pitch에서 결과를 확인하고 프린트
        self.game_status['pitch_result'] = '볼'
        cur_balls = self.game_status['balls']
        if cur_balls == 3:
            self.game_status['pa_result'] = '볼넷'
        else:
            self.ball_and_not_hbp = True
            # 볼넷이 아니고, 연속 볼이 들어올 때만.
        self.game_status['balls'] += 1

    def get_strike(self):
        self.game_status['pitch_result'] = '스트라이크'
        cur_strikes = self.game_status['strikes']
        if cur_strikes == 2:
            self.game_status['pa_result'] = '삼진'
        else:
            self.print_row()
            # self.print_row_debug()
        self.game_status['strikes'] += 1

    def get_swing_miss(self):
        self.game_status['pitch_result'] = '헛스윙'
        cur_strikes = self.game_status['strikes']
        if cur_strikes == 2:
            self.game_status['pa_result'] = '삼진'
        else:
            self.print_row()
            # self.print_row_debug()
        self.game_status['strikes'] += 1

    def get_bunt_swing_miss(self):
        self.game_status['pitch_result'] = '번트헛스윙'
        cur_strikes = self.game_status['strikes']
        if cur_strikes == 2:
            self.game_status['pa_result'] = '삼진'
        else:
            self.print_row()
            # self.print_row_debug()
        self.game_status['strikes'] += 1

    def get_foul(self):
        self.game_status['pitch_result'] = '파울'
        self.print_row()
        # self.print_row_debug()
        if self.game_status['strikes'] < 2:
            self.game_status['strikes'] += 1

    def get_bunt_foul(self):
        self.game_status['pitch_result'] = '번트파울'
        self.print_row()
        # self.print_row_debug()
        if self.game_status['strikes'] < 2:
            self.game_status['strikes'] += 1

    def get_in_play(self):
        self.game_status['pitch_result'] = '타격'
        self.made_in_play = True
        # print row -> @go_to_next_pa

    # --------------------------------
    # pa result
    # --------------------------------

    def single(self):
        self.game_status['pa_result'] = '1루타'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    def double(self):
        self.game_status['pa_result'] = '2루타'
        self.runner_change = True
        self.change_2b = True
        self.next_2b = self.game_status['batter']
        self.set_hitter_to_base = True

    def triple(self):
        self.game_status['pa_result'] = '3루타'
        self.runner_change = True
        self.change_3b = True
        self.next_3b = self.game_status['batter']
        self.set_hitter_to_base = True

    def homerun(self):
        self.game_status['pa_result'] = '홈런'
        self.runner_change = True
        self.change_1b = True
        self.change_2b = True
        self.change_3b = True
        self.next_1b = None
        self.next_2b = None
        self.next_3b = None
        # runner score -> runner 처리할 때 같이 처리.
        self.score(1)

    def infield_hit(self):
        self.game_status['pa_result'] = '내야안타'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    # 번트 안타
    def bunt_hit(self):
        self.game_status['pa_result'] = '번트 안타'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    # 기타 안타
    def other_hit(self):
        self.game_status['pa_result'] = '안타'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    def strike_out(self):
        self.game_status['pa_result'] = '삼진'
        self.out(1)

    def four_ball(self):
        self.game_status['pa_result'] = '볼넷'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    def ibb(self):
        self.game_status['pa_result'] = '고의4구'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True
        # ball_and_not_hbp가 True일 때:
        #       '볼넷이 아니면서' & '연속 볼로 인한 진루'일 때만
        # 여기에 '(자동) 고의4구가 아닐때'도 추가.
        # True면 False로 바꿔준다.
        # (고의4구로 타석 바뀌면 리셋되니까, False로 한다)
        self.ball_and_not_hbp = False

    # BUGBUG: 자동 고의4구 직전 투구 기록되지 않음
    # 20180720WONC02018 스크럭스 8회말 타석
    # print_row()를 실행하지 않음(ibb()로 game_status에 pa_result만 기록함)
    # 타석 종료된 경우 (5) 자동 고의4구 추가
    def auto_ibb(self):
        self.game_status['pa_result'] = '자동 고의4구'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True
        self.ball_and_not_hbp = False
        # ball_and_not_hbp가 True일 때:
        #       '볼넷이 아니면서' & '연속 볼'일 때만
        # 여기에 '자동 고의4구가 아닐때'도 추가.
        # True면 False로 바꿔준다.
        # (고의4구로 타석 바뀌면 리셋되니까, False로 한다)

    def hbp(self):
        self.game_status['pa_result'] = '몸에 맞는 볼'
        self.runner_change = True
        self.change_1b = True
        self.ball_and_not_hbp = False
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    # 낫아웃 삼진
    def not_out(self):
        self.game_status['pa_result'] = '삼진'
        self.out(1)

    # 낫아웃 폭투
    def not_out_wp(self):
        self.game_status['pa_result'] = '낫아웃 폭투'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    # 낫아웃 포일
    def not_out_pb(self):
        self.game_status['pa_result'] = '낫아웃 포일'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    # 주자 포스 아웃, 타자 주자 출루
    # 땅볼로 출루
    def force_out(self):
        self.game_status['pa_result'] = '포스 아웃'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True
        # out -> runner 처리할 때 같이 처리.

    # 타자 주자 아웃
    # 땅볼 아웃, 플라이 아웃, 인필드 플라이, 파울 플라이
    # 라인드라이브 아웃, 번트 아웃
    # 현재 수비방해(송구방해)는 땅볼 아웃으로 기록 중
    def field_out(self):
        self.game_status['pa_result'] = '필드 아웃'
        self.out(1)

    def double_play(self):
        self.game_status['pa_result'] = '병살타'
        self.out(1)

    def sac_hit(self):
        self.game_status['pa_result'] = '희생번트'
        self.out(1)

    def sac_fly(self):
        self.game_status['pa_result'] = '희생플라이'
        self.out(1)

    def three_bunt(self):
        self.game_status['pa_result'] = '쓰리번트 삼진'
        self.out(1)

    def hit_by_ball_out(self):
        self.game_status['pa_result'] = '타구맞음 아웃'
        self.out(1)

    # 실책 출루
    # 플라이 실책, 번트 실책
    # 라인드라이브 실책, 병살실책, 실책
    def roe(self):
        self.game_status['pa_result'] = '실책'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True
        self.made_errors = True

    def fc(self):
        self.game_status['pa_result'] = '야수선택'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    # 희생번트 실책
    def sac_hit_error(self):
        self.game_status['pa_result'] = '희생번트 실책'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True
        self.made_errors = True

    # 희생번트 야수선택
    def sac_hit_fc(self):
        self.game_status['pa_result'] = '희생번트 야수선택'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    def interference(self):
        self.game_status['pa_result'] = '타격방해'
        self.runner_change = True
        self.change_1b = True
        self.next_1b = self.game_status['batter']
        self.set_hitter_to_base = True

    def triple_play(self):
        self.game_status['pa_result'] = '삼중살'
        self.out(1)

    # --------------------------------
    # runner result
    # --------------------------------

    # 도루, 폭투, 포일, 보크 포함
    # 진루 후 투구시 parse_pitch에서 next runner 업데이트
    def runner_advance(self, src_base, src_name, dst):
        if dst == 1:
            self.next_1b = src_name
            self.change_1b = True
        elif dst == 2:
            if src_base == 1:
                if self.change_1b is True:
                    # 이미 1루 주자 변경 있었음
                    if src_name == self.next_1b:
                        # 새 1루 주자가 진루한 경우
                        if self.set_hitter_to_base is True:
                            # 타자 주자 이름이 바로 나오는 경우
                            # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                            # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                            #    이름이 같으면 타인으로 본다.
                            #    타자주자라면 None 또는 다른 이름일 것이다.
                            if self.game_status['on_1b'] == src_name:
                                self.set_hitter_to_base = False
                            else:
                                self.next_1b = None
                        else:
                            # 1루를 비운다
                            self.next_1b = None
                else:
                    # 기존 1루 주자 진루.
                    self.next_1b = None
                    self.change_1b = True
            # BUGBUG : 실책시 주자 주루 처리 오류
            # 
            # - 시작일 : 20191107
            #  - 종료일 : 20191107
            # 
            # CASE :
            # 20190713OBLT0 9회말 배성근 타석, 주자 1-2루
            # 땅볼-실책으로
            # 타자주자->실책->2루로
            # 1루주자->3루로
            # 2루주자->3루->홈인
            # 여기서 2루주자가 3루 거쳐가는게 늦게 나오는데
            # 이 텍스트 파싱 과정에서 홈인한 2루주자가 3루에 있는 것처럼
            # (self.next_3b=3루주자) 처리되는 버그
            #
            # OBJECT :
            # 선행주자(2루->3루->홈)의 중계 문구 처리가 뒤따라올 떄
            # 앞에서 이동해온 후발주자(1루->3루)가
            # 정상적으로 베이스에 남아있도록 처리
            # 베이스 change된 기록이 있으면(change_Xb == True)
            # 그냥 넘어가고, 아닐 때만 next_Xb를 바꾼다
            if self.change_2b == False:
                self.next_2b = src_name
                self.change_2b = True
        else:
            if src_base == 1:
                if self.change_1b is True:
                    # 이미 1루주자 변경 있었음
                    if src_name == self.next_1b:
                        # 새 1루 주자가 진루한 경우
                        if self.set_hitter_to_base is True:
                            # 타자 주자 이름이 바로 나오는 경우
                            # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                            # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                            #    이름이 같으면 타인으로 본다.
                            #    타자주자라면 None 또는 다른 이름일 것이다.
                            if self.game_status['on_1b'] == src_name:
                                self.set_hitter_to_base = False
                            else:
                                self.next_1b = None
                        else:
                            # 1루를 비운다
                            self.next_1b = None
                    # else:
                    # 기존 1루 주자가 진루
                    # self.next_1b는 그대로 보존
                else:
                    # 기존 1루 주자 진루.
                    self.next_1b = None
                    self.change_1b = True
            elif src_base == 2:
                if self.change_2b is True:
                    # 이미 2루주자 변경 있었음
                    if src_name == self.next_2b:
                        # 새 2루 주자가 진루한 경우
                        if self.set_hitter_to_base is True:
                            # 타자 주자 이름이 바로 나오는 경우
                            # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                            # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                            #    이름이 같으면 타인으로 본다.
                            #    타자주자라면 None 또는 다른 이름일 것이다.
                            if self.game_status['on_2b'] == src_name:
                                self.set_hitter_to_base = False
                            else:
                                self.next_2b = None
                        else:
                            # 2루를 비운다
                            self.next_2b = None
                    # else:
                    # 기존 2루 주자가 진루
                    # self.next_2b는 그대로 보존
                else:
                    # 기존 2루 주자 진루.
                    self.next_2b = None
                    self.change_2b = True
            # BUGBUG : 실책시 주자 주루 처리 오류
            # 
            # - 시작일 : 20191107
            #  - 종료일 : 20191107
            # 
            # CASE :
            # 20190713OBLT0 9회말 배성근 타석, 주자 1-2루
            # 땅볼-실책으로
            # 타자주자->실책->2루로
            # 1루주자->3루로
            # 2루주자->3루->홈인
            # 여기서 2루주자가 3루 거쳐가는게 늦게 나오는데
            # 이 텍스트 파싱 과정에서 홈인한 2루주자가 3루에 있는 것처럼
            # (self.next_3b=3루주자) 처리되는 버그
            #
            # OBJECT :
            # 선행주자(2루->3루->홈)의 중계 문구 처리가 뒤따라올 떄
            # 앞에서 이동해온 후발주자(1루->3루)가
            # 정상적으로 베이스에 남아있도록 처리
            # 베이스 change된 기록이 있으면(change_Xb == True)
            # 그냥 넘어가고, 아닐 때만 next_Xb를 바꾼다
            if self.change_3b == False:
                self.next_3b = src_name
                self.change_3b = True

        self.runner_change = True

    def runner_home_in(self, src_base, src_name):
        self.score(1)
        if src_base == 1:
            if self.change_1b is True:
                # 이미 1루 주자 변경 있었음 -> 타자가 1루 진루
                if src_name == self.next_1b:
                    # 타자가 다시 진루, 홈인하는 경우
                    if self.set_hitter_to_base is True:
                        # 타자 주자 이름이 바로 나오는 경우
                        # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                        # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                        #    이름이 같으면 타인으로 본다.
                        #    타자주자라면 None 또는 다른 이름일 것이다.
                        if self.game_status['on_1b'] == src_name:
                            self.set_hitter_to_base = False
                        else:
                            self.next_1b = None
                    else:
                        # 기존(새로 들어온) 1루 주자를 홈인시킨다.
                        self.next_1b = None
                # else:
                # 기존 1루 주자가 홈인하는 경우(타자가 1루 진루)
                # self.next_1b는 그대로 보존
            else:
                # 기존 1루 주자가 홈인
                self.next_1b = None
                self.change_1b = True
        elif src_base == 2:
            if self.change_2b is True:
                # 이미 2루 주자 변경 있었음 -> 타자 혹은 1루 주자가 2루 진루
                if src_name == self.next_2b:
                    # 2루 들어온 새 주자가 진루, 홈인하는 경우
                    if self.set_hitter_to_base is True:
                        # 타자 주자 이름이 바로 나오는 경우
                        # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                        # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                        #    이름이 같으면 타인으로 본다.
                        #    타자주자라면 None 또는 다른 이름일 것이다.
                        if self.game_status['on_2b'] == src_name:
                            self.set_hitter_to_base = False
                        else:
                            self.next_2b = None
                    else:
                        # 기존(새로 들어온) 2루 주자를 홈인시킨다.
                        self.next_2b = None
                # else:
                # 기존 2루 주자가 홈인
                # self.next_2b는 그대로 보존
            else:
                # 기존 2루 주자가 홈인
                self.next_2b = None
                self.change_2b = True
        else:
            if self.change_3b is True:
                # 이미 2루 주자 변경 있었음 -> 타자 혹은 1/2루 주자가 3루 진루
                if src_name == self.next_3b:
                    # 3루 들어온 새 주자가 진루, 홈인하는 경우
                    if self.set_hitter_to_base is True:
                        # 타자 주자 이름이 바로 나오는 경우
                        # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                        # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                        #    이름이 같으면 타인으로 본다.
                        #    타자주자라면 None 또는 다른 이름일 것이다.
                        if self.game_status['on_3b'] == src_name:
                            self.set_hitter_to_base = False
                        else:
                            self.next_3b = None
                    else:
                        # 기존(새로 들어온) 3루 주자를 홈인시킨다.
                        self.next_3b = None
                # else:
                # 기존 3루 주자가 홈인
                # self.next_3b는 그대로 보존
            else:
                # 기존 3루 주자가 홈인
                self.next_3b = None
                self.change_3b = True
        self.runner_change = True

    # 도루실패, 견제사, 진루실패 포함
    def runner_out(self, src_base, src_name):
        self.out(1)
        if src_base == 1:
            if self.change_1b is True:
                # 1루 주자에 변경 있음
                if src_name == self.next_1b:
                    # 새로 들어온 1루 주자가 아웃
                    if self.set_hitter_to_base is True:
                        # 타자 주자 이름이 바로 나오는 경우
                        # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                        # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                        #    이름이 같으면 타인으로 본다.
                        #    타자주자라면 None 또는 다른 이름일 것이다.
                        if self.game_status['on_1b'] == src_name:
                            self.set_hitter_to_base = False
                        else:
                            self.next_1b = None
                    else:
                        # 기존(새로 들어온) 1루 주자가 아웃.
                        self.next_1b = None
                # else:
                # 기존 1루 주자가 아웃
                # self.next_1b는 그대로 보존; 새로 1루에 간 주자 이름이 들어감
            else:
                # 기존 1루 주자가 아웃
                self.next_1b = None
                self.change_1b = True
        elif src_base == 2:
            if self.change_2b is True:
                # 2루 주자에 변경 있음
                if src_name == self.next_2b:
                    # 새로 들어온 2루 주자가 아웃
                    if self.set_hitter_to_base is True:
                        # 타자 주자 이름이 바로 나오는 경우
                        # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                        # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                        #    이름이 같으면 타인으로 본다.
                        #    타자주자라면 None 또는 다른 이름일 것이다.
                        if self.game_status['on_2b'] == src_name:
                            self.set_hitter_to_base = False
                        else:
                            self.next_2b = None
                    else:
                        # 기존(새로 들어온) 2루 주자가 아웃.
                        self.next_2b = None
                # else:
                # 기존 2루 주자가 아웃
                # self.next_2b는 그대로 보존; 새로 2루에 간 주자 이름이 들어감
            else:
                # 기존 2루 주자가 아웃
                self.next_2b = None
                self.change_2b = True
        else:
            if self.change_3b is True:
                # 3루 주자에 변경 있음
                if src_name == self.next_3b:
                    # 새로 들어온 3루 주자가 아웃
                    if self.set_hitter_to_base is True:
                        # 타자 주자 이름이 바로 나오는 경우
                        # (타자 이병규 ~~~, 1루주자 이병규 ~~~)
                        # -> 현재 루상에 있는(game_status['on_1b'] 체크) 주자가
                        #    이름이 같으면 타인으로 본다.
                        #    타자주자라면 None 또는 다른 이름일 것이다.
                        if self.game_status['on_3b'] == src_name:
                            self.set_hitter_to_base = False
                        else:
                            self.next_3b = None
                    else:
                        # 기존(새로 들어온) 3루 주자가 아웃.
                        self.next_3b = None
                # else:
                # 기존 3루 주자가 아웃
                # self.next_3b는 그대로 보존; 새로 3루에 간 주자 이름이 들어감
            else:
                # 기존 3루 주자가 아웃
                self.next_3b = None
                self.change_3b = True
        self.runner_change = True


##########

pa_pattern = regex.compile('^\p{Hangul}+ : [\p{Hangul}|0-9|\ ]+')
pitch_pattern = regex.compile('^[0-9]+구 [0-9\ C\p{Hangul}]+')
ibb_pattern = regex.compile('^[0-9]+구 I')
auto_ibb_pattern = regex.compile('자동 고의4구')
runner_pattern = regex.compile('^[0-9]루주자 \p{Hangul}+ : [\p{Hangul}|0-9|\ ()->FD]+')
batter_pattern = regex.compile('^(([1-9]번타자)|(대타)) \p{Hangul}+')
src_pattern = regex.compile('[\p{Hangul}|0-9]+ \p{Hangul}+ : ')
dst_pattern = regex.compile('[\p{Hangul}|0-9|\ ]+ \(으\)로 교체')
dst_pattern2 = regex.compile('[\p{Hangul}|0-9|\ ]+\(으\)로 교체')
dst_pattern3 = regex.compile('[\p{Hangul}|0-9|\ ]+\(으\)로 수비위치 변경')
num_pattern = regex.compile('[1-3]루까지')


def parse_pa_result(text, ball_game):
    pa = pa_pattern.search(text)
    if pa is None:
        rc = 'parse error - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter']
        )
        return rc
    result = pa.group().split(':')[-1].strip()

    if result.find('삼진') >= 0:
        ball_game.strike_out()
    elif result.find('볼넷') >= 0:
        ball_game.four_ball()
    elif result.find('자동 고의4구') >= 0:
        ball_game.auto_ibb()
    elif result.find('고의4구') >= 0:
        ball_game.ibb()
    elif result.find('몸에') >= 0:
        ball_game.hbp()
    elif result.find('1루타') >= 0:
        ball_game.single()
    elif result.find('내야안타') >= 0:
        ball_game.infield_hit()
    elif result.find('번트안타') >= 0:
        ball_game.bunt_hit()
    elif result.find('안타') >= 0:
        ball_game.other_hit()
    elif result.find('2루타') >= 0:
        ball_game.double()
    elif result.find('3루타') >= 0:
        ball_game.triple()
    elif result.find('홈런') >= 0:
        ball_game.homerun()
    elif result.find('낫아웃 폭투') >= 0:
        ball_game.not_out_wp()
    elif result.find('낫아웃 포일') >= 0:
        ball_game.not_out_pb()
    elif result.find('낫 아웃') >= 0:
        ball_game.not_out()
    elif result.find('낫아웃 다른주자 수비') >= 0:
        # BUGBUG : 낫아웃 + 포스아웃인데 전부 실책 기록
        #
        # - 시작일 : 20190706
        # - 종료일 : 20190706
        #
        # CASE :
        # 20170829SKWO0 4회말 넥센 공격, 4번 김하성 타석
        # 낫아웃 이후 홈을 터치해서 포스 아웃
        # 하지만 '낫아웃 다른주자 수비' 텍스트를
        # 전부 실책으로 기록하고 있었음.
        #
        # OBJECT :
        # '실책' 발견되는 경우에만 실책 기록
        # 나머지는 포스 아웃
        if result.find('실책') >= 0:
            ball_game.roe()
        else:
            ball_game.force_out()
    elif result.find('낫아웃') >= 0:
        ball_game.not_out()
    elif result.find('땅볼로 출루') >= 0:
        ball_game.force_out()
    elif result.find('땅볼 아웃') >= 0:
        ball_game.field_out()
    elif result.find(' 플라이 아웃') >= 0:
        ball_game.field_out()
    elif result.find('인필드') >= 0:
        ball_game.field_out()
    elif result.find('파울플라이') >= 0:
        ball_game.field_out()
    elif result.find('라인드라이브 아웃') >= 0:
        ball_game.field_out()
    elif result.find(' 번트 아웃') >= 0:
        ball_game.field_out()
    elif result.find('병살타') >= 0:
        ball_game.double_play()
    elif result.find('희생번트 아웃') >= 0:
        ball_game.sac_hit()
    elif result.find('희생플라이 아웃') >= 0:
        ball_game.sac_fly()
    elif result.find('희생플라이아웃') >= 0:
        ball_game.sac_fly()
    elif result.find('쓰리번트') >= 0:
        ball_game.three_bunt()
    elif result.find('타구맞음') >= 0:
        ball_game.hit_by_ball_out()
    elif result.find('희생번트 실책') >= 0:
        ball_game.sac_hit_error()
    elif result.find('희생번트 야수선택') >= 0:
        ball_game.sac_hit_fc()
    elif result.find('야수선택') >= 0:
        ball_game.fc()
    elif result.find('실책') >= 0:
        ball_game.roe()
    elif result.find('타격방해') >= 0:
        ball_game.interference()
    elif result.find('삼중살') >= 0:
        ball_game.triple_play()
    elif result.find('부정타격') >= 0:
        ball_game.field_out()
    elif result.find('번트') >= 0:
        # '이용규 : 좌익수 앞 번트' 하나 있는 예외(20160709SSHH0)
        ball_game.other_hit()
    else:
        rc = 'unexpected PA result - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter']
        )
        return rc

    return True


def parse_pitch(text, ball_game, home_pitchers, away_pitchers, pitch_num, pid, bid, pts_data):
    # B/S/O 오류 있는 경우
    # 에러메시지 리턴하고 종료
    # 볼 > 3 or 스트라이크 > 2 or 아웃 > 2인 경우
    if ball_game.game_status['balls'] > 3:
        rc = 'balls > 3 - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석 {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc
    elif ball_game.game_status['strikes'] > 2:
        rc = 'strikes > 2 - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석 {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc
    elif ball_game.game_status['outs'] > 2:
        rc = 'outs > 3 - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석 {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc

    # 투구수가 앞선 텍스트와 같은 경우
    # parse 오류로 판단, 에러메시지 리턴하고 종료
    if pitch_num == ball_game.game_status['pitch_number']:
        # relay error
        rc = 'same pitch number - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석 {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc

    if ball_game.ball_and_not_hbp is True:
        # 연속으로 볼이 들어왔을 때
        # 사구가 아니라면 앞선 볼 상황을 출력
        ball_game.game_status['balls'] -= 1
        ball_game.print_row()
        # ball_game.print_row_debug()
        ball_game.game_status['balls'] += 1

    if ball_game.made_runs is True:
        # BUGBUG : 초구 전에 견제실책/보크/홈스틸 등으로 득점 발생시 row 출력 필요
        # - 시작일 : 20190706
        # - 종료일 : 20190706
        #
        # CASE :
        # 20190608HTNC0 4회말 NC 공격, 1번 박민우 타석
        # 초구 전에 견제 실책으로 2명 홈인
        #
        # 초구 전에 견제 실책으로 홈인이 발생하는 경우.
        # 직전 타석 결과로 정확하게 몇 점이 났는지
        # 지금은 기록하지 않고 있기 때문에(after play score)
        # 다음 타석에서 바로 견제로 홈인이 나는 경우
        # 이전 play로 생긴 run 계산에 오차가 생긴다.
        #
        # 아마도 초구 전 홈스틸/보크도 가능.
        #
        # OBJECT :
        # 투구 이외 요인으로 초구 전에 득점 발생하는 경우 row로 기록
        # 투구 데이터와 구분
        if ball_game.game_status['pitch_number'] == 0:
            # ball_game.print_row_debug()
            # handle pts data
            ball_game.game_status['pa_result'] = '투구 외 득점'
            ball_game.reset_pfx()
            ball_game.print_row()
            ball_game.game_status['pa_result'] = 'None'
        if ball_game.game_status['inning_top_bot'] is 0:
            ball_game.game_status['score_away'] += ball_game.runs_how_many
        else:
            ball_game.game_status['score_home'] += ball_game.runs_how_many
        ball_game.made_runs = False
        ball_game.runs_how_many = 0

    ball_game.game_status['pitch_number'] += 1
    ball_game.ball_and_not_hbp = False
    ball_game.set_hitter_to_base = False

    if ball_game.runner_change is True:
        if ball_game.change_3b is True:
            ball_game.game_status['on_3b'] = ball_game.next_3b

        if ball_game.change_2b is True:
            ball_game.game_status['on_2b'] = ball_game.next_2b

        if ball_game.change_1b is True:
            ball_game.game_status['on_1b'] = ball_game.next_1b

        ball_game.next_1b = None
        ball_game.next_2b = None
        ball_game.next_3b = None
        ball_game.change_1b = False
        ball_game.change_2b = False
        ball_game.change_3b = False
        ball_game.runner_change = False

    if ball_game.made_outs is True:
        ball_game.game_status['outs'] += ball_game.outs_how_many
        ball_game.made_outs = False
        ball_game.outs_how_many = 0

    if ball_game.made_errors is True:
        ball_game.made_errors = False

    pitch = pitch_pattern.search(text)
    if pitch is None:
        if ibb_pattern.search(text) is not None:
            pass
        else:
            rc = 'parse error - text : {}\n'.format(text)
            rc += '{}회 {}:{} 타석 {}구'.format(
                ball_game.game_status['inning'],
                ball_game.game_status['pitcher'],
                ball_game.game_status['batter'],
                pitch_num
            )
            return rc
    if ibb_pattern.search(text) is not None:
        result = 'AI'  # 자동 고의4구
    else:
        result = pitch.group().split(' ')[1]

    #  투수 교체된 경우
    #  change pid
    #  change throws
    if ball_game.prev_pid != pid:
        ball_game.game_status['pitcher_ID'] = pid
        throws = None
        if ball_game.game_status['inning_top_bot'] == 0:
            for p in home_pitchers:
                if pid == p['pCode']:
                    if p['hitType'] is not None:
                        throws = p['hitType'][0]
                    else:
                        throws = None
                    break
        else:
            for p in away_pitchers:
                if pid == p['pCode']:
                    if p['hitType'] is not None:
                        throws = p['hitType'][0]
                    else:
                        throws = None
                    break

        ball_game.game_status['throws'] = throws
        ball_game.prev_pid = pid
    ball_game.game_status['batter_ID'] = bid

    # handle pts data
    ball_game.reset_pfx()
    if pts_data is not None:
        ball_game.game_status['pitch_type'] = pts_data['stuff']
        ball_game.game_status['speed'] = pts_data['speed']
        ball_game.game_status['x0'] = pts_data['x0']
        ball_game.game_status['z0'] = pts_data['z0']
        ball_game.game_status['sz_top'] = pts_data['topSz']
        ball_game.game_status['sz_bot'] = pts_data['bottomSz']

        # raw data 추가
        ball_game.game_status['y0'] = pts_data['y0']
        ball_game.game_status['vx0'] = pts_data['vx0']
        ball_game.game_status['vy0'] = pts_data['vy0']
        ball_game.game_status['vz0'] = pts_data['vz0']
        ball_game.game_status['ax'] = pts_data['ax']
        ball_game.game_status['ay'] = pts_data['ay']
        ball_game.game_status['az'] = pts_data['az']

        # ball id 추가
        ball_game.game_status['pitchId'] = pts_data['pitchId']

        ax = pts_data['ax']
        ay = pts_data['ay']
        az = pts_data['az']
        vx0 = pts_data['vx0']
        vy0 = pts_data['vy0']
        vz0 = pts_data['vz0']
        x0 = pts_data['x0']
        y0 = pts_data['y0']
        z0 = pts_data['z0']
        cross_plate_y = pts_data['crossPlateY']

        t = (-vy0 - (vy0 * vy0 - 2 * ay * (y0 - cross_plate_y)) ** 0.5) / ay
        px = x0 + vx0 * t + ax * t * t * 0.5
        pz = z0 + vz0 * t + az * t * t * 0.5
        ball_game.game_status['px'] = round(px, 5)
        ball_game.game_status['pz'] = round(pz, 5)

        t40 = (-vy0 - (vy0 * vy0 - 2 * ay * (y0 - 40)) ** 0.5) / ay
        x40 = x0 + vx0 * t40 + 0.5 * ax * t40 * t40
        vx40 = vx0 + ax * t40
        z40 = z0 + vz0 * t40 + 0.5 * az * t40 * t40
        vz40 = vz0 + az * t40
        th = t - t40
        x_no_air = x40 + vx40 * th
        z_no_air = z40 + vz40 * th - 0.5 * 32.174 * th * th
        z_no_induced = z0 + vz0 * t

        ball_game.game_status['pfx_x'] = round((px - x_no_air) * 12, 5)
        ball_game.game_status['pfx_z'] = round((pz - z_no_air) * 12, 5)
        ball_game.game_status['pfx_x_raw'] = round(px * 12, 5)
        ball_game.game_status['pfx_z_raw'] = round((pz - z_no_induced) * 12, 5)

    if result == '볼':
        ball_game.get_ball()
    elif result == '스트라이크':
        ball_game.get_strike()
    elif result == '번트파울':
        ball_game.get_bunt_foul()
    elif result == '파울':
        ball_game.get_foul()
    elif result == '번트헛스윙':
        ball_game.get_bunt_swing_miss()
    elif result == '헛스윙':
        ball_game.get_swing_miss()
    elif result == '타격':
        ball_game.get_in_play()
    elif result == 'C':
        ball_game.get_ball()
    elif result == '12초':
        ball_game.get_ball()
    elif result == 'AI':
        pass
    else:
        rc = 'unexpected pitch result - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석 {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc

    return True


def parse_runner(text, ball_game):
    if ball_game.ball_and_not_hbp is True:
        # 폭투/포일로 인한 진루
        # status에 1볼 추가된 상황
        # '볼' row를 출력한다
        ball_game.game_status['balls'] -= 1
        ball_game.print_row()
        # ball_game.print_row_debug()
        # 다시 게임 status에 볼 1 추가
        ball_game.game_status['balls'] += 1
    ball_game.ball_and_not_hbp = False

    run = runner_pattern.search(text)
    if not run:
        if text.find('BH') > 0:
            # 특수한 예외 - 넘어감
            return True
        else:
            rc = 'parse error - text : {}\n'.format(text)
            rc += '{}회 {}:{} 타석'.format(
                ball_game.game_status['inning'],
                ball_game.game_status['pitcher'],
                ball_game.game_status['batter']
            )
            return rc
    result = run.group()
    src_base = int(result.split(' ')[0][0])
    src_name = result.split(' ')[1]

    if result.find('진루') > 0:
        if result.find('아웃') > 0:
            # 진루실패 아웃
            ball_game.runner_out(src_base, src_name)
        else:
            dst = 0
            tokens = result.split(' ')
            for i in range(len(tokens)):
                if tokens[i].find('까지') >= 0:
                    dst = int(num_pattern.search(tokens[i]).group()[0])
                    break
            if dst is 0:
                rc = 'parse error - 진루 목표지 확정 불가; text : {}\n'.format(text)
                rc += '{}회 {}:{} 타석'.format(
                    ball_game.game_status['inning'],
                    ball_game.game_status['pitcher'],
                    ball_game.game_status['batter']
                )
                return rc
            ball_game.runner_advance(src_base, src_name, dst)
    elif result.find('아웃') > 0:
        ball_game.runner_out(src_base, src_name)
    elif result.find('홈인') > 0:
        ball_game.runner_home_in(src_base, src_name)
    else:
        rc = 'unexpected type of runner text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter']
        )
        return rc

    if result.find('실책') > 0:
        ball_game.made_errors = True

    # BUGBUG : 볼넷/삼진 이후 실책으로 득점시 row print
    # - 시작일 : 20190706
    # - 종료일 : 미정
    #
    # 볼넷/삼진 이후 실책(폭투 등)으로 득점시
    # run value 계산할 때 play result에 따른 run scored 값이
    # 왜곡될 여지가 있다.
    #
    # WORKAROUND :
    #   run scored 값 계산할 때
    #   조건을 잘 걸어주면 해결 가능

    return True


def parse_change(text, ball_game):
    # 수비위치 변경, 교체
    change = False
    move = False
    src = src_pattern.search(text)
    if src:
        src_pos = src.group().strip().split(' ')[0]
        src_name = src.group().strip().split(' ')[1]
    else:
        rc = 'parse error - change text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter']
        )
        return rc

    dst = dst_pattern.search(text)
    dst2 = None
    dst_name = None
    if dst:
        dst_pos = dst.group().strip().split(' ')[0]
        dst_name = dst.group().strip().split(' ')[1]
        change = True
    else:
        dst2 = dst_pattern2.search(text)
        if dst2:
            dst_pos = dst2.group().strip().split(' ')[0]
            dst_name = dst2.group().strip().split('(')[0]
            change = True
        else:
            dst3 = dst_pattern3.search(text)
            if dst3:
                dst_pos = dst3.group().strip().split('(')[0]
                move = True
            else:
                rc = 'parse error - change text : {}\n'.format(text)
                rc += '{}회 {}:{} 타석'.format(
                    ball_game.game_status['inning'],
                    ball_game.game_status['pitcher'],
                    ball_game.game_status['batter']
                )
                return rc
            
    if dst or dst2:
        if dst_name is None:
            rc = 'parse error - no name in change text : {}\n'.format(text)
            rc += '{}회 {}:{} 타석'.format(
                ball_game.game_status['inning'],
                ball_game.game_status['pitcher'],
                ball_game.game_status['batter']
            )
            return rc
            
    if change:
        if (dst_pos == '대타') or (dst_pos == '대주자'):
            lineup_no = int(src_pos[0]) - 1
            if ball_game.game_status['inning_top_bot'] == 0:
                # away
                if (ball_game.game_status['away_lineup'][lineup_no]['pos'] == src_pos) and \
                        (ball_game.game_status['away_lineup'][lineup_no]['name'] == src_name):
                    ball_game.game_status['away_lineup'][lineup_no]['pos'] = dst_pos
                    ball_game.game_status['away_lineup'][lineup_no]['name'] = dst_name
            else:
                # home
                if (ball_game.game_status['home_lineup'][lineup_no]['pos'] == src_pos) and \
                        (ball_game.game_status['home_lineup'][lineup_no]['name'] == src_name):
                    ball_game.game_status['home_lineup'][lineup_no]['pos'] = dst_pos
                    ball_game.game_status['home_lineup'][lineup_no]['name'] = dst_name

            if dst_pos == '대주자':
                src_base = int(src_pos[0])
                if src_base == 1:
                    ball_game.game_status['on_1b'] = dst_name
                elif src_base == 2:
                    ball_game.game_status['on_2b'] = dst_name
                else:
                    ball_game.game_status['on_3b'] = dst_name
        else:
            # exception
            if dst_pos == '0':
                dst_pos = '투수'

            try:
                pos_num = position[dst_pos]
            except KeyError:
                rc = 'unexpected position - text : {}\n'.format(text)
                rc += '{}회 {}:{} 타석'.format(
                    ball_game.game_status['inning'],
                    ball_game.game_status['pitcher'],
                    ball_game.game_status['batter']
                )
                return rc
            if pos_num == 1:
                ball_game.game_status['pitcher'] = dst_name
            if ball_game.game_status['inning_top_bot'] == 0:
                # away
                for i in range(9):
                    if (ball_game.game_status['home_lineup'][i]['pos'] == src_pos) and \
                            (ball_game.game_status['home_lineup'][i]['name'] == src_name):
                        ball_game.game_status['home_lineup'][i]['pos'] = dst_pos
                        ball_game.game_status['home_lineup'][i]['name'] = dst_name
                        break
                ball_game.game_status[field_home[pos_num]] = dst_name
            else:
                # home
                for i in range(9):
                    if (ball_game.game_status['away_lineup'][i]['pos'] == src_pos) and \
                            (ball_game.game_status['away_lineup'][i]['name'] == src_name):
                        ball_game.game_status['away_lineup'][i]['pos'] = dst_pos
                        ball_game.game_status['away_lineup'][i]['name'] = dst_name
                        break
                ball_game.game_status[field_away[pos_num]] = dst_name
    elif move:
        pos_num = position[dst_pos]
        if pos_num == 1:
            ball_game.game_status['pitcher'] = src_name
        if ball_game.game_status['inning_top_bot'] == 0:
            # top; change home lineup
            ball_game.game_status[field_home[pos_num]] = src_name
            for i in range(9):
                if (ball_game.game_status['home_lineup'][i]['pos'] == src_pos) and \
                        (ball_game.game_status['home_lineup'][i]['name'] == src_name):
                    ball_game.game_status['home_lineup'][i]['pos'] = dst_pos
                    break
        else:
            # bot; change away lineup
            ball_game.game_status[field_away[pos_num]] = src_name
            for i in range(9):
                if (ball_game.game_status['away_lineup'][i]['pos'] == src_pos) and \
                        (ball_game.game_status['away_lineup'][i]['name'] == src_name):
                    ball_game.game_status['away_lineup'][i]['pos'] = dst_pos
                    break
    else:
        rc = 'unexpected type of text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter']
        )
        return rc

    return True


def parse_batter(text, home_batters, away_batters, bid, ball_game):
    bat = batter_pattern.search(text)
    if not bat:
        rc = 'parse error - text : {}\n'.format(text)
        rc += '{}회'.format(
            ball_game.game_status['inning'],
        )
        return rc
    result = batter_pattern.search(text).group().split(' ')[-1]
    bat_order = batter_pattern.search(text).group().split(' ')[0][0]

    # 대타면은 그냥 냅둔다.
    if bat_order.isnumeric() is True:
        # substitution - > @parse_change
        rc = ball_game.go_to_next_pa()
        if type(rc) is str:
            return rc

    # 초/말에 따라 타자 검색, 치는손 기록
    ball_game.game_status['batter'] = result
    if ball_game.game_status['inning_top_bot'] == 0:
        for b in away_batters:
            if b['pCode'] == bid:
                if b['hitType'] is None:
                    ball_game.game_status['stands'] = None
                else:
                    ball_game.game_status['stands'] = b['hitType'][2]
                break
    else:
        for b in home_batters:
            if b['pCode'] == bid:
                if b['hitType'] is None:
                    ball_game.game_status['stands'] = None
                else:
                    ball_game.game_status['stands'] = b['hitType'][2]
                break

    return True


def parse_text(text, text_type, ball_game, game_over,
               home_batters, away_batters, home_pitchers, away_pitchers,
               pitch_num, pid, bid, pts_data):
    if text_type == 0:
        rc = ball_game.go_to_next_inning()
        if type(rc) is str:
            return rc + '\ntext : {}'.format(text)
        else:
            return True
    elif text_type == 1:  # 투구
        rc = parse_pitch(text, ball_game, home_pitchers, away_pitchers, pitch_num, pid, bid, pts_data)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 2:  # 교체
        rc = parse_change(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 7:  # 시스템 메시지
        # system text
        # BUGBUG : 우천중단으로 종료시 마지막 공 출력 안됨
        # 20180809SKNC02018
        #      경기종료
        #      8번타자 나주환
        #        (22:34~23:14분) 우천으로 40분간 경기중단.
        #        - 4구 볼
        #        - 3구 볼
        #        - 2구 스트라이크
        #        - 1구 볼
        # '우천 중단' 텍스트 출력 후 경기 종료하는 경우의 flag 필요
        # 여기서 flag 세우고
        # 다음 text에서 종료(text_type == 99)일 때 flag 확인
        # game over 하기 전에
        # pa_result 안 정해진 row 남아있으면 출력.
        if (text.find('우천') > 0) & (text.find('중단') > 0):
            ball_game.rain_delay = True
        return True
    elif text_type == 8:  # 타자 이름
        # batter name
        parse_batter(text, home_batters, away_batters, bid, ball_game)
        return True
    elif text_type == 13:  # 타자 타석 결과
        rc = parse_pa_result(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 14:  # 주자 이동/아웃
        rc = parse_runner(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 23:  # 타자주자 홈인
        # bat-runner with home-in
        rc = parse_pa_result(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 24:  # 주자 홈인
        # base-runner with home-in
        rc = parse_runner(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 44:  # 파울 에러
        # foul error; pass
        return True
    elif text_type == 99:  # 경기 종료 메시지
        # game end
        # 정상 종료일 때만 여기로 이동
        # 끝내기, 비정상 종료는
        # 상위 parse_game에서 check_game_over를 통해 판단
        # 정상 종료가 아닌 경우 @go_to_next_pa로 이동, game_over 확인
        game_over[0] = True
        return True
    else:
        rc = 'unexpected type of text : {}\n'.format(text)
        rc += '{}회, text type : {}'.format(
            ball_game.game_status['inning'],
            text_type
        )
        return rc


header_row = ['pitch_type', 'pitcher', 'batter', 'pitcher_ID', 'batter_ID',
              'speed', 'pitch_result', 'pa_result', 'balls', 'strikes', 'outs',
              'inning', 'inning_topbot', 'score_away', 'score_home',
              'stands', 'throws', 'on_1b', 'on_2b', 'on_3b', 'px', 'pz', 'pfx_x', 'pfx_z', 'pfx_x_raw', 'pfx_z_raw',
              'x0', 'z0', 'sz_top', 'sz_bot', 'pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5',
              'pos_6', 'pos_7', 'pos_8', 'pos_9', 'game_date', 'home', 'away',
              'stadium', 'referee', 'pa_number', 'pitch_number',
              'y0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'pitchID']  # raw data 추가


def parse_game(game, lm=None, month_file=None, year_file=None):

    fp = open(game, 'r', encoding='utf-8')
    try:
        js = json.loads(fp.read(),
                        object_pairs_hook=OrderedDict)
    except UnicodeDecodeError:
        fp.close()
        fp = open(game, 'r', encoding='cp949')
        js = json.loads(fp.read(),
                        object_pairs_hook=OrderedDict)

    fp.close()

    ball_game = BallGame(game_date=game[:8])

    ball_game.set_home_away(game[10:12], game[8:10])
    ball_game.set_referee(js['referee'])
    ball_game.set_stadium(js['stadium'])
    ball_game.set_lineup(js)

    relayList = js['relayList']
    home_pitchers = js['homeTeamLineUp']['pitcher']
    away_pitchers = js['awayTeamLineUp']['pitcher']
    home_batters = js['homeTeamLineUp']['batter']
    away_batters = js['awayTeamLineUp']['batter']

    game_over = [False]
    rc = True

    keys = [int(x) for x in sorted(relayList.keys())]
    keys.sort()
    
    for k in keys:
        textOptionList = relayList[str(k)]['textOptionList']

        if textOptionList[0]['type'] == 99:
            # 끝내기인지, 정상적 종료인지 체크해야 한다.
            if ball_game.rain_delay is True:
                ball_game.go_to_next_pa()
                game_over[0] = True
                break

            if ball_game.check_game_over() is not True:
                ball_game.go_to_next_pa()
                game_over[0] = True
                break
        if ball_game.rain_delay is True:
            ball_game.rain_delay = False

        if 'ptsOptionList' in relayList[str(k)].keys():
            ptsOptionList = relayList[str(k)]['ptsOptionList']
        else:
            ptsOptionList = []

        pts_dict = {}
        if len(ptsOptionList) > 0:
            pts_dict = {x['pitchId']: x for x in ptsOptionList}

        for i in range(len(textOptionList)):
            text = textOptionList[i]['text']
            text_type = textOptionList[i]['type']
            pid = textOptionList[i]['currentGameState']['pitcher']
            bid = textOptionList[i]['currentGameState']['batter']

            pts_data = None

            if (len(ptsOptionList) > 0) and\
                    (text_type == 1):
                if 'ptsPitchId' in textOptionList[i].keys():
                    ptsPitchId = textOptionList[i]['ptsPitchId']
                    if ptsPitchId in pts_dict.keys():
                        pts_data = pts_dict[ptsPitchId]
                        if pts_data['ballcount'] == textOptionList[i]['pitchNum']:
                            pts_data['speed'] = textOptionList[i]['speed']
                            pts_data['stuff'] = textOptionList[i]['stuff']
                        else:
                            pts_data = None

            pitch_num = None
            if text_type == 1:
                try:
                    pitch_num = textOptionList[i]['pitchNum']
                except KeyError:
                    pitch_num = None

            rc = parse_text(text, text_type, ball_game, game_over,
                            home_batters, away_batters, home_pitchers, away_pitchers,
                            pitch_num, pid, bid, pts_data)
            if type(rc) is str:
                lm.log('error - ignore and run rest')
                rc += '    game id : {}\n'.format(game[:13])
                lm.log(rc)
                game_over[0] = True
                break
            if game_over[0]:
                break

        if game_over[0]:
            break

    returns = 1
    if type(rc) is str:
        of = game[:13] + '_error.csv'
        returns = 1
    else:
        of = game[:13] + '.csv'
        returns = True

    ofp = open(of, 'w', newline='\n')
    cf = csv.writer(ofp)
    cf.writerow(header_row)
    for i in range(len(ball_game.text_row)):
        cf.writerow(ball_game.text_row[i])
        if month_file is not None:
            month_file.writerow(ball_game.text_row[i])
        if year_file is not None:
            year_file.writerow(ball_game.text_row[i])
    ofp.close()

    return returns


def parse_main(args, lm=None):
    mon_start = args[0]
    mon_end = args[1]
    year_start = args[2]
    year_end = args[3]

    if lm is not None:
        lm.resetLogHandler()
        lm.setLogPath(os.getcwd())
        lm.setLogFileName('relay_parsing_log.txt')
        lm.cleanLog()
        lm.createLogHandler()
        lm.log('---- Relay Text Parse Log ----')

    if not os.path.isdir('pbp_data'):
        print('no data folder')
        lm.log('no data folder')
        return False

    os.chdir('pbp_data')

    print("##################################################")
    print("#######          PARSE RELAY TEXT          #######")
    print("##################################################")
    total_done = 0
    total_skipped = 0
    for year in range(year_start, year_end + 1):
        print(" Year {}".format(year))
        if not os.path.isdir(str(year)):
            print('no year dir : {}'.format(year))
            lm.log('no year dir : {}'.format(year))
            os.chdir('..')
            return False

        os.chdir(str(year))

        year_filename = '{}.csv'.format(year)
        yf = open(year_filename, 'w', newline='\n')
        cy = csv.writer(yf)
        cy.writerow(header_row)

        for month in range(mon_start, mon_end + 1):
            print("  Month {}".format(month))
            if not os.path.isdir(str(month)):
                print('no month dir : {}'.format(month))
                lm.log('no month dir : {}'.format(month))
                os.chdir('../../')
                return False

            os.chdir(str(month))

            games = [f for f in os.listdir('.') if (os.path.isfile(f)) and
                     (f.lower().find('relay.json') > 0) and
                     (os.path.getsize(f) > 512)]
            if not len(games) > 0:
                print('no games : {}/{}'.format(year, month))
                lm.log('no games : {}/{}'.format(year, month))
                os.chdir('..')
                continue

            games.sort()

            month_filename = '{}_{}.csv'.format(year, month)
            mf = open(month_filename, 'w', newline='\n')
            cm = csv.writer(mf)
            cm.writerow(header_row)

            done = 0
            skipped = 0
            for game in games:
                rc = parse_game(game, lm, month_file=cm, year_file=cy)
                if type(rc) is int:
                    skipped += 1
                    total_skipped += 1
                elif type(rc) is str:
                    print('\nparse game failure')
                    lm.log('parse game failure')
                    os.chdir('../../../')
                    return False
                else:
                    done += 1
                    total_done += 1
                print_progress('    Parsing: ', len(games), done, skipped)
            print()
            print('          Done : {}'.format(done))
            print('          Skipped : {}'.format(skipped))

            # end
            mf.close()
            os.chdir('..')

        # end
        yf.close()
        os.chdir('..')

    # end
    os.chdir('..')
    print('Done : {}'.format(total_done))
    print('Skipped : {}'.format(total_skipped))
