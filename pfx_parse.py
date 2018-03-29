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
        ]
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

    def reset_pfx(self):
        self.game_status['pitch_type'] = None
        self.game_status['speed'] = None
        self.game_status['px'] = None
        self.game_status['pz'] = None
        self.game_status['pfx_x'] = None
        self.game_status['pfx_z'] = None
        self.game_status['x0'] = None
        self.game_status['z0'] = None
        self.game_status['sz_top'] = None
        self.game_status['sz_bot'] = None

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
        self.text_row.append(row)

    def print_row_debug(self):
        # for debug
        row = str(self.game_status['pitch_result']) + ', '
        row += str(self.game_status['pitcher']) + ', '
        row += str(self.game_status['batter']) + ', '
        row += str(self.game_status['balls']) + '/'
        row += str(self.game_status['strikes']) + '/'
        row += str(self.game_status['outs']) + ', '
        row += str(self.game_status['inning'])
        if self.game_status['inning_top_bot'] == 0:
            row += '초,'
        else:
            row += '말,'
        row += str(self.game_status['score_away']) + ':'
        row += str(self.game_status['score_home']) + ', '
        row += str(self.game_status['on_1b']) + '/'
        row += str(self.game_status['on_2b']) + '/'
        row += str(self.game_status['on_3b']) + ', '
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
        if self.made_in_play is True:
            self.print_row()
            # self.print_row_debug()
        elif self.game_status['strikes'] == 3:
            self.game_status['strikes'] = 2
            self.print_row()
            # self.print_row_debug()
        elif self.game_status['balls'] == 4:
            self.game_status['balls'] = 3
            self.print_row()
            # self.print_row_debug()
        elif self.game_status['pa_result'] is not None:
            if self.game_status['pa_result'].find('몸에') > -1:
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
                            # (타자 이병규, 1루주자 이병규)
                            # -> 같은 주자로 보지 않는다
                            # 기존(새로 들어온) 1루 주자는 내버려둔다.
                            self.set_hitter_to_base = False
                        else:
                            # 1루를 비운다
                            self.next_1b = None
                else:
                    # 기존 1루 주자 진루.
                    self.next_1b = None
                    self.change_1b = True
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
                            # (타자 이병규, 1루주자 이병규)
                            # -> 같은 주자로 보지 않는다
                            # 기존(새로 들어온) 1루 주자는 내버려둔다.
                            self.set_hitter_to_base = False
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
                            # (타자 이병규, 2루주자 이병규)
                            # -> 같은 주자로 보지 않는다
                            # 기존(새로 들어온) 2루 주자는 내버려둔다.
                            self.set_hitter_to_base = False
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
                        # (타자 이병규, 1루주자 이병규)
                        # -> 같은 주자로 보지 않는다
                        # 기존(새로 들어온) 1루 주자는 내버려둔다.
                        self.set_hitter_to_base = False
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
                        # (타자 이병규, 2루주자 이병규)
                        # -> 같은 주자로 보지 않는다
                        # 기존(새로 들어온) 2루 주자는 내버려둔다.
                        self.set_hitter_to_base = False
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
                        # (타자 이병규, 3루주자 이병규)
                        # -> 같은 주자로 보지 않는다
                        # 기존(새로 들어온) 3루 주자는 내버려둔다.
                        self.set_hitter_to_base = False
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
                        # (타자 이병규, 1루주자 이병규)
                        # -> 같은 주자로 보지 않는다
                        # 기존(새로 들어온) 1루 주자는 내버려둔다.
                        self.set_hitter_to_base = False
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
                        # (타자 이병규, 2루주자 이병규)
                        # -> 같은 주자로 보지 않는다
                        # 기존(새로 들어온) 2루 주자는 내버려둔다.
                        self.set_hitter_to_base = False
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
                        # (타자 이병규, 3루주자 이병규)
                        # -> 같은 주자로 보지 않는다
                        # 기존(새로 들어온) 3루 주자는 내버려둔다.
                        self.set_hitter_to_base = False
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
    elif result.find('고의4구') >= 0:
        ball_game.ibb()
    elif result.find('몸에') >= 0:
        ball_game.hbp()
    elif result.find('1루타') >= 0:
        ball_game.single()
    elif result.find('내야안타') >= 0:
        ball_game.infield_hit()
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
        # 실책 기록
        ball_game.roe()
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
    if ball_game.game_status['balls'] > 3:
        rc = 'balls > 3 - text : {}\n'.format(text)
        rc += '{}회 {}:{} {}구'.format(
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
        rc += '{}회 {}:{} {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc

    if pitch_num == ball_game.game_status['pitch_number']:
        # relay error
        rc = 'same pitch number - text : {}\n'.format(text)
        rc += '{}회 {}:{} {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc

    if ball_game.ball_and_not_hbp is True:
        ball_game.game_status['balls'] -= 1
        ball_game.print_row()
        # ball_game.print_row_debug()
        ball_game.game_status['balls'] += 1

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

    if ball_game.made_runs is True:
        if ball_game.game_status['inning_top_bot'] is 0:
            ball_game.game_status['score_away'] += ball_game.runs_how_many
        else:
            ball_game.game_status['score_home'] += ball_game.runs_how_many
        ball_game.made_runs = False
        ball_game.runs_how_many = 0

    pitch = pitch_pattern.search(text)
    if pitch is None:
        rc = 'parse error - text : {}\n'.format(text)
        rc += '{}회 {}:{} 타석 {}구'.format(
            ball_game.game_status['inning'],
            ball_game.game_status['pitcher'],
            ball_game.game_status['batter'],
            pitch_num
        )
        return rc
    result = pitch.group().split(' ')[1]

    if ball_game.prev_pid != pid:
        # change pid
        # change throws
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
        ball_game.game_status['pfx_x'] = round((px - x_no_air) * 12, 5)
        ball_game.game_status['pfx_z'] = round((pz - z_no_air) * 12, 5)

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
        ball_game.game_status['balls'] -= 1
        ball_game.print_row()
        # ball_game.print_row_debug()
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

    # substitution - > @parse_change
    rc = ball_game.go_to_next_pa()
    if type(rc) is str:
        return rc
    else:
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
    elif text_type == 1:
        rc = parse_pitch(text, ball_game, home_pitchers, away_pitchers, pitch_num, pid, bid, pts_data)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 2:
        rc = parse_change(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 7:
        # system text
        return True
    elif text_type == 8:
        # batter name
        parse_batter(text, home_batters, away_batters, bid, ball_game)
        return True
    elif text_type == 13:
        rc = parse_pa_result(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 14:
        rc = parse_runner(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 23:
        # bat-runner with home-in
        rc = parse_pa_result(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 24:
        # base-runner with home-in
        rc = parse_runner(text, ball_game)
        if type(rc) is str:
            return rc
        else:
            return True
    elif text_type == 44:
        # foul error; pass
        return True
    elif text_type == 99:
        # game end
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
              'stands', 'throws', 'on_1b', 'on_2b', 'on_3b', 'px', 'pz', 'pfx_x', 'pfx_z',
              'x0', 'z0', 'sz_top', 'sz_bot', 'pos_1', 'pos_2', 'pos_3', 'pos_4', 'pos_5',
              'pos_6', 'pos_7', 'pos_8', 'pos_9', 'game_date', 'home', 'away',
              'stadium', 'referee', 'pa_number', 'pitch_number']


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

    rl = js['relayList']
    home_pitchers = js['homeTeamLineUp']['pitcher']
    away_pitchers = js['awayTeamLineUp']['pitcher']
    home_batters = js['homeTeamLineUp']['batter']
    away_batters = js['awayTeamLineUp']['batter']

    game_over = [False]
    rc = True

    for k in range(len(rl.keys())):
        if str(k) not in rl.keys():
            lm.log('error - ignore and run rest')
            rc = 'no rl key : game id - {}'.format(game)
            lm.log(rc)
            return 1
        text_set = rl[str(k)]['textOptionList']

        if text_set[0]['type'] == 99:
            game_over[0] = True
            break
        pts_set = rl[str(k)]['ptsOptionList']
        pts_dict = {}
        if len(pts_set) > 0:
            for i in range(len(pts_set)):
                pitch_id = pts_set[i]['pitchId']
                pts_dict[pitch_id] = pts_set[i]

        for i in range(len(text_set)):
            text = text_set[i]['text']
            text_type = text_set[i]['type']
            pid = text_set[i]['currentGameState']['pitcher']
            bid = text_set[i]['currentGameState']['batter']

            pts_data = None
            if (len(pts_set) > 0) and\
                    (text_type == 1) and\
                    (len(text_set[i]['ptsPitchId']) > 2):
                if text_set[i]['ptsPitchId'] in pts_dict.keys():
                    pts_data = pts_dict[text_set[i]['ptsPitchId']]
                    pts_data['speed'] = text_set[i]['speed']
                    pts_data['stuff'] = text_set[i]['stuff']

            pitch_num = None
            if text_type == 1:
                try:
                    pitch_num = text_set[i]['pitchNum']
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

    if type(rc) is str:
        return 1
    else:
        of = game[:13] + '.csv'
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

        return True


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
