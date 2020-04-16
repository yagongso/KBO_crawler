# bb_convert_to_csv.py
#
# IN: year/month - target match time period
# OUT: CSV files for target match time period
#
# (1) read each JSON files.
# (2) load JSON data structure.
# (3) reconstruct as string list.
# (4) dump to the CSV file.

import os
import json
import sys
import csv
from logManager import getTracebackStr


pos_lst = [
    '투수', '포수', '1루수', '2루수',
    '3루수', '유격수', '좌익수', '중견수',
    '우익수', '좌중간', '우중간'
]

res_lst = [
    '1루타', '2루타', '3루타', '홈런', '실책',
    '땅볼', '플라이 아웃', '희생플라이',
    '희생번트', '야수선택', '삼진',
    '스트라이크', '타격방해',
    '라인드라이브', '병살타',
    '번트', '내야안타'
]

positions = dict(enumerate(pos_lst))

results = dict(enumerate(res_lst))

teams = {
    "HT": "KIA",
    "SS": "삼성",
    "WO": "넥센",
    "SK": "SK",
    "HH": "한화",
    "LG": "LG",
    "OB": "두산",
    "LT": "롯데",
    "NC": "NC",
    "KT": "KT",
}

fieldNames = ['date', 'batterName', 'pitcherName', 'inn',
              'gx', 'gy', 'fielder_position', 'actual_result',
              'result', 'batterTeam', 'pitcherTeam', 'seqno']


def find_pos(detail_result):
    for i in range(len(positions)):
        if detail_result.find(positions[i]):
            return [True, positions[i]]
    return [False, '']


def find_result(detail_result):
    for i in range(len(results)):
        if detail_result.find(results[i]):
            return [True, results[i]]
    return [False, '']


def write_data(ball_ids, csv_file, js, match_info, is_home, lm):
    if is_home is True:
        homeaway = 'home'
    else:
        homeaway = 'away'

    match_date = str(match_info[0:8])
    match_away = str(match_info[8:10])
    match_home = str(match_info[10:12])
    date = '{0}-{1}-{2}'.format(match_date[0:4], match_date[4:6], match_date[6:8])

    csv_writer = csv.DictWriter(csv_file, delimiter=',',
                                dialect='excel', fieldnames=fieldNames,
                                lineterminator='\n')

    for ball in ball_ids:
        ball_key = str(ball)

        # initialization
        batter_name = pitcher_name = result = gx = gy = ''
        detail_result = inn = seq_no = batter_team = pitcher_team = ''

        try:
            batter_name = js['ballsInfo'][homeaway][ball_key]['batterName']
            pitcher_name = js['ballsInfo'][homeaway][ball_key]['pitcherName']
            result = js['ballsInfo'][homeaway][ball_key]['result']
            gx = js['ballsInfo'][homeaway][ball_key]['gx']
            gy = js['ballsInfo'][homeaway][ball_key]['gy']
            detail_result = js['ballsInfo'][homeaway][ball_key]['detailResult']
            inn = js['ballsInfo'][homeaway][ball_key]['inn']
            seq_no = js['ballsInfo'][homeaway][ball_key]['seqno']
            batter_team = teams[match_away]
            pitcher_team = teams[match_home]
        except KeyError as e:
            print()
            print(getTracebackStr())
            lm.bugLog(getTracebackStr())
            lm.bugLog("Key Value Error : {}".format(e))
            lm.bugLog("Match Info : {}".format(match_info))
            lm.bugLog("homeaway : {}".format(homeaway))
            lm.bugLog("ball id : {}".format(ball))
            # lm.bugLog("all ball id : {}".format(ball_ids))
            lm.killLogManager()
            exit(1)

        [posfound, fielder_position] = find_pos(detail_result)
        [resfound, actual_result] = find_pos(detail_result)
        if not posfound:
            print()
            print("Invalid Position : {}".format(detail_result))
            lm.buglog("Invalid Position : {}".format(detail_result))
            lm.bugLog("Match Info : {}".format(match_info))
            lm.bugLog("homeaway : {}".format(homeaway))
            lm.bugLog("ball id : {}".format(ball))
            # lm.bugLog("all ball id : {}".format(ball_ids))
            lm.killLogManager()
            exit(1)
        if not resfound:
            print()
            print("Invalid Result : {}".format(detail_result))
            lm.buglog("Invalid Result : {}".format(detail_result))
            lm.bugLog("Match Info : {}".format(match_info))
            lm.bugLog("homeaway : {}".format(homeaway))
            lm.bugLog("ball id : {}".format(ball))
            # lm.bugLog("all ball id : {}".format(ball_ids))
            lm.killLogManager()
            exit(1)

        d = {
            'date': date,
            'batterName': batter_name,
            'pitcherName': pitcher_name,
            'inn': inn,
            'gx': gx,
            'gy': gy,
            'fielder_position': fielder_position,
            'actual_result': actual_result,
            'result': result,
            'batterTeam': batter_team,
            'pitcherTeam': pitcher_team,
            'seqno': seq_no,
        }
        csv_writer.writerow(d)


def write_csv(csv_file, csv_month, csv_year, js, match_info, lm):
    # away first
    hr_id = js['teamsInfo']['away']['hr']
    hit_id = js['teamsInfo']['away']['hit']
    o_id = js['teamsInfo']['away']['o']

    write_data(hr_id, csv_file, js, match_info, False, lm)
    write_data(hit_id, csv_file, js, match_info, False, lm)
    write_data(o_id, csv_file, js, match_info, False, lm)

    write_data(hr_id, csv_month, js, match_info, False, lm)
    write_data(hit_id, csv_month, js, match_info, False, lm)
    write_data(o_id, csv_month, js, match_info, False, lm)

    write_data(hr_id, csv_year, js, match_info, False, lm)
    write_data(hit_id, csv_year, js, match_info, False, lm)
    write_data(o_id, csv_year, js, match_info, False, lm)

    # home next
    hr_id = js['teamsInfo']['home']['hr']
    hit_id = js['teamsInfo']['home']['hit']
    o_id = js['teamsInfo']['home']['o']

    write_data(hr_id, csv_file, js, match_info, True, lm)
    write_data(hit_id, csv_file, js, match_info, True, lm)
    write_data(o_id, csv_file, js, match_info, True, lm)

    write_data(hr_id, csv_month, js, match_info, True, lm)
    write_data(hit_id, csv_month, js, match_info, True, lm)
    write_data(o_id, csv_month, js, match_info, True, lm)

    write_data(hr_id, csv_year, js, match_info, True, lm)
    write_data(hit_id, csv_year, js, match_info, True, lm)
    write_data(o_id, csv_year, js, match_info, True, lm)


def bb_convert_to_csv(mon_start, mon_end, year_start, year_end, lm=None):
    if not os.path.isdir("./bb_data"):
        print("DOWNLOAD DATA FIRST")
        exit(1)
    os.chdir("./bb_data")
    # current path : ./bb_data/
    print("##################################################")
    print("###### CONVERT BB DATA(JSON) TO CSV FORMAT #######")
    print("##################################################")

    for year in range(year_start, year_end+1):
        print("  for Year {0}...".format(str(year)))

        if not os.path.isdir("./{0}".format(str(year))):
            print(os.getcwd())
            print("DOWNLOAD YEAR {0} DATA FIRST".format(str(year)))
            continue

        csv_year = ''  # dummy
        for month in range(mon_start, mon_end+1):
            print("    Month {0}... ".format(str(month)))

            if month == mon_start:
                if sys.platform == 'win32':
                    csv_year = open("{0}/{1}.csv".format(str(year), str(year)), 'w',
                                    encoding='cp949')
                else:
                    csv_year = open("{0}/{1}.csv".format(str(year), str(year)), 'w',
                                    encoding='utf-8')

            csv_writer = csv.DictWriter(csv_year, delimiter=',',
                                        dialect='excel', fieldnames=fieldNames,
                                        lineterminator='\n')
            csv_writer.writeheader()

            if month < 10:
                mon = '0{}'.format(str(month))
            else:
                mon = str(month)
            if not os.path.isdir("./{0}/{1}".format(str(year), mon)):
                print(os.getcwd())
                print("DOWNLOAD MONTH {0} DATA FIRST".format(str(month)))
                continue
            os.chdir("{0}/{1}".format(str(year), mon))
            # current path : ./bb_data/YEAR/MONTH/

            # 파일 이름에 'bb.json'이 붙고 사이즈 1024byte 이상인 파일만 체크
            files = [f for f in os.listdir('.') if (os.path.isfile(f) and
                                                    (f.find('bb.json') > 0) and
                                                    (os.path.getsize(f) > 1024))
                     ]
            mon_file_num = len(files)

            if not mon_file_num > 0:
                print(os.getcwd())
                print("DOWNLOAD MONTH {0} DATA FIRST".format(str(month)))
                os.chdir('../../')
                # current path : ./bb_data/
                continue

            bar_prefix = '    Converting: '
            print('\r{}[waiting]'.format(bar_prefix), end="")

            if not os.path.isdir("./csv"):
                os.mkdir("./csv")

            csv_month_name = "{0}{1}.csv".format(str(year), mon)

            if sys.platform == 'win32':
                csv_month = open(csv_month_name, 'w', encoding='cp949')
            else:
                csv_month = open(csv_month_name, 'w', encoding='utf-8')

            csv_writer = csv.DictWriter(csv_month, delimiter=',',
                                        dialect='excel', fieldnames=fieldNames,
                                        lineterminator='\n')
            csv_writer.writeheader()
            done = 0

            lm.resetLogHandler()
            lm.setLogPath(os.getcwd() + '/log/')
            lm.setLogFileName('bbConvertLog.txt')
            lm.cleanLog()
            lm.createLogHandler()

            for f in files:
                js_in = open(f, 'r', encoding='utf-8')
                js = json.loads(js_in.read(), encoding='utf-8')
                js_in.close()

                match_info = f[:16]

                # (5) open csv file
                if sys.platform == 'win32':
                    csv_file = open('./csv/{0}.csv'.format(match_info), 'w', encoding='cp949')
                else:
                    csv_file = open('./csv/{0}.csv'.format(match_info), 'w', encoding='utf-8')

                csv_writer = csv.DictWriter(csv_file, delimiter=',',
                                            dialect='excel', fieldnames=fieldNames,
                                            lineterminator='\n')
                csv_writer.writeheader()

                write_csv(csv_file, csv_month, csv_year, js, match_info, lm)

                csv_file.close()

                done += 1
                lm.log('{} convert'.format(match_info))

                if mon_file_num > 30:
                    progress_pct = (float(done) / float(mon_file_num))
                    bar = '+' * int(progress_pct*30) + '-'*(30-int(progress_pct*30))
                    print('\r{}[{}] {} / {}, {:2.1f} %'.format(bar_prefix, bar, done, mon_file_num,
                                                               progress_pct * 100), end="")
                elif mon_file_num == 0:
                    mon_file_num = 0
                    # do nothing
                else:
                    bar = '+' * done + '-' * (mon_file_num - done)
                    print('\r{}[{}] {} / {}, {:2.1f} %'.format(bar_prefix, bar, done, mon_file_num,
                                                               float(done) / float(mon_file_num) * 100), end="")

            csv_month.close()
            print()
            print('        Converted {0} files'.format(str(done)))
            os.chdir('../../')
            # current path : ./bb_data/

        csv_year.close()
        # current path : ./bb_data/
    os.chdir('..')
    # current path : ./
    print("JSON to CSV convert Done.")
