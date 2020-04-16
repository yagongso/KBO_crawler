# pbp_download.py
#
# 'N' gameday crawler
#
# Play-by-play 데이터를 parse하기 위해 raw resource를 가져온다.
# xml 포맷의 데이터에서 원하는 JSON 영역만 추출해온다.

import os
import sys
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import requests
import numpy as np
import pandas as pd
import datetime
import time
import regex
import collections
import ast
import csv

# custom library
import logManager
from utils import print_progress
from utils import check_url
from utils import get_args

regular_start = {
    '3333': '0101', # playoff
    '4444': '0101', # playoff
    '5555': '0101', # playoff
    '7777': '0101', # playoff
    '2008': '0329',
    '2009': '0404',
    '2010': '0327',
    '2011': '0402',
    '2012': '0407',
    '2013': '0330',
    '2014': '0329',
    '2015': '0328',
    '2016': '0401',
    '2017': '0331',
    '2018': '0324',
    '2019': '0323',
}

playoff_start = {
    '3333': '1231', # playoff
    '4444': '1231', # playoff
    '5555': '1231', # playoff
    '7777': '1231', # playoff
    '2008': '1008',
    '2009': '0920',
    '2010': '1005',
    '2011': '1008',
    '2012': '1008',
    '2013': '1008',
    '2014': '1019',
    '2015': '1010',
    '2016': '1021',
    '2017': '1010',
    '2018': '1015',
    '2019': '1003',
}

teams = ['LG', 'KT', 'NC', 'SK', 'WO', 'SS', 'HH', 'HT', 'LT', 'OB']


def get_game_ids(args):
    # set url prefix
    timetable_url = "https://sports.news.naver.com/kbaseball/schedule/index.nhn?month="

    # parse arguments
    mon_start = args[0]
    mon_end = args[1]
    year_start = args[2]
    year_end = args[3]

    # get game ids
    game_ids = {}

    for year in range(year_start, year_end + 1):
        year_ids = {}

        for month in range(mon_start, mon_end + 1):
            month_ids = []
            timetable = timetable_url + '{}&year={}'.format(str(month), str(year))

            response = requests.get(timetable)
            table_page = response.text
            response.close()
            soup = BeautifulSoup(table_page, 'lxml')
            buttons = soup.findAll('span', attrs={'class': 'td_btn'})

            for btn in buttons:
                address = btn.a['href']
                game_id = address.split('gameId=')[1]
                month_ids.append(game_id)

            year_ids[month] = month_ids

        game_ids[year] = year_ids

    return game_ids


def download_relay(args, lm=None):
    # return True or False
    relay_url = 'http://m.sports.naver.com/ajax/baseball/gamecenter/kbo/relayText.nhn'
    record_url = 'http://m.sports.naver.com/ajax/baseball/gamecenter/kbo/record.nhn'

    game_ids = get_game_ids(args)
    if (game_ids is None) or (len(game_ids) == 0):
        print('no game ids')
        print('args: {}'.format(args))
        if lm is not None:
            lm.log('no game ids')
            lm.log('args: {}'.format(args))
        return False

    if lm is not None:
        lm.resetLogHandler()
        lm.setLogPath(os.getcwd())
        lm.setLogFileName('relay_download_log.txt')
        lm.cleanLog()
        lm.createLogHandler()
        lm.log('---- Relay Text Download Log ----')

    if not os.path.isdir('pbp_data'):
        os.mkdir('pbp_data')
    os.chdir('pbp_data')
    # path: pbp_data

    print("##################################################")
    print("######        DOWNLOAD RELAY DATA          #######")
    print("##################################################")

    for year in game_ids.keys():
        start1 = time.time()
        print(" Year {}".format(year))
        if len(game_ids[year]) == 0:
            print('month id is empty')
            print('args: {}'.format(args))
            if lm is not None:
                lm.log('month id is empty')
                lm.log('args : {}'.format(args))
            return False

        if not os.path.isdir(str(year)):
            os.mkdir(str(year))
        os.chdir(str(year))
        # path: pbp_data/year

        for month in game_ids[year].keys():
            start2 = time.time()
            print("  Month {}".format(month))
            if len(game_ids[year][month]) == 0:
                print('month id is empty')
                print('args: {}'.format(args))
                if lm is not None:
                    lm.log('month id is empty')
                    lm.log('args : {}'.format(args))
                return False

            if not os.path.isdir(str(month)):
                os.mkdir(str(month))
            os.chdir(str(month))
            # path: pbp_data/year/month

            # download
            done = 0
            skipped = 0
            for game_id in game_ids[year][month]:
                if (int(game_id[:4]) < 2008) or (int(game_id[:4]) > 7777):
                    skipped += 1
                    continue
                if (int(game_id[:4]) == datetime.datetime.now().year) and (int(game_id[4:8]) > int(datetime.datetime.now().date().strftime('%m%d'))):
                    skipped += 1
                    continue
                if int(game_id[4:8]) < int(regular_start[game_id[:4]]):
                    skipped += 1
                    continue
                if int(game_id[4:8]) >= int(playoff_start[game_id[:4]]):
                    skipped += 1
                    continue
                if game_id[8:10] not in teams:
                    skipped += 1
                    continue

                if not check_url(relay_url):
                    skipped += 1
                    if lm is not None:
                        lm.log('URL error : {}'.format(relay_url))
                    continue

                if (int(game_id[:4]) == datetime.datetime.now().year) &\
                   (int(game_id[4:6]) == datetime.datetime.now().month) &\
                   (int(game_id[6:8]) == datetime.datetime.now().day):
                        # do nothing
                       done = done
                elif (os.path.isfile(game_id + '_relay.json')) and \
                        (os.path.getsize(game_id + '_relay.json') > 0):
                    done += 1
                    if lm is not None:
                        lm.log('File Duplicate : {}'.format(game_id))
                    continue

                params = {
                    'gameId': game_id,
                    'half': '1'
                }

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/59.0.3071.115 Safari/537.36',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Host': 'm.sports.naver.com',
                    'Referer': 'http://m.sports.naver.com/baseball/gamecenter/kbo/index.nhn?&gameId='
                               + game_id
                               + '&tab=relay'
                }

                response = requests.get(relay_url, params=params, headers=headers)

                if response is not None:
                    txt = {}
                    js = response.json()
                    if isinstance(js, str):
                        js = json.loads(js)
                    last_inning = js['currentInning']

                    if last_inning is None:
                        skipped += 1
                        lm.log('Gameday not found : {}'.format(game_id))
                        continue

                    txt['relayList'] = {}
                    for i in range(len(js['relayList'])):
                        txt['relayList'][js['relayList'][i]['no']] = js['relayList'][i]
                    txt['homeTeamLineUp'] = js['homeTeamLineUp']
                    txt['awayTeamLineUp'] = js['awayTeamLineUp']

                    txt['stadium'] = js['schedule']['stadium']

                    response.close()

                    for inn in range(2, last_inning + 1):
                        params = {
                            'gameId': game_id,
                            'half': str(inn)
                        }

                        response = requests.get(relay_url, params=params, headers=headers)
                        if response is not None:
                            js = response.json()
                            if isinstance(js, str):
                                js = json.loads(js)
                                #js = ast.literal_eval(js)

                            # BUGBUG
                            # 문자중계 텍스트에 비 unicode 문자가 들어간 경우.
                            # gameid : 20180717LGWO02018
                            # 문제가 되는 텍스트: \ufffd (REPLACEMENT CHARACTER) - cp949로 저장 불가
                            # 해결책: cp949로 encoding 불가능한 문자가 있을 때는 blank text로 교체.
                            for i in range(len(js['relayList'])):
                                txt['relayList'][js['relayList'][i]['no']] = js['relayList'][i]
                                texts = txt['relayList'][js['relayList'][i]['no']]['textOptionList']
                                for i in range(len(texts)):
                                    try:
                                        texts[i]['text'].encode('cp949')
                                    except UnicodeEncodeError:
                                        texts[i]['text'] = ''
                        else:
                            skipped += 1
                            if lm is not None:
                                lm.log('Cannot get response : {}'.format(game_id))

                        response.close()

                    # get referee
                    params = {
                        'gameId': game_id
                    }

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                                      'like Gecko) Chrome/59.0.3071.115 Safari/537.36',
                        'X-Requested-With': 'XMLHttpRequest',
                        'Host': 'm.sports.naver.com',
                        'Referer': 'http://m.sports.naver.com/baseball/gamecenter/kbo/index.nhn?gameId='
                                   + game_id
                                   + '&tab=record'
                    }

                    response = requests.get(record_url, params=params, headers=headers)

                    p = regex.compile('(?<=\"etcRecords\":\[)[\\\.\{\}\"0-9:\s\(\)\,\ba-z가-힣\{\}]+')
                    result = p.findall(response.text)
                    if len(result) == 0:
                        txt['referee'] = ''
                    else:
                        txt['referee'] = result[0].split('{')[-1].split('":"')[1].split(' ')[0]

                    '''
                    p = regex.compile('stadiumName: \'\w+\'')
                    result = p.findall(response.text)
                    if len(result) == 0:
                        txt['stadium'] = ''
                    else:
                        txt['stadium'] = result[0].split('\'')[1]
                    '''

                    response.close()

                    fp = open(game_id + '_relay.json', 'w', newline='\n')
                    json.dump(txt, fp, ensure_ascii=False, sort_keys=False, indent=4)
                    fp.close()
                    
                    ##### 텍스트만 저장
                    text_list = []
                    pts_list = []
                    text_list_header = ["textOrder","textType","text","ptsPitchId","stuff","speed"]
                    pts_list_header = ["textOrder","inn","ballcount","crossPlateX","topSz","crossPlateY","pitchId","vy0","vz0","vx0","z0","y0","ax","x0","ay","az","bottomSz","stance"]
                    for k in sorted(txt['relayList'].keys()):
                        textset = txt['relayList'][k]
                        textOptionList = textset['textOptionList']
                        for to in textOptionList:
                            row = [k, to['type'], to['text']]
                            if 'ptsPitchId' in to.keys():
                                row.append(to['ptsPitchId'])
                            else:
                                row.append('')
                            if 'stuff' in to.keys():
                                row.append(to['stuff'])
                            else:
                                row.append('')
                            if 'speed' in to.keys():
                                row.append(to['speed'])
                            else:
                                row.append('')
                            text_list.append(row)
                        if 'ptsOptionList' in textset.keys():
                            ptsOptionList = textset['ptsOptionList']
                            for po in ptsOptionList:
                                row = [k] + list(po.values())
                                pts_list.append(row)

                    fp = open(game_id + '_textset.csv', 'w', newline='\n')
                    cf = csv.writer(fp)
                    cf.writerow(text_list_header)
                    for tl in text_list:
                        cf.writerow(tl)
                    fp.close()

                    fp = open(game_id + '_ptsset.csv', 'w', newline='\n')
                    cf = csv.writer(fp)
                    cf.writerow(pts_list_header)
                    for pl in pts_list:
                        cf.writerow(pl)
                    fp.close()
                    #####

                    done += 1
                else:
                    skipped += 1
                    if lm is not None:
                        lm.log('Cannot get response : {}'.format(game_id))

                print_progress('    Downloading: ', len(game_ids[year][month]), done, skipped)

            # download done
            print_progress('    Downloading: ', len(game_ids[year][month]), done, skipped)
            print('\n        Downloaded {} files'.format(done))
            print('        (Skipped {} files)'.format(skipped))
            end2 = time.time()
            print('            -- elapsed {:.3f} sec for month {}'.format(end2 - start2, month))

            os.chdir('..')
            # path: pbp_data/year
        end1 = time.time()
        print('   -- elapsed {:.3f} sec for year {}'.format(end1 - start1, year))
        # months done
        os.chdir('..')
        # path: pbp_data/
    # years done
    os.chdir('..')
    # path: root
    return True


def download_relay2(args, lm=None):
    # return True or False
    relay_url = 'http://m.sports.naver.com/ajax/baseball/gamecenter/kbo/relayText.nhn'
    record_url = 'http://m.sports.naver.com/ajax/baseball/gamecenter/kbo/record.nhn'

    now = datetime.datetime.now()
    today_year = now.year
    today_date = int(now.date().strftime('%m%d'))

    game_ids = get_game_ids(args)
    if (game_ids is None) or (len(game_ids) == 0):
        print('no game ids')
        print('args: {}'.format(args))
        if lm is not None:
            lm.log('no game ids')
            lm.log('args: {}'.format(args))
        return False

    if lm is not None:
        lm.resetLogHandler()
        lm.setLogPath(os.getcwd())
        lm.setLogFileName('relay_download_log.txt')
        lm.cleanLog()
        lm.createLogHandler()
        lm.log('---- Relay Text Download Log ----')

    if not os.path.isdir('pbp_data'):
        os.mkdir('pbp_data')
    os.chdir('pbp_data')
    # path: pbp_data

    print("##################################################")
    print("######        DOWNLOAD RELAY DATA          #######")
    print("##################################################")

    for year in game_ids.keys():
        start1 = time.time()
        print(" Year {}".format(year))
        if len(game_ids[year]) == 0:
            print('month id is empty')
            print('args: {}'.format(args))
            if lm is not None:
                lm.log('month id is empty')
                lm.log('args : {}'.format(args))
            os.chdir('../..')
            return False

        if not os.path.isdir(str(year)):
            os.mkdir(str(year))
        os.chdir(str(year))
        # path: pbp_data/year

        for month in game_ids[year].keys():
            start2 = time.time()
            print("  Month {}".format(month))
            if len(game_ids[year][month]) == 0:
                print('month id is empty')
                print('args: {}'.format(args))
                if lm is not None:
                    lm.log('month id is empty')
                    lm.log('args : {}'.format(args))
                os.chdir('../..')
                return False

            if not os.path.isdir(str(month)):
                os.mkdir(str(month))
            os.chdir(str(month))
            # path: pbp_data/year/month

            # download
            done = 0
            skipped = 0
            for game_id in game_ids[year][month]:
                game_id_year = int(game_id[:4])
                game_id_date = int(game_id[4:8])
                game_id_team = game_id[8:10]
                if (game_id_year < 2008) or (game_id_year > 7777):
                    skipped += 1
                    continue
                if (game_id_year == today_year) and (game_id_date > today_date):
                    skipped += 1
                    continue
                if game_id_date < int(regular_start[game_id[:4]]):
                    skipped += 1
                    continue
                if game_id_date >= int(playoff_start[game_id[:4]]):
                    skipped += 1
                    continue
                if game_id_team not in teams:
                    skipped += 1
                    continue

                relay_text_output_file = game_id + '_relay.csv'
                relay_batting_lineup_file = game_id + '_batting.csv'
                relay_pitching_lineup_file = game_id + '_pitching.csv'
                if (int(game_id[:4]) == today_year) &\
                   (int(game_id[4:6]) == now.month) &\
                   (int(game_id[6:8]) == now.day):
                       done = done
                elif (os.path.isfile(relay_text_output_file)) and \
                        (os.path.getsize(relay_text_output_file) > 0):
                    done += 1
                    if lm is not None:
                        lm.log('File Duplicate : {}'.format(game_id))
                    continue

                params = {
                    'gameId': game_id,
                    'half': '1'
                }

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/59.0.3071.115 Safari/537.36',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Host': 'm.sports.naver.com',
                    'Referer': 'http://m.sports.naver.com/baseball/gamecenter/kbo/index.nhn?&gameId='
                               + game_id
                               + '&tab=relay'
                }

                response = requests.get(relay_url, params=params, headers=headers)

                if (response is not None) & (response.status_code < 400):
                    txt = {}
                    js = response.json()
                    if isinstance(js, str):
                        js = json.loads(js)
                    last_inning = js['currentInning']

                    if last_inning is None:
                        skipped += 1
                        lm.log('Gameday not found : {}'.format(game_id))
                        continue

                    txt['relayList'] = {}
                    for i in range(len(js['relayList'])):
                        text_index = js['relayList'][i]['no']
                        txt['relayList'][text_index] = js['relayList'][i]
                        texts = txt['relayList'][text_index]['textOptionList']
                        for i in range(len(texts)):
                            texts[i]['text'].encode('cp949', 'ignore')
                    txt['homeTeamLineUp'] = js['homeTeamLineUp']
                    txt['awayTeamLineUp'] = js['awayTeamLineUp']

                    txt['stadium'] = js['schedule']['stadium']

                    response.close()

                    for inn in range(2, last_inning + 1):
                        params = {
                            'gameId': game_id,
                            'half': str(inn)
                        }

                        response = requests.get(relay_url, params=params, headers=headers)
                        if response is not None:
                            js = response.json()
                            if isinstance(js, str):
                                js = json.loads(js)
                            for i in range(len(js['relayList'])):
                                txt['relayList'][js['relayList'][i]['no']] = js['relayList'][i]
                                texts = txt['relayList'][js['relayList'][i]['no']]['textOptionList']
                                for i in range(len(texts)):
                                    texts[i]['text'].encode('cp949', 'ignore')
                        else:
                            skipped += 1
                            if lm is not None:
                                lm.log('Cannot get response : {}'.format(game_id))

                        response.close()

                    # get referee
                    params = {
                        'gameId': game_id
                    }

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                                      'like Gecko) Chrome/59.0.3071.115 Safari/537.36',
                        'X-Requested-With': 'XMLHttpRequest',
                        'Host': 'm.sports.naver.com',
                        'Referer': 'http://m.sports.naver.com/baseball/gamecenter/kbo/index.nhn?gameId='
                                   + game_id
                                   + '&tab=record'
                    }

                    response = requests.get(record_url, params=params, headers=headers)

                    p = regex.compile('(?<=\"etcRecords\":\[)[\\\.\{\}\"0-9:\s\(\)\,\ba-z가-힣\{\}]+')
                    result = p.findall(response.text)
                    if len(result) == 0:
                        txt['referee'] = ''
                    else:
                        txt['referee'] = result[0].split('{')[-1].split('":"')[1].split(' ')[0]

                    response.close()

                    ### 필요한 내용 담아서 저장 ###
                    rl = txt['relayList']

                    tl_keys = []
                    rl_keys = []
                    pts_keys = []
                    for k in rl.keys():
                        keys = rl.get(k).keys()
                        for key in keys:
                            if key in rl_keys:
                                continue
                            else:
                                rl_keys.append(key)

                        for j in range(len(rl.get(k).get('textOptionList'))):
                            keys = rl.get(k).get('textOptionList')[j].keys()
                            for key in keys:
                                if key in tl_keys:
                                    continue
                                else:
                                    tl_keys.append(key)
                        for j in range(len(rl.get(k).get('ptsOptionList'))):
                            keys = rl.get(k).get('ptsOptionList')[j].keys()
                            for key in keys:
                                if key in pts_keys:
                                    continue
                                else:
                                    pts_keys.append(key)

                    tl_keys_copy = tl_keys.copy()
                    if 'currentGameState' in tl_keys:
                        tl_keys_copy.remove('currentGameState')
                    if 'batterRecord' in tl_keys:
                        tl_keys_copy.remove('batterRecord')
                    if 'pitcherResult' in tl_keys:
                        tl_keys_copy.remove('pitcherResult')
                    if 'pitchResult' in tl_keys:
                        tl_keys_copy.remove('pitchResult')
                    if 'pitchNum' in tl_keys:
                        tl_keys_copy.remove('pitchNum')

                    ts_set = []
                    referee = txt['referee']
                    stadium = txt['stadium']
                    for k in rl.keys():
                        for j in range(len(rl.get(k).get('textOptionList'))):
                            ts = rl.get(k).get('textOptionList')[j]

                            ts_dict = {}
                            ts_dict['textOrder'] = int(k)
                            for key in tl_keys_copy:
                                if key == 'playerChange':
                                    if ts.get(key) is not None:
                                        for x in ['outPlayer', 'inPlayer', 'shiftPlayer']:
                                            if x in ts.get(key).keys():
                                                ts_dict[x] = ts.get(key).get(x).get('playerId')

                                else:
                                    ts_dict[key] = None if key not in ts.keys() else ts.get(key)
                            ts_dict['referee'] = referee
                            ts_dict['stadium'] = stadium
                            ts_set.append(ts_dict)
                    ts_df = pd.DataFrame(ts_set)
                    ts_df = ts_df.rename(index=str, columns={'ptsPitchId': 'pitchId'})

                    pdata_set = []
                    if len(pts_keys) > 0:
                        for k in rl.keys():
                            for j in range(len(rl.get(k).get('ptsOptionList'))):
                                pdata = rl.get(k).get('ptsOptionList')[j]

                                pdata_dict = {}
                                pdata_dict['textOrder'] = int(k)
                                for key in pts_keys:
                                    pdata_dict[key] = None if key not in pdata.keys() else pdata.get(key)
                                pdata_dict.pop('crossPlateY')
                                pdata_dict.pop('y0')
                                pdata_dict.pop('inn')
                                pdata_dict.pop('ballcount')
                                pdata_set.append(pdata_dict)

                        pdata_df = pd.DataFrame(pdata_set)
                        pdata_df.head()
                    else:
                        pdata_df = None

                    if pdata_df is not None:
                        merge_df = pd.merge(ts_df, pdata_df, how='outer').sort_values(['textOrder', 'seqno'])
                    else:
                        merge_df = ts_df.sort_values(['textOrder', 'seqno'])

                    #######################
                    ### 라인업 다운로드 ###
                    #######################
                    lineup_url = 'https://sports.news.naver.com/gameCenter/gameRecord.nhn?category=kbo&gameId='
                    lurl = lineup_url + game_id
                    lreq = requests.get(lurl)
                    lsoup = BeautifulSoup(lreq.text, 'lxml')
                    lreq.close()

                    scripts = lsoup.find_all('script')
                    team_names = lsoup.find_all('span', attrs={'class': 't_name_txt'})
                    away_team_name = team_names[0].contents[0].split(' ')[0]
                    home_team_name = team_names[1].contents[0].split(' ')[0]
                    contents = None

                    for tag in scripts:
                        if len(tag.contents) > 0:
                            if tag.contents[0].find('DataClass = ') > 0:
                                contents = tag.contents[0]
                                start = contents.find('DataClass = ') + 36
                                end = contents.find('_homeTeam')
                                oldjs = contents[start:end].strip()
                                while oldjs[-1] != '}':
                                    oldjs = oldjs[:-1]
                                while oldjs[0] != '{':
                                    oldjs = oldjs[1:]
                                cont = json.loads(oldjs)
                                break

                    bbs = cont.get('battersBoxscore')
                    al = bbs.get('away')
                    hl = bbs.get('home')

                    pos_dict = {'중': '중견수', '좌': '좌익수', '우': '우익수', '유': '유격수', '포': '포수', '지': '지명타자',
                                '一': '1루수', '二': '2루수', '三': '3루수'}

                    homes = []
                    aways = []
                    for i in range(len(hl)):
                        player = hl[i]
                        name = player.get('name')
                        pos = player.get('pos')[0]
                        pCode = player.get('playerCode')
                        homes.append({'name': name, 'pos': pos, 'pCode': pCode})

                    for i in range(len(al)):
                        player = al[i]
                        name = player.get('name')
                        pos = player.get('pos')[0]
                        pCode = player.get('playerCode')
                        aways.append({'name': name, 'pos': pos, 'pCode': pCode})

                    ### 라인업 가져다와서 더하기 ###
                    hit_columns = ['name', 'pCode', 'posName',
                                'hitType', 'seqno', 'batOrder']
                    pit_columns = ['name', 'pCode', 'hitType', 'seqno']

                    atl = txt.get('awayTeamLineUp')
                    abat = atl.get('batter')
                    apit = atl.get('pitcher')
                    abats = pd.DataFrame(abat, columns=hit_columns).sort_values(['batOrder', 'seqno'])
                    apits = pd.DataFrame(apit, columns=pit_columns).sort_values('seqno')

                    htl = txt.get('homeTeamLineUp')
                    hbat = htl.get('batter')
                    hpit = htl.get('pitcher')
                    hbats = pd.DataFrame(hbat, columns=hit_columns).sort_values(['batOrder', 'seqno'])
                    hpits = pd.DataFrame(hpit, columns=pit_columns).sort_values('seqno')

                    for a in aways:
                        if a.get('pos') == '교':
                            continue
                        abats.loc[(abats.name == a.get('name')) &
                                  (abats.pCode == a.get('pCode')), 'posName'] = pos_dict.get(a.get('pos'))

                    for h in homes:
                        if h.get('pos') == '교':
                            continue
                        hbats.loc[(hbats.name == h.get('name')) &
                                  (hbats.pCode == h.get('pCode')), 'posName'] = pos_dict.get(h.get('pos'))
                    abats['homeaway'] = 'a'
                    hbats['homeaway'] = 'h'
                    apits['homeaway'] = 'a'
                    hpits['homeaway'] = 'h'
                    abats['team_name'] = away_team_name
                    hbats['team_name'] = home_team_name
                    apits['team_name'] = away_team_name
                    hpits['team_name'] = home_team_name

                    bats = pd.concat([abats, hbats])
                    pits = pd.concat([apits, hpits])
                    bats.pCode = pd.to_numeric(bats.pCode)
                    pits.pCode = pd.to_numeric(pits.pCode)

                    ### 저장
                    if sys.platform == 'win32':
                        bats.to_csv(relay_batting_lineup_file, index=False, encoding='cp949')
                        pits.to_csv(relay_pitching_lineup_file, index=False, encoding='cp949')
                        merge_df.to_csv(relay_text_output_file, index=False, encoding='cp949')
                    else:
                        bats.to_csv(relay_batting_lineup_file, index=False)
                        pits.to_csv(relay_pitching_lineup_file, index=False)
                        merge_df.to_csv(relay_text_output_file, index=False)

                    done += 1
                else:
                    skipped += 1
                    if lm is not None:
                        lm.log('Cannot get response : {}'.format(game_id))

                print_progress('    Downloading: ', len(game_ids[year][month]), done, skipped)

            # download done
            print_progress('    Downloading: ', len(game_ids[year][month]), done, skipped)
            print('\n        Downloaded {} files'.format(done))
            print('        (Skipped {} files)'.format(skipped))
            end2 = time.time()
            print('            -- elapsed {:.3f} sec for month {}'.format(end2 - start2, month))

            os.chdir('..')
            # path: pbp_data/year
        end1 = time.time()
        print('   -- elapsed {:.3f} sec for year {}'.format(end1 - start1, year))
        # months done
        os.chdir('..')
        # path: pbp_data/
    # years done
    os.chdir('..')
    # path: root
    return True


def download_pitch_data_only(args, lm=None):
    # return True or False
    pdata_url = 'http://m.sports.naver.com/ajax/baseball/gamecenter/kbo/pitches.nhn'
    pdata_header_row = ['x0', 'y0', 'z0', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'plateX', 'plateZ', 'crossPlateX', 'crossPlateY', 'topSz', 'bottomSz', 'stuff', 'speed', 'pitcherName', 'batterName']

    game_ids = get_game_ids(args)
    if (game_ids is None) or (len(game_ids) == 0):
        print('no game ids')
        print('args: {}'.format(args))
        if lm is not None:
            lm.log('no game ids')
            lm.log('args: {}'.format(args))
        return False

    if lm is not None:
        lm.resetLogHandler()
        lm.setLogPath(os.getcwd())
        lm.setLogFileName('pitch_data_download_log.txt')
        lm.cleanLog()
        lm.createLogHandler()
        lm.log('---- Pitch Data Download Log ----')

    if not os.path.isdir('pbp_data'):
        os.mkdir('pbp_data')
    os.chdir('pbp_data')
    # path: pbp_data

    print("##################################################")
    print("######         DOWNLOAD PITCH DATA         #######")
    print("##################################################")

    for year in game_ids.keys():
        start1 = time.time()
        print(" Year {}".format(year))
        if len(game_ids[year]) == 0:
            print('month id is empty')
            print('args: {}'.format(args))
            if lm is not None:
                lm.log('month id is empty')
                lm.log('args : {}'.format(args))
            return False

        if not os.path.isdir(str(year)):
            os.mkdir(str(year))
        os.chdir(str(year))
        # path: pbp_data/year
        
        year_fp = open(f'{year}_pdata.csv', 'w', newline='\n')
        year_cf = csv.writer(year_fp)
        year_cf.writerow(pdata_header_row)

        for month in game_ids[year].keys():
            start2 = time.time()
            print("  Month {}".format(month))
            if len(game_ids[year][month]) == 0:
                print('month id is empty')
                print('args: {}'.format(args))
                if lm is not None:
                    lm.log('month id is empty')
                    lm.log('args : {}'.format(args))
                return False

            if not os.path.isdir(str(month)):
                os.mkdir(str(month))
            os.chdir(str(month))
            # path: pbp_data/year/month

            month_fp = open(f'{year}_{month}_pdata.csv', 'w', newline='\n')
            month_cf = csv.writer(month_fp)
            month_cf.writerow(pdata_header_row)

            # download
            done = 0
            skipped = 0
            for game_id in game_ids[year][month]:
                if (int(game_id[:4]) < 2008) or (int(game_id[:4]) > datetime.datetime.now().year):
                    skipped += 1
                    continue
                if (int(game_id[:4]) == datetime.datetime.now().year) and (int(game_id[4:8]) > int(datetime.datetime.now().date().strftime('%m%d'))):
                    skipped += 1
                    continue
                if int(game_id[4:8]) < int(regular_start[game_id[:4]]):
                    skipped += 1
                    continue
                if int(game_id[4:8]) >= int(playoff_start[game_id[:4]]):
                    skipped += 1
                    continue
                if game_id[8:10] not in teams:
                    skipped += 1
                    continue

                if not check_url(pdata_url):
                    skipped += 1
                    if lm is not None:
                        lm.log('URL error : {}'.format(pdata_url))
                    continue

                if (int(game_id[:4]) == datetime.datetime.now().year) &\
                   (int(game_id[4:6]) == datetime.datetime.now().month) &\
                   (int(game_id[6:8]) == datetime.datetime.now().day):
                        # do nothing
                       done = done
                elif (os.path.isfile(game_id + '_pdata.json')) and \
                        (os.path.getsize(game_id + '_pdata.json') > 0):
                    done += 1
                    if lm is not None:
                        lm.log('File Duplicate : {}'.format(game_id))
                    continue

                params = {
                    'gameId': game_id
                }

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/59.0.3071.115 Safari/537.36',
                    'X-Requested-With': 'XMLHttpRequest',
                    'Host': 'm.sports.naver.com',
                    'Referer': 'http://m.sports.naver.com/baseball/gamecenter/kbo/index.nhn?&gameId='
                               + game_id
                               + '&tab=relay'
                }

                response = requests.get(pdata_url, params=params, headers=headers)

                if response is not None:
                    # load json structure
                    js = response.json()
                    if isinstance(js, str):
                        js = json.loads(js)
                        #js = ast.literal_eval(js)

                    if js is None:
                        lm.log('Pitch data missing : {}'.format(game_id))
                        skipped += 1
                        continue
                    elif len(js) == 0:
                        lm.log('Pitch data missing : {}'.format(game_id))
                        skipped += 1
                        continue

                    # json to pandas dataframe
                    #df = pd.read_json(json.dumps(js))
                    df = pd.DataFrame(js)

                    # calculate pitch location(px, pz)
                    t = -df['vy0'] - np.sqrt(df['vy0'] * df['vy0'] - 2 * df['ay'] * (df['y0'] - df['crossPlateY']))
                    t /= df['ay']
                    xp = df['x0'] + df['vx0'] * t + df['ax'] * t * t * 0.5
                    zp = df['z0'] + df['vz0'] * t + df['az'] * t * t * 0.5
                    df['plateX'] = np.round(xp, 5)
                    df['plateZ'] = np.round(zp, 5)

                    # calculate pitch movement(pfx_x, pfx_z)
                    t40 = -df['vy0'] - np.sqrt(df['vy0'] * df['vy0'] - 2 * df['ay'] * (df['y0'] - 40))
                    t40 /= df['ay']
                    x40 = df['x0'] + df['vx0'] * t40 + 0.5 * df['ax'] * t40 * t40
                    vx40 = df['vx0'] + df['ax'] * t40
                    z40 = df['z0'] + df['vz0'] * t40 + 0.5 * df['az'] * t40 * t40
                    vz40 = df['vz0'] + df['az'] * t40
                    th = t - t40
                    x_no_air = x40 + vx40 * th
                    z_no_air = z40 + vz40 * th - 0.5 * 32.174 * th * th
                    df['pfx_x'] = np.round((xp - x_no_air) * 12, 5)
                    df['pfx_z'] = np.round((zp - z_no_air) * 12, 5)

                    # load back to json structure
                    dfjsstr = df.to_json(orient='records', force_ascii=False)
                    dfjs = json.loads(dfjsstr)

                    # dump to json file
                    fp = open(game_id + '_pdata.json', 'w', newline='\n')
                    json.dump(dfjs, fp, ensure_ascii=False, sort_keys=False, indent=4)
                    fp.close()

                    # dump to csv file
                    fp = open(game_id + '_pdata.csv', 'w', newline='\n')
                    cf = csv.writer(fp)
                    cf.writerow(pdata_header_row)

                    for x in dfjs:
                        row = [x['x0'],
                               x['y0'],
                               x['z0'],
                               x['vx0'],
                               x['vy0'],
                               x['vz0'],
                               x['ax'],
                               x['ay'],
                               x['az'],
                               x['plateX'],
                               x['plateZ'],
                               x['crossPlateX'],
                               x['crossPlateY'],
                               x['topSz'],
                               x['bottomSz'],
                               x['stuff'],
                               x['speed'],
                               x['pitcherName'],
                               x['batterName']]
                        month_cf.writerow(row)
                        year_cf.writerow(row)
                        cf.writerow(row)

                    fp.close()

                    done += 1
                else:
                    skipped += 1
                    if lm is not None:
                        lm.log('Cannot get response : {}'.format(game_id))

                print_progress('    Downloading: ', len(game_ids[year][month]), done, skipped)

            # download done
            print_progress('    Downloading: ', len(game_ids[year][month]), done, skipped)
            print('\n        Downloaded {} files'.format(done))
            print('        (Skipped {} files)'.format(skipped))
            end2 = time.time()
            print('            -- elapsed {:.3f} sec for month {}'.format(end2 - start2, month))
            month_fp.close()

            os.chdir('..')
            # path: pbp_data/year
        end1 = time.time()
        print('   -- elapsed {:.3f} sec for year {}'.format(end1 - start1, year))
        # months done
        year_fp.close()
        
        os.chdir('..')

        # path: pbp_data/
    # years done
    os.chdir('..')
    # path: root
    return True


if __name__ == '__main__':
    args = []  # m_start, m_end, y_start, y_end
    options = []  # onlyConvert, onlyDownload
    get_args(args, options)

    if options[1] is True:
        relaylm = logManager.LogManager()
        rc = download_relay(args, relaylm)
        if rc is False:
            print('Error')
            exit(1)
        relaylm.killLogManager()

    if options[2] is True:
        pbplm = logManager.LogManager()
        rc = download_pitch_data_only(args, pbplm)
        if rc is False:
            print('Error')
            exit(1)
        pbplm.killLogManager()
