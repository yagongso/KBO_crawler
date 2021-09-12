import pandas as pd
import sys, time, requests, json, datetime, pathlib, warnings
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm import tqdm, trange
from bs4 import BeautifulSoup

from game_parse import game_status

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
    '2020': '0505',
    '2021': '0403',
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
    '2020': '1101',
    '2021': '1101',
}


def get_game_ids(start_date, end_date, playoff=False):
    """
    KBO 경기 ID를 가져온다.

    Parameters
    -----------
    start_date, end_date : datetime.date
        ID를 가져올 경기 기간의 시작일과 종료일.
        start_date <= Game Date of Games <= end_date

    playoff : bool, default False
        True일 경우 플레이오프(포스트시즌) 경기 ID도 받는다.
    """

    timetable_url = 'https://sports.news.naver.com/'\
                    'kbaseball/schedule/index.nhn?month='

    mon1 = start_date.replace(day=1)
    r = []
    while mon1 <= end_date:
        r.append(mon1)
        mon1 += relativedelta(months=1)

    game_ids = []

    for d in r:
        month = d.month
        year = d.year

        year_regular_start = regular_start[str(year)]
        year_playoff_start = playoff_start[str(year)]
        year_regular_start_date = datetime.date(year,
                                                int(year_regular_start[:2]),
                                                int(year_regular_start[2:]))
        year_playoff_start_date = datetime.date(year,
                                                int(year_playoff_start[:2]),
                                                int(year_playoff_start[2:]))
        year_last_date = datetime.date(year, 12, 31)
        sch_url = timetable_url + f'{month}&year={year}'

        response = requests.get(sch_url)
        soup = BeautifulSoup(response.text, 'lxml')
        response.close()

        sch_tbs = soup.findAll('div', attrs={'class': 'sch_tb'})
        sch_tb2s = soup.findAll('div', attrs={'class': 'sch_tb2'})
        sch_tbs += sch_tb2s

        for row in sch_tbs:
            day_table = row.findAll('tr')
            for game in day_table:
                add_state = game.findAll('td', attrs={'class': 'add_state'})
                tds = game.findAll('td')
                if len(tds) < 4:
                    continue
                links = game.findAll('span',
                                    attrs={'class': 'td_btn'})
                date_td = game.findAll('span', attrs={'class': 'td_date'})
                if len(date_td) > 0:
                    date_text = date_td[0].text
                if len(add_state) > 0:
                    status = add_state[0].findAll('span')[-1].get('class')[0]
                    if status == 'suspended':
                        continue

                for btn in links:
                    gid = btn.a['href'].split('/')[2]
                    try:
                        gid_date = datetime.date(int(gid[:4]),
                                                 int(gid[4:6]),
                                                 int(gid[6:8]))
                    except:
                        continue
                    if start_date <= gid_date <= end_date:
                        if playoff == False:
                            if year_regular_start_date <= gid_date < year_playoff_start_date:
                                game_ids.append(gid)
                        else:
                            if year_regular_start_date <= gid_date < year_last_date:
                                game_ids.append(gid)
    return game_ids


def get_game_data(game_id):
    """
    KBO 경기 PBP 데이터를 가져온다.

    Parameters
    -----------
    game_id : str
        가져올 게임 ID.
    """

    relay_url = 'http://m.sports.naver.com/ajax/baseball/'\
            'gamecenter/kbo/relayText.nhn'
    record_url = 'http://m.sports.naver.com/ajax/baseball/'\
                'gamecenter/kbo/record.nhn'
    params = {'gameId': game_id, 'half': '1'}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/59.0.3071.115 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'Host': 'm.sports.naver.com',
        'Referer': 'http://m.sports.naver.com/baseball'\
                   '/gamecenter/kbo/index.nhn?&gameId='
                   + game_id
                   + '&tab=relay'
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #####################################
        # 1. pitch by pitch 데이터 가져오기 #
        #####################################
        relay_response = requests.get(relay_url,
                                      params=params,
                                      headers=headers)
        if relay_response.status_code > 200:
            relay_response.close()
            return [None, None, 'response error\n']

        relay_json = relay_response.json()
        js = None
        try:
            js = json.loads(relay_json)
            relay_response.close()
        except JSONDecodeError:
            relay_response.close()
            return [None, None, 'got no valid data\n']

        if js.get('gameId') is None:
            return [None, None, 'invalid game ID\n']

        last_inning = js['currentInning']

        if last_inning is None:
            return [None, None, 'no last inning\n']

        game_data_set = {}
        game_data_set['relayList'] = []
        for x in js['relayList']:
            game_data_set['relayList'].append(x)

        # 라인업에 대한 기초 정보가 담겨 있음
        game_data_set['homeTeamLineUp'] = js['homeTeamLineUp']
        game_data_set['awayTeamLineUp'] = js['awayTeamLineUp']

        game_data_set['stadium'] = js['schedule']['stadium']

        for inn in range(2, last_inning + 1):
            params = {
                'gameId': game_id,
                'half': str(inn)
            }

            relay_inn_response = requests.get(relay_url, params=params, headers=headers)
            if relay_inn_response.status_code > 200:
                relay_inn_response.close()
                return [None, None, 'response error\n']

            relay_json = relay_inn_response.json()
            try:
                js = json.loads(relay_json)
                relay_response.close()
            except JSONDecodeError:
                relay_inn_response.close()
                return [None, None, 'got no valid data\n']

            for x in js['relayList']:
                game_data_set['relayList'].append(x)

        #########################
        # 2. 가져온 정보 다듬기 #
        #########################
        relay_list = game_data_set['relayList']
        text_keys = ['seqno', 'text', 'type', 'stuff',
                     'ptsPitchId', 'speed', 'playerChange']
        pitch_keys = ['crossPlateX', 'topSz',
                      'pitchId', 'vy0', 'vz0', 'vx0',
                      'z0', 'ax', 'x0', 'ay', 'az',
                      'bottomSz']

        # pitch by pitch 텍스트 데이터 취합
        text_set = []
        stadium = game_data_set['stadium']
        for k in range(len(relay_list)):
            for j in range(len(relay_list[k].get('textOptionList'))):
                text_row = relay_list[k].get('textOptionList')[j]

                text_row_dict = {}
                text_row_dict['textOrder'] = relay_list[k].get('no')
                for key in text_keys:
                    if key == 'playerChange':
                        if text_row.get(key) is not None:
                            for x in ['outPlayer', 'inPlayer', 'shiftPlayer']:
                                if x in text_row.get(key).keys():
                                    text_row_dict[x] = text_row.get(key).get(x).get('playerId')
                    else:
                        text_row_dict[key] = None if key not in text_row.keys() else text_row.get(key)
                # text_row_dict['referee'] = referee
                text_row_dict['stadium'] = stadium
                text_set.append(text_row_dict)
        text_set_df = pd.DataFrame(text_set)
        text_set_df = text_set_df.rename(index=str, columns={'ptsPitchId': 'pitchId'})
        text_set_df.seqno = pd.to_numeric(text_set_df.seqno)

        # pitch by pitch 트래킹 데이터 취합
        pitch_data_set = []
        pitch_data_df = None
        for k in range(len(relay_list)):
            if relay_list[k].get('ptsOptionList') is not None:
                for j in range(len(relay_list[k].get('ptsOptionList'))):
                    pitch_data = relay_list[k].get('ptsOptionList')[j]

                    pitch_data_dict = {}
                    pitch_data_dict['textOrder'] = relay_list[k].get('no')
                    for key in pitch_keys:
                        pitch_data_dict[key] = None if key not in pitch_data.keys() else pitch_data.get(key)
                    pitch_data_set.append(pitch_data_dict)

        # 텍스트(중계) 데이터, 트래킹 데이터 취합
        if len(pitch_data_set) > 0:
            pitch_data_df = pd.DataFrame(pitch_data_set)
            relay_df = pd.merge(text_set_df, pitch_data_df, how='outer').sort_values(['textOrder', 'seqno'])
        else:
            relay_df = text_set_df.sort_values(['textOrder', 'seqno'])

        ##################################################
        # 3. 선발 라인업, 포지션, 레퍼리 데이터 가져오기 #
        ##################################################
        lineup_url = 'https://sports.news.naver.com/gameCenter'\
                     '/gameRecord.nhn?category=kbo&gameId='
        lineup_response = requests.get(lineup_url + game_id)

        if lineup_response.status_code > 200:
            lineup_response.close()
            return [None, None, 'response error\n']

        lineup_soup = BeautifulSoup(lineup_response.text, 'lxml')
        lineup_response.close()

        scripts = lineup_soup.find_all('script')
        if scripts[10].contents[0].find('잘못된') > 0:
            return [None, None, 'false script page\n']

        team_names = lineup_soup.find_all('span', attrs={'class': 't_name_txt'})
        away_team_name = team_names[0].contents[0].split(' ')[0]
        home_team_name = team_names[1].contents[0].split(' ')[0]

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
                    try:
                        cont = json.loads(oldjs)
                    except JSONDecodeError:
                        return [None, None, f'JSONDecodeError - gameID {game_id}\n']
                    break

        # 구심 정보 가져와서 취합
        referee = cont.get('etcRecords')[-1]['result'].split(' ')[0]
        relay_df = relay_df.assign(referee = referee)

        # 경기 끝나고 나오는 박스스코어, 홈/어웨이 라인업
        boxscore = cont.get('battersBoxscore')
        away_lineup = boxscore.get('away')
        home_lineup = boxscore.get('home')

        pos_dict = {'중': '중견수', '좌': '좌익수', '우': '우익수',
                    '유': '유격수', '포': '포수', '지': '지명타자',
                    '一': '1루수', '二': '2루수', '三': '3루수'}

        home_players = []
        away_players = []

        for i in range(len(home_lineup)):
            player = home_lineup[i]
            name = player.get('name')
            pos = player.get('pos')[0]
            pCode = player.get('playerCode')
            home_players.append({'name': name, 'pos': pos, 'pCode': pCode})

        for i in range(len(away_lineup)):
            player = away_lineup[i]
            name = player.get('name')
            pos = player.get('pos')[0]
            pCode = player.get('playerCode')
            away_players.append({'name': name, 'pos': pos, 'pCode': pCode})


        ##############################
        # 4. 기존 라인업 정보와 취합 #
        ##############################
        hit_columns = ['name', 'pCode', 'posName',
                       'hitType', 'seqno', 'batOrder']
        pit_columns = ['name', 'pCode', 'hitType', 'seqno']

        atl = game_data_set.get('awayTeamLineUp')
        abat = atl.get('batter')
        apit = atl.get('pitcher')
        abats = pd.DataFrame(abat, columns=hit_columns).sort_values(['batOrder', 'seqno'])
        apits = pd.DataFrame(apit, columns=pit_columns).sort_values('seqno')

        htl = game_data_set.get('homeTeamLineUp')
        hbat = htl.get('batter')
        hpit = htl.get('pitcher')
        hbats = pd.DataFrame(hbat, columns=hit_columns).sort_values(['batOrder', 'seqno'])
        hpits = pd.DataFrame(hpit, columns=pit_columns).sort_values('seqno')

        #####################################
        # 5. 라인업 정보 보강             #
        #####################################
        record_response = requests.get(record_url,
                                      params=params,
                                      headers=headers)
        if record_response.status_code > 200:
            record_response.close()
            return [None, None, 'response error\n']

        record_json = record_response.json()
        record_response.close()

        apr = pd.DataFrame(record_json['awayPitcher'])
        hpr = pd.DataFrame(record_json['homePitcher'])
        abr = pd.DataFrame(record_json['awayBatter'])
        hbr = pd.DataFrame(record_json['homeBatter'])
        apr = apr.rename(index=str, columns={'pcode':'pCode'})
        hpr = hpr.rename(index=str, columns={'pcode':'pCode'})
        abr = abr.rename(index=str, columns={'pcode':'pCode'})
        hbr = hbr.rename(index=str, columns={'pcode':'pCode'})

        apr.loc[:, 'seqno'] = 10
        apr.loc[:, 'hitType'] = None
        hpr.loc[:, 'seqno'] = 10
        hpr.loc[:, 'hitType'] = None

        abr.loc[:, 'seqno'] = 10
        abr.loc[:, 'hitType'] = None
        abr.loc[:, 'posName'] = None
        hbr.loc[:, 'seqno'] = 10
        hbr.loc[:, 'hitType'] = None
        hbr.loc[:, 'posName'] = None

        for p in apr.pCode.unique():
            if p in apits.pCode.unique():
                apr.loc[(apr.pCode == p), 'seqno'] = int(apits.loc[apits.pCode == p].seqno.values[0])
                apr.loc[(apr.pCode == p), 'hitType'] = apits.loc[apits.pCode == p].hitType.values[0]
            else:
                apr.loc[(apr.pCode == p), 'seqno'] = 10
        for p in hpr.pCode.unique():
            if p in hpits.pCode.unique():
                hpr.loc[(hpr.pCode == p), 'seqno'] = int(hpits.loc[hpits.pCode == p].seqno.values[0])
                hpr.loc[(hpr.pCode == p), 'hitType'] = hpits.loc[hpits.pCode == p].hitType.values[0]
            else:
                hpr.loc[(hpr.pCode == p), 'seqno'] = 10
        for p in abats.pCode.unique():
            if p in abats.pCode.unique():
                abr.loc[(abr.pCode == p), 'seqno'] = int(abats.loc[abats.pCode == p].seqno.values[0])
                abr.loc[(abr.pCode == p), 'posName'] = abats.loc[abats.pCode == p].posName.values[0]
                abr.loc[(abr.pCode == p), 'hitType'] = abats.loc[abats.pCode == p].hitType.values[0]
            else:
                abr.loc[(abr.pCode == p), 'seqno'] = 10
        for p in hbats.pCode.unique():
            if p in hbats.pCode.unique():
                hbr.loc[(hbr.pCode == p), 'seqno'] = int(hbats.loc[hbats.pCode == p].seqno.values[0])
                hbr.loc[(hbr.pCode == p), 'posName'] = hbats.loc[hbats.pCode == p].posName.values[0]
                hbr.loc[(hbr.pCode == p), 'hitType'] = hbats.loc[hbats.pCode == p].hitType.values[0]
            else:
                hbr.loc[(hbr.pCode == p), 'seqno'] = 10

        apr = apr.astype({'seqno': int})
        hpr = hpr.astype({'seqno': int})
        abr = abr.astype({'seqno': int})
        hbr = hbr.astype({'seqno': int})

        apits = apr[pit_columns]
        hpits = hpr[pit_columns]
        abats = abr[hit_columns]
        hbats = hbr[hit_columns]

        # 선발 출장한 경우, 선수의 포지션을 경기 시작할 때 포지션으로 수정
        # (pitch by pitch 데이터에서 가져온 정보는 경기 종료 시의 포지션임)
        for player in away_players:
            # '교'로 적혀있는 교체 선수는 넘어간다
            if player.get('pos') == '교':
                continue
            abats.loc[(abats.name == player.get('name')) &
                      (abats.pCode == player.get('pCode')), 'posName'] = pos_dict.get(player.get('pos'))
            if len(player.get('name')) > 3:
                pname = player.get('name')
                for i in range(len(abats)):
                    if abats.iloc[i].values[0].find(pname) > -1:
                        pCode = abats.iloc[i].pCode
                        abats.loc[(abats.pCode == pCode), 'posName'] = pos_dict.get(player.get('pos'))
                        break

        for player in home_players:
            # '교'로 적혀있는 교체 선수는 넘어간다
            if player.get('pos') == '교':
                continue
            hbats.loc[(hbats.name == player.get('name')) &
                      (hbats.pCode == player.get('pCode')), 'posName'] = pos_dict.get(player.get('pos'))
            if len(player.get('name')) > 3:
                pname = player.get('name')
                for i in range(len(hbats)):
                    if hbats.iloc[i].values[0].find(pname) > -1:
                        pCode = hbats.iloc[i].pCode
                        hbats.loc[(hbats.pCode == pCode), 'posName'] = pos_dict.get(player.get('pos'))
                        break

        abats = abats.assign(homeaway = 'a', team_name = away_team_name)
        hbats = hbats.assign(homeaway = 'h', team_name = home_team_name)
        apits = apits.assign(homeaway = 'a', team_name = away_team_name)
        hpits = hpits.assign(homeaway = 'h', team_name = home_team_name)

        batting_df = pd.concat([abats, hbats])
        pitching_df = pd.concat([apits, hpits])
        batting_df.pCode = pd.to_numeric(batting_df.pCode)
        pitching_df.pCode = pd.to_numeric(pitching_df.pCode)

        return pitching_df, batting_df, relay_df


def get_game_data_renewed(game_id):
    nav_api_header = 'https://api-gw.sports.naver.com/schedule/games/'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        #####################################
        # 0. 게임 메타 데이터 가져오기      #
        #####################################
        game_req = requests.get(nav_api_header + game_id)
        if game_req.status_code > 200:
            game_req.close()
            return [None, None, 'meta data request error\n']
        game_req_result = game_req.json()
        if game_req_result.get('code') > 200:
            game_req.close()
            return [None, None, 'meta data request error\n']
        game_req.close()

        game_meta_data = game_req_result.get('result').get('game')
        stadium = game_meta_data.get('stadium')
        homeTeamCode = game_meta_data.get('homeTeamCode')
        homeTeamName = game_meta_data.get('homeTeamName')
        awayTeamCode = game_meta_data.get('awayTeamCode')
        awayTeamName = game_meta_data.get('awayTeamName')
        if game_meta_data.get('currentInning') is not None:
            max_inning = int(game_meta_data.get('currentInning').split('회')[0])
        else:
            max_inning = int(game_meta_data.get('statusInfo').split('회')[0])

        box_score_req = requests.get(f'{nav_api_header}{game_id}/record')
        if box_score_req.status_code > 200:
            box_score_req.close()
            return [None, None, 'meta data(box score) request error\n']
        box_score_req_result = box_score_req.json()
        if box_score_req_result.get('code') > 200:
            box_score_req.close()
            return [None, None, 'meta data(box score) request error\n']
        box_score_req.close()

        box_score_data = box_score_req_result.get('result').get('recordData')
        if len(box_score_data.get('etcRecords')) > 0:
            referees = box_score_data.get('etcRecords')[-1].get('result').split(' ')
        else:
            print(game_id)
            referees = ['']
        away_batting_order = box_score_data.get('battersBoxscore').get('away')
        home_batting_order = box_score_data.get('battersBoxscore').get('home')
        away_pitchers = box_score_data.get('pitchersBoxscore').get('away')
        home_pitchers = box_score_data.get('pitchersBoxscore').get('home')

        #####################################
        # 1. pitch by pitch 데이터 가져오기 #
        #####################################
        game_data_set = {}
        game_data_set['pitchTextList'] = []
        game_data_set['pitchTrackDataList'] = []

        text_keys = ['seqno', 'text', 'type', 'stuff',
                     'ptsPitchId', 'speed', 'playerChange']
        pitch_keys = ['crossPlateX', 'topSz',
                      'pitchId', 'vy0', 'vz0', 'vx0',
                      'z0', 'ax', 'x0', 'ay', 'az',
                      'bottomSz']
        for inning in range(1, max_inning+1):
            pbp_req = requests.get(f'{nav_api_header}{game_id}/relay?inning={inning}')
            if pbp_req.status_code > 200:
                pbp_req.close()
                print([None, None, 'pbp relay data request error\n'])
                assert False
            pbp_req_result = pbp_req.json()
            if pbp_req_result.get('code') > 200:
                pbp_req.close()
                print([None, None, 'pbp relay data request error\n'])
                assert False
            pbp_req.close()

            pbp_data = pbp_req_result.get('result').get('textRelayData')

            for textSetList in pbp_data.get('textRelays')[::-1]:
                textRow = {}
                pitchTrackerRow = {}

                textSet = textSetList.get('textOptions')
                textSetNo = textSetList.get('no')
                for pitchTextData in textSet:
                    textRow = {}
                    textRow['textOrder'] = textSetNo
                    for key in text_keys:
                        if key == 'playerChange':
                            if pitchTextData.get(key) is not None:
                                for x in ['outPlayer', 'inPlayer', 'shiftPlayer']:
                                    if x in pitchTextData.get(key).keys():
                                        textRow[x] = pitchTextData.get(key).get(x).get('playerId')
                        else:
                            if key not in pitchTextData.keys():
                                textRow[key] = None
                            else:
                                textRow[key] = pitchTextData.get(key)
                    textRow['referee'] = referees[0]
                    textRow['stadium'] = stadium
                    game_data_set['pitchTextList'].append(textRow)

                pitchTrackerSet = textSetList.get('ptsOptions')
                for pitchTrackData in pitchTrackerSet:
                    pitchTrackerRow = {}
                    pitchTrackerRow['textOrder'] = textSetNo

                    for key in pitch_keys:
                        if key not in pitchTrackData.keys():
                            pitchTrackerRow[key] = None
                        else:
                            pitchTrackerRow[key] = pitchTrackData.get(key)

                    game_data_set['pitchTrackDataList'].append(pitchTrackData)

        text_set_df = pd.DataFrame(game_data_set['pitchTextList'])
        text_set_df = text_set_df.rename(index=str, columns={'ptsPitchId': 'pitchId'})
        text_set_df.seqno = pd.to_numeric(text_set_df.seqno)

        # 텍스트(중계) 데이터, 트래킹 데이터 취합
        if len(game_data_set['pitchTrackDataList']) > 0:
            pitch_data_df = pd.DataFrame(game_data_set['pitchTrackDataList'])
            relay_df = pd.merge(text_set_df, pitch_data_df, how='outer').sort_values(['textOrder', 'seqno'])
        else:
            relay_df = text_set_df.sort_values(['textOrder', 'seqno'])

        ########################################
        # 2. 라인업 정리                       #
        ########################################
        # 라인업에 대한 기초 정보가 담겨 있음
        # 경기 끝나고나서 최종 정보 -> 포지션은 마지막 상황
        game_data_set['awayLineup'] = pbp_data.get('awayLineup')
        game_data_set['homeLineup'] = pbp_data.get('homeLineup')

        game_data_set['stadium'] = stadium

        pos_dict = {'중': '중견수', '좌': '좌익수', '우': '우익수',
            '유': '유격수', '포': '포수', '지': '지명타자',
            '一': '1루수', '二': '2루수', '三': '3루수'}

        home_players = []
        away_players = []

        for i in range(len(home_batting_order)):
            player = home_batting_order[i]
            name = player.get('name')
            pos = player.get('pos')[0]
            pcode = player.get('playerCode')
            home_players.append({'name': name, 'pos': pos, 'pcode': pcode})

        for i in range(len(away_batting_order)):
            player = away_batting_order[i]
            name = player.get('name')
            pos = player.get('pos')[0]
            pcode = player.get('playerCode')
            away_players.append({'name': name, 'pos': pos, 'pcode': pcode})

        ############################################
        # 3. 메타 데이터에 있는 라인업 정보와 취합 #
        ############################################
        # 메타 데이터에는 경기 시작했을 때 포지션 정보가 있음
        hit_columns = ['name', 'pcode', 'posName',
                       'hitType', 'seqno', 'batOrder']
        pit_columns = ['name', 'pcode', 'hitType', 'seqno']

        away_lineup_meta_data = game_data_set.get('awayLineup')
        away_batters = away_lineup_meta_data.get('batter')
        away_pitchers = away_lineup_meta_data.get('pitcher')
        away_lineup_df = pd.DataFrame(away_batters, columns=hit_columns).sort_values(['batOrder', 'seqno'])
        away_pitcher_df = pd.DataFrame(away_pitchers, columns=pit_columns).sort_values('seqno')

        home_lineup_meta_data = game_data_set.get('homeLineup')
        home_batters = home_lineup_meta_data.get('batter')
        home_pitchers = home_lineup_meta_data.get('pitcher')
        home_lineup_df = pd.DataFrame(home_batters, columns=hit_columns).sort_values(['batOrder', 'seqno'])
        home_pitcher_df = pd.DataFrame(home_pitchers, columns=pit_columns).sort_values('seqno')

        away_lineup_df = away_lineup_df.assign(pcode = pd.to_numeric(away_lineup_df.pcode))
        away_pitcher_df = away_pitcher_df.assign(pcode = pd.to_numeric(away_pitcher_df.pcode))
        home_lineup_df = home_lineup_df.assign(pcode = pd.to_numeric(home_lineup_df.pcode))
        home_pitcher_df = home_pitcher_df.assign(pcode = pd.to_numeric(home_pitcher_df.pcode))

        ap = pd.DataFrame(away_players)
        ap = ap.assign(pcode = pd.to_numeric(ap.pcode))

        hp = pd.DataFrame(home_players)
        hp = hp.assign(pcode = pd.to_numeric(hp.pcode))
        away_lineup_df = pd.merge(away_lineup_df, ap, on='pcode', how='outer')
        home_lineup_df = pd.merge(home_lineup_df, hp, on='pcode', how='outer')

        # 선발 출장한 경우, 선수의 포지션을 경기 시작할 때 포지션으로 수정
        # (pitch by pitch 데이터에서 가져온 정보는 경기 종료 시의 포지션임)
        away_lineup_df = away_lineup_df.assign(name = np.where(away_lineup_df.name_x.isnull(),
                                                               away_lineup_df.name_y,
                                                               away_lineup_df.name_x))
        home_lineup_df = home_lineup_df.assign(name = np.where(home_lineup_df.name_x.isnull(),
                                                               home_lineup_df.name_y,
                                                               home_lineup_df.name_x))
        lineup_df_columns = ['name', 'pcode', 'posName', 'hitType', 'seqno', 'batOrder', 'pos']
        away_lineup_df = away_lineup_df[lineup_df_columns]
        home_lineup_df = home_lineup_df[lineup_df_columns]

        away_lineup_df = away_lineup_df\
                        .assign(posName = np.where(away_lineup_df.pos != '교',
                                                   away_lineup_df.pos\
                                                       .apply(lambda x: pos_dict.get(x)),
                                                   away_lineup_df.posName))
        home_lineup_df = home_lineup_df\
                        .assign(posName = np.where(home_lineup_df.pos != '교',
                                                   home_lineup_df.pos\
                                                       .apply(lambda x: pos_dict.get(x)),
                                                   home_lineup_df.posName))

        away_lineup_df = away_lineup_df.assign(homeaway = 'a', team_name = awayTeamName)
        home_lineup_df = home_lineup_df.assign(homeaway = 'h', team_name = homeTeamName)
        away_pitcher_df = away_pitcher_df.assign(homeaway = 'a', team_name = awayTeamName)
        home_pitcher_df = home_pitcher_df.assign(homeaway = 'h', team_name = homeTeamName)

        batting_df = pd.concat([away_lineup_df, home_lineup_df])
        pitching_df = pd.concat([away_pitcher_df, home_pitcher_df])
        batting_df.pcode = pd.to_numeric(batting_df.pcode)
        pitching_df.pcode = pd.to_numeric(pitching_df.pcode)
    return pitching_df, batting_df, relay_df


def download_pbp_files(start_date, end_date, playoff=False,
                       save_path=None, debug_mode=False,
                       save_source=False):
    """
    KBO 피치 바이 피치(PBP) 파일을 다운로드.

    Parameters
    -----------
    start_date, end_date : datetime.date
        PBP 파일을 받을 경기 기간의 시작일과 종료일.
        start_date <= Game Date of Downloaded Files <= end_date

    playoff : bool, default False
        True일 경우 플레이오프(포스트시즌) 경기 파일도 받는다.

    save_path : pathlib.Path, default None
        PBP 파일을 저장할 경로.
        값이 없을 경우(None) 현재 경로에 저장.

    debug_mode : bool, default False
        True일 경우 sys.stdout을 통해 디버그 메시지와 수행 시간이 출력됨.

    save_source : bool, default False
        True일 경우 parsing 이전의 소스 데이터를 csv 형식으로 저장.
    """
    start_time = time.time()
    game_ids = get_game_ids(start_date, end_date, playoff)
    end_time = time.time()
    get_game_id_time = end_time - start_time

    enc = 'cp949' if sys.platform == 'win32' else 'utf-8'

    now = datetime.datetime.now()

    logfile = open('./log.txt', 'a', encoding=enc)
    logfile.write('\n\n')
    logfile.write('====================================\n')
    logfile.write(f"Current Time : {now.isoformat()}\n")
    logfile.write('====================================\n')

    skipped = 0
    broken = 0
    done = 0
    start_time = time.time()
    get_data_time = 0
    gid = None

    years = list(set([x[:4] for x in game_ids]))

    try:
        for y in years:
            y_path = save_path / y

            if not y_path.is_dir():
                try:
                    y_path.mkdir()
                except FileExistsError:
                    logfile.write(f'ERROR : path {y_path} exists, but not a directory')
                    logfile.write(f'\tclean path and try again')
                    print(f'ERROR : path {y_path} exists, but not a directory')
                    print(f'\tclean path and try again')
                    exit(1)

        for gid in tqdm(game_ids):
            now = datetime.datetime.now().date()
            gid_to_date = datetime.date(int(gid[:4]),
                                        int(gid[4:6]),
                                        int(gid[6:8]))
            if gid_to_date > now:
                continue

            if (save_path / gid[:4] / f'{gid}.csv').exists():
                skipped += 1
                continue

            ptime = time.time()
            source_path = save_path / gid[:4] / 'source'
            if (source_path / f'{gid}_pitching.csv').exists() &\
                (source_path / f'{gid}_batting.csv').exists() &\
                (source_path / f'{gid}_relay.csv').exists():
                game_data_dfs = []
                game_data_dfs.append(pd.read_csv(str(source_path / f'{gid}_pitching.csv'), encoding=enc))
                game_data_dfs.append(pd.read_csv(str(source_path / f'{gid}_batting.csv'), encoding=enc))
                game_data_dfs.append(pd.read_csv(str(source_path / f'{gid}_relay.csv'), encoding=enc))
            else:
                game_data_dfs = get_game_data_renewed(gid)

            if game_data_dfs[0] is None:
                logfile.write(game_data_dfs[-1])
                if debug_mode == True:
                    print(game_data_dfs[-1])
                exit(1)

            if save_source == True:
                if not source_path.is_dir():
                    try:
                        source_path.mkdir()
                    except FileExistsError:
                        source_path = save_path / gid[:4]
                        logfile.write(f'NOTE: {gid[:4]}/source exists but not a directory.')
                        logfile.write(f'source files will be saved in {gid[:4]} instead.')

                if not (source_path / f'{gid}_pitching.csv').exists():
                    game_data_dfs[0].to_csv(str(source_path / f'{gid}_pitching.csv'),
                                            index=False, encoding=enc, errors='replace')
                if not (source_path / f'{gid}_batting.csv').exists():
                    game_data_dfs[1].to_csv(str(source_path / f'{gid}_batting.csv'),
                                            index=False, encoding=enc, errors='replace')
                if not (source_path / f'{gid}_relay.csv').exists():
                    game_data_dfs[2].to_csv(str(source_path / f'{gid}_relay.csv'),
                                            index=False, encoding=enc, errors='replace')

            get_data_time += time.time() - ptime
            if game_data_dfs is not None:
                gs = game_status()
                gs.load(gid, game_data_dfs[0], game_data_dfs[1], game_data_dfs[2], log_file=logfile)
                parse = gs.parse_game(debug_mode)
                gs.save_game(save_path / gid[:4])
                if parse == True:
                    done += 1
                else:
                    broken += 1
            else:
                broken += 1
                continue

        end_time = time.time()
        parse_time = end_time - start_time - get_data_time
        logfile.write('====================================\n')
        logfile.write(f'Start date : {start_date.strftime("%Y%m%d")}\n')
        logfile.write(f'End date : {end_date.strftime("%Y%m%d")}\n')
        logfile.write(f'Successfully downloaded games : {done}\n')
        logfile.write(f'Skipped games(already exists) : {skipped}\n')
        logfile.write(f'Broken games(bad data) : {broken}\n')
        logfile.write('====================================\n')
        if debug_mode == True:
            logfile.write(f'Elapsed {get_game_id_time:.2f} sec in get_game_ids\n')
            logfile.write(f'Elapsed {(get_data_time):.2f} sec in get_game_data\n')
            logfile.write(f'Elapsed {(parse_time):.2f} sec in parse_game\n')
        logfile.write(f'Total {(parse_time+get_game_id_time+get_data_time):.2f} sec elapsed with {len(game_ids)} games\n')

        if logfile.closed is not True:
            logfile.close()
    except:
        logfile.write(f'=== gameID : {gid}\n')
        if logfile.closed is not True:
            logfile.close()
        assert False

