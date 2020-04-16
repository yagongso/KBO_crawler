# bb_download.py
# (1) get all match text relay broadcast URL; all match in YEAR/MONTH.
# (2) open URL and parse.

import json
from urllib.request import urlopen
import os
from bs4 import BeautifulSoup
import re
import sys
from logManager import getTracebackStr

regular_start = {
    '2012': '0407',
    '2013': '0330',
    '2014': '0329',
    '2015': '0328',
    '2016': '0401',
    '2017': '0331'
}

playoff_start = {
    '2012': '1008',
    '2013': '1008',
    '2014': '1019',
    '2015': '1010',
    '2016': '1021',
    '2017': '1010'
}


def print_progress(bar_prefix, mon_file_num, done, skipped):
    if mon_file_num > 30:
        progress_pct = (float(done + skipped) / float(mon_file_num))
        bar = '+' * int(progress_pct * 30) + '-' * (30 - int(progress_pct * 30))
        print('\r{}[{}] {} / {}, {:2.1f} %'.format(bar_prefix, bar, (done + skipped), mon_file_num,
                                                   progress_pct * 100), end="")
    elif mon_file_num > 0:
        bar = '+' * (done + skipped) + '-' * (mon_file_num - done - skipped)
        print('\r{}[{}] {} / {}, {:2.1f} %'.format(bar_prefix, bar, (done + skipped), mon_file_num,
                                                   float(done + skipped) / float(mon_file_num) * 100),
              end="")


def bb_download(mon_start, mon_end, year_start, year_end, lm=None):
    # set url prefix
    schedule_url_prefix = "http://sports.news.naver.com/kbaseball/schedule/index.nhn?"
    result_url_prefix = "http://sports.news.naver.com/gameCenter/gameResult.nhn?category=kbo&gameId="

    # make directory
    if not os.path.isdir("./bb_data"):
        os.mkdir("./bb_data")

    for year in range(year_start, year_end + 1):
        # make year directory
        if not os.path.isdir("./bb_data/{0}".format(str(year))):
            os.mkdir("./bb_data/{0}".format(str(year)))

        for month in range(mon_start, mon_end + 1):
            if month < 10:
                mon = '0{}'.format(str(month))
            else:
                mon = str(month)
            if not os.path.isdir("./bb_data/{0}/{1}".format(str(year), mon)):
                os.mkdir("./bb_data/{0}/{1}".format(str(year), mon))

    os.chdir("./bb_data")
    # current path : ./bb_data/

    print("##################################################")
    print("######        DOWNLOAD BB DATA             #######")
    print("##################################################")

    for year in range(year_start, year_end + 1):
        print("  for Year {0}... ".format(str(year)))

        for month in range(mon_start, mon_end + 1):
            if month < 10:
                mon = '0{}'.format(str(month))
            else:
                mon = str(month)

            os.chdir("{0}/{1}".format(str(year), mon))
            # current path : ./bb_data/YEAR/MONTH/

            # get URL
            schedule_url = "{0}month={1}&year={2}".format(schedule_url_prefix, str(month), str(year))

            # open URL
            print("    Month {0}... ".format(str(month)))
            # get relay URL list, write in text file
            # all GAME RESULTS in month/year
            bar_prefix = '    Downloading: '
            print('\r{}[waiting]'.format(bar_prefix), end="")

            # '경기결과' 버튼을 찾아서 태그를 모두 리스트에 저장.
            schedule_html = urlopen(schedule_url).read()
            schedule_soup = BeautifulSoup(schedule_html, 'lxml')
            schedule_button = schedule_soup.findAll('span', attrs={'class': 'td_btn'})

            # '경기결과' 버튼을 찾아서 태그를 모두 리스트에 저장.
            game_ids = []
            for btn in schedule_button:
                link = btn.a['href']
                suffix = link.split('gameId=')[1]
                game_ids.append(suffix)

            mon_file_num = sum(1 for game_id in game_ids if int(game_id[:4]) <= 2050)

            # gameID가 있는 게임은 모두 경기 결과가 있는 것으로 판단함
            done = 0
            skipped = 0

            lm.resetLogHandler()
            lm.setLogPath(os.getcwd() + '/log/')
            lm.setLogFileName('bbDownloadLog.txt')
            lm.cleanLog()
            lm.createLogHandler()

            for game_id in game_ids:
                if int(game_id[0:4]) < 2010:
                    continue
                if int(game_id[0:4]) > 2050:
                    continue

                if int(regular_start[game_id[:4]]) > int(game_id[4:8]):
                    skipped += 1
                    continue

                if int(playoff_start[game_id[:4]]) <= int(game_id[4:8]):
                    skipped += 1
                    continue

                bb_data_filename = game_id[:13] + '_bb.json'

                if os.path.isfile(bb_data_filename) and (os.path.getsize(bb_data_filename) > 0):
                    done += 1
                    continue

                result_html = urlopen(result_url_prefix + game_id)

                soup = BeautifulSoup(result_html.read(), 'lxml')
                script = soup.find('script', text=re.compile('ChartDataClass'))
                if script is None:
                    skipped += 1
                    continue
                try:
                    json_text = re.search(r'({"teamsInfo":{.*?}}}})', script.string, flags=re.DOTALL).group(1)
                except AttributeError:
                    print()
                    print('JSON parse error in : {}'.format(game_id))
                    print(getTracebackStr())
                    lm.bugLog('JSON parse error in : {}'.format(game_id))
                    lm.bugLog(getTracebackStr())
                    lm.killLogManager()
                    exit(1)

                if sys.platform == 'win32':
                    try:
                        data = json.loads(json_text, encoding='iso-8859-1')
                    except json.decoder.JSONDecodeError:
                        json_text = re.search(r'({"teamsInfo":{.*?}}}},.*?}}}})', script.string, flags=re.DOTALL).group(1)
                        data = json.loads(json_text, encoding='iso-8859-1')
                else:
                    try:
                        data = json.loads(json_text)
                    except json.decoder.JSONDecodeError:
                        json_text = re.search(r'({"teamsInfo":{.*?}}}},.*?}}}})', script.string, flags=re.DOTALL).group(1)
                        data = json.loads(json_text)

                # BUGBUG : 20170509 KT-HT 경기 타구 정보 실종
                # 생략
                if game_id[:8] != '20170509':
                    with open(bb_data_filename, 'w', encoding='utf-8') as bb_data_file:
                        json.dump(data, bb_data_file, indent=4, ensure_ascii=False)
                    bb_data_file.close()

                done += 1
                lm.log('{} download'.format(game_id))

                print_progress(bar_prefix, mon_file_num, done, skipped)
            print_progress(bar_prefix, mon_file_num, done, skipped)
            print()
            print('        Downloaded {0} files.'.format(str(done)))
            print('        (Skipped {0} files)'.format(str(skipped)))
            os.chdir('../..')
            # current path : ./bb_data/
    os.chdir('..')
    # current path : ./
    print("DOWNLOAD BB DATA DONE.")
