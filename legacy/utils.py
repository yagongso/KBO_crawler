#utils.py

import datetime
import argparse
import http.client
from urllib.parse import urlparse
from urllib.request import urlopen
import time
import requests


def get_args(output, options):
    # convert arguments
    # check if not number, integer

    parser = argparse.ArgumentParser(description='Get pitch by pitch data.')
    parser.add_argument('dates',
                        metavar='dates',
                        type=int,
                        nargs='*',
                        default=datetime.datetime.now().year,
                        help='start/end (month/year); year > 2007')

    parser.add_argument('-c',
                        action='store_true',
                        help='convert pitch data to .csv format')

    parser.add_argument('-d',
                        action='store_true',
                        help='Download pitch data')

    parser.add_argument('-p',
                        action='store_true',
                        help='Download pitch f/x data only')

    args = parser.parse_args()

    options.append(args.c)
    options.append(args.d)
    options.append(args.p)

    if (args.c and args.p) or (args.c and args.d) or (args.d and args.p):
        print('choose one option at once!\n')
        parser.print_help()
        exit(1)

    dates = args.dates
    now = datetime.datetime.now()

    if type(dates) is int:
        # month or year?
        if dates > 12:
            # year
            if (dates < 2008) or (dates > now.year):
                print('invalid year')
                exit(1)
            else:
                year = dates
                if year == now.year:
                    # current season
                    if now.month < 3:
                        print('invalid year : season has not begun...')
                        exit(1)
                    else:
                        # dates = [now.year, now.year, 3, now.month]
                        output.append(3)
                        output.append(now.month)
                else:
                    # previous season
                    # dates = [year, year, 3, 10]
                    output.append(3)
                    output.append(10)
                output.append(year)
                output.append(year)
        elif dates > 0:
            # month
            if (dates < 3) or (dates > 10):
                print('invalid month : possible range is 3~10')
                exit(1)
            else:
                month = dates
                if month <= now.month:
                    # dates = [now.year, now.year, month, month]
                    output.append(month)
                    output.append(month)
                    output.append(now.year)
                    output.append(now.year)
                else:
                    # trying for future...
                    print('invalid month : current month is {}; you entered{}.'.format(now.month, month))
                    exit(1)
        else:
            print('invalid parameter')
            exit(1)
    elif len(dates) > 4:
        print('too many date option')
        exit(1)
    else:
        months = []
        years = []

        for d in dates:
            if (d > 12) & (d > 2007) & (d <= now.year):
                years.append(d)
            elif (d >= 1) & (d <= 12):
                months.append(d)
            else:
                print('invalid date')
                print('possible year range: 2008~%d'%(now.year))
                print('possible month range: 1~12')
                exit(1)

            if len(years) > 2:
                print('too many year')
                exit(1)

            if len(months) > 2:
                print('too many month')
                exit(1)

        mmin = 3
        mmax = 3
        ymin = now.year
        ymax = now.year

        if len(months) == 0:
            mmin = 3
            mmax = 10
        elif len(months) == 1:
            mmin = mmax = months[0]
        else:
            mmin = min(months)
            mmax = max(months)

        if len(years) == 0:
            ymin = now.year
            ymax = now.year
        elif len(years) == 1:
            ymin = ymax = years[0]
        else:
            ymin = min(years)
            ymax = max(years)

        output.append(mmin)
        output.append(mmax)
        output.append(ymin)
        output.append(ymax)

    return parser



def print_progress(bar_prefix, total, done, skipped):
    if total > 30:
        progress_pct = (float(done + skipped) / float(total))
        bar = '+' * int(progress_pct * 30) + '-' * (30 - int(progress_pct * 30))
        print('\r{}[{}] {} / {}, {:2.1f} %'.format(bar_prefix, bar, (done + skipped), total,
                                                   progress_pct * 100), end="")
    elif total > 0:
        bar = '+' * (done + skipped) + '-' * (total - done - skipped)
        print('\r{}[{}] {} / {}, {:2.1f} %'.format(bar_prefix, bar, (done + skipped), total,
                                                   float(done + skipped) / float(total) * 100),
              end="")


def check_url(url):
    p = urlparse(url)
    conn = http.client.HTTPConnection(p.netloc)
    conn.request('HEAD', p.path)
    resp = conn.getresponse()
    return resp.status < 400


def check_url2(url):
    resp = requests.get(url)
    status = resp.status_code < 400
    resp.close()
    return status


def retry_urlopen(url, num_of_retries=10, time_interval=2):
    page = None
    for _ in range(num_of_retries):
        try:
            page = urlopen(url, timeout=10)
            return page
            break
        except:
            time.sleep(time_interval)
    else:
        raise

