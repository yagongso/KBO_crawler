# pbp.py

import argparse, traceback, sys, pathlib, datetime
import time

from download import download_pbp_files
from game_parse import header_row

def getTracebackStr():
    lines = traceback.format_exc().strip().split('\n')
    lines.reverse()
    rl = []
    for i in range(len(lines)):
        rl.append(lines[i].strip())
    return '\n'.join(rl)


def join_csvs(path, start_date, end_date):
    csvs = list(path.glob('*/*csv'))
    if len(csvs) < 1:
        print('no csv file found')
    else:
        years = list(set([str(filename.stem)[:4] for filename in csvs]))

        years_given = list(range(start_date.year, end_date.year+1))
        
        enc = 'cp949' if sys.platform == 'win32' else 'utf-8'
        for y in years:
            if int(y) not in years_given:
                continue
            yfilepath = str(path / f'{y}.csv')
            yfile = open(yfilepath, 'w', encoding=enc)
            
            yearfiles = [x for x in csvs if (x.stem.find(str(y)) > -1) & (len(x.stem) > 4)]

            header = ','.join(header_row) + '\n'
            yfile.write(header)
            
            for file in yearfiles:
                fp = file.open()
                lines = fp.readlines()
                if len(lines) < 2:
                    fp.close()
                    continue
                for line in lines[1:]:
                    yfile.write(line)
                fp.close()
                
            yfile.close()            


parser = argparse.ArgumentParser()

now = datetime.datetime.now().date()
default_start_date = now.replace(month=1,day=1).strftime('%Y%m%d')
default_end_date = now.replace(month=12,day=31).strftime('%Y%m%d')

parser.add_argument('-f', '--from',
                    type=str,
                    dest='start_date',
                    metavar='START_DATE',
                    default=default_start_date,
                    help="Game Date >= YYYYMMDD(ex:20201231); "
                         "default: YYYY0101")

parser.add_argument('-t', '--to',
                    type=str,
                    dest='end_date',
                    metavar='END_DATE',
                    default=default_end_date,
                    help="Game Date >= YYYYMMDD(ex:20201231); "
                         "default: YYYY1231")

parser.add_argument('-s', '--savedir',
                    type=str,
                    dest='save_path',
                    metavar='SAVE_FILE_PATH',
                    default=None,
                    help="save path; default: CURRENT_PATH/save")

parser.add_argument('-p', '--playoff', '--postseason',
                    action='store_true',
                    dest='playoff',
                    help="get playoff files")

parser.add_argument('-g', '--debug',
                    action='store_true',
                    dest='debug_mode',
                    help='print debug log')

parser.add_argument('-j', '--join',
                    action='store_true',
                    dest='join_csv',
                    help='join output files with same year')

parser.add_argument('--save-source',
                    action='store_true',
                    dest='save_source',
                    help='save source data too')


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    try:
        if len(args.start_date) != 8:
            assert False
        else:
            int(args.start_date)
    except (ValueError, AssertionError):
        print('ERROR: -f')
        print('\tcheck -f argument(GAME DATE >=) string format')
        print('\tshould give 8 integer(YYYYMMDD)')
        exit(1)
        
    try:
        if len(args.end_date) != 8:
            assert False
        else:
            int(args.end_date)
    except (ValueError, AssertionError):
        print('ERROR: -t')
        print('\tcheck -t argument(GAME DATE <=) string format')
        print('\tshould give 8 integer(YYYYMMDD)')
        exit(1)
    
    try:        
        start_date = datetime.date(int(args.start_date[:4]),
                                   int(args.start_date[4:6]),
                                   int(args.start_date[6:]))
        end_date = datetime.date(int(args.end_date[:4]),
                                 int(args.end_date[4:6]),
                                 int(args.end_date[6:]))
    except ValueError:
        print('date value error : month or day out of range')        
        exit(1)
    
    save_path = args.save_path
    sp = None
    if args.save_path is not None:
        sp = pathlib.Path(args.save_path)
        if sp.exists() & ~(sp.is_dir()):
            print('ERROR: -s')
            print(f"\tsave path('{args.save_path}') exists, but not directory(folder)")
            exit(1)
        elif not sp.exists():
            sp.mkdir()
    else:
        save_path = 'save'
        suffix = 1
        sp = pathlib.Path(save_path)
        while True:
            sp = pathlib.Path(save_path)
            if sp.exists() & sp.is_dir():
                break
            try:
                sp.mkdir()
                print(f"NOTE: save path '{sp.stem}' created")
            except FileExistsError:
                save_path = 'save' + str(suffix)
                suffix += 1
                continue
            break
    
    try:
        download_pbp_files(start_date, end_date, playoff=args.playoff,
                           save_path=sp, debug_mode=args.debug_mode,
                           save_source=args.save_source)
    except:
        print('Closed during download')
        log = open('log.txt', 'a')
        log.write('====================================\n')
        log.write(getTracebackStr())
        print(getTracebackStr())
        log.close()
        exit(1)
        
    if args.join_csv is True:
        try:
            join_csvs(sp, start_date, end_date)
        except:
            print('ERROR: -j')
            exit(1)
