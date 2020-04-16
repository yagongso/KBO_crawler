# bb.py
# 타구 데이터 JSON 형식으로 받아서 CSV 파일로 변환
# get batted ball data in JSON form & convert to CSV files

from utils import get_args
from bb_download import bb_download
from bb_convert_to_csv import bb_convert_to_csv
import logManager


def run_bb_download(arg, lm=None):
    bb_download(arg[0], arg[1], arg[2], arg[3], lm)


def run_bb_convert_to_csv(arg, lm=None):
    bb_convert_to_csv(arg[0], arg[1], arg[2], arg[3], lm)


if __name__ == "__main__":
    args = []       # m_start, m_end, y_start, y_end
    options = []    # onlyConvert, onlyDownload
    parser = get_args(args, options)

    # option : -c, -d, -p
    if (options[0] is True) & (options[1] is False) & (options[2] is False):
        lm = logManager.LogManager()
        run_bb_convert_to_csv(args, lm)
        lm.killLogManager()
    elif (options[0] is False) & (options[1] is True) & (options[2] is False):
        lm = logManager.LogManager()
        run_bb_download(args, lm)
        lm.killLogManager()
    elif (options[0] is False) & (options[1] is False) & (options[2] is True):
        print('pfx option is not supported for bb.py')
        parser.print_help()
    elif (options[0] is False) & (options[1] is False) & (options[2] is False):
        print('choose at least one option!\n')
        parser.print_help()
    else:
        print('choose one option at once!\n')
        parser.print_help()
