# pbp.py
# Play-By-Play 데이터를 JSON 형식으로 다운로드받고, 파싱해서 CSV 파일로 변환&저장한다.
# get batted ball data in JSON form & convert to CSV files

from utils import get_args
from pbp_parse import parse_main
from pbp_download import *
import logManager


def run_pbp_download(args, lm=None):
    download_relay(args, lm)


def run_pbp_parser(args, lm=None):
    parse_main(args, lm)


def run_pitch_data_only_download(args, lm=None):
    download_pitch_data_only(args, lm)


if __name__ == "__main__":
    args = []  # m_start, m_end, y_start, y_end
    options = []  # onlyConvert, onlyDownload, onlyPitchDataDownload
    parser = get_args(args, options)

    # option : -c, -d, -p
    if (options[0] is True) & (options[1] is False) & (options[2] is False):
        relay_lm = logManager.LogManager()  # background logger
        run_pbp_parser(args, relay_lm)
        relay_lm.killLogManager()
    elif (options[0] is False) & (options[1] is True) & (options[2] is False):
        download_lm = logManager.LogManager()  # background logger
        run_pbp_download(args, download_lm)
        download_lm.killLogManager()
    elif (options[0] is False) & (options[1] is False) & (options[2] is True):
        pbp_lm = logManager.LogManager()  # background logger
        run_pitch_data_only_download(args, pbp_lm)
        pbp_lm.killLogManager()
    elif (options[0] is False) & (options[1] is False) & (options[2] is False):
        print('choose at least one option!\n')
        parser.print_help()
    else:
        print('choose one option at once!\n')
        parser.print_help()
