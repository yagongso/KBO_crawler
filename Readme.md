﻿# 개요
[Naver](https://www.naver.com)의 문자 중계 시스템에서 Play-by-play(PBP) 데이터를 가져와 정리하여 csv 파일로 출력한다.

# 필요한 것들
- [Python 3.5 이상](https://www.python.org/downloads/).
  - [Anaconda](https://www.anaconda.com/download/)를 사용한 설치를 추천
  - 필요 모듈 : `lxml`, `BeautifulSoup`, `regex`, `numpy`, `matplotlib`, `pandas`, `IPython`
  - 모듈 설치는 `pip`를 사용 (설명은 인터넷 검색 참조)
- 터미널
  - 윈도우에선 Git Bash나 MINGW 추천
- Git
  - 검색해서 설치하자

# 실행법
- 여기 있는걸 `git clone` 등으로 다운로드
- 기본 사용법 : `python pbp.py -f 시작일 -t 종료일`
  - `pbp.py`가 있는 경로에 `save` 폴더가 생성되면서 그 안에 파일이 저장됨
- `python pbp.py -h` 치면 간략한 도움말 출력

# 옵션
  - `-f`, `--from` : 시작일 `START_DATE` 설정
  - `-t`, `--to`: 종료일 `END_DATE` 설정
    - `START_DATE` <= GAME DATE <= `FROM_DATE` 기간의 경기 데이터를 다운로드
    - 포맷은 YYYYMMDD (연도 월 일)
  - `-p`, `--playoff`, `--postseason` : 이 옵션을 주면 포스트시즌 경기도 저장함
  - `-s`, `--savedir` : 저장경로 설정, 기본은 `현재경로\save`
    - 아래에 연도 별로 폴더 생성돼서 차곡차곡 저장
  - `-j`, `--join` : 이 옵션을 주면 연도별로 묶은 파일이 생성됨(ex: 2019.csv)
  - `-g` : 이 옵션을 주면 디버그용 메시지 출력

