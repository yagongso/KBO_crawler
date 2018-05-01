# 개요
[Naver](https://www.naver.com)의 문자 중계 시스템에서 Play-by-play(PBP) 데이터를 가져와 정리하여 csv 파일로 출력한다.

# 요구 사항
- [Python 3.5 이상](https://www.python.org/downloads/).
  - [Anaconda](https://www.anaconda.com/download/)를 사용한 설치를 추천한다. __설치 경로 이름에는 가급적 한글을 빼자__!
  - 필요 모듈 : `lxml`, `BeautifulSoup`, `regex`, `Numpy`, `Matplotlib`, `Pandas`, `IPython`
  - [Anaconda](https://www.anaconda.com/download/) 설치시 대부분의 모듈이 모두 자동으로 설치된다.
  - `regex`는 추가 설치가 필요한데, Anaconda 설치 후 Anaconda Navigator를 통해 설치하거나 Git Bash(아래 설명 참조), 커맨드 프롬프트 등에서 `conda install regex` 또는 `pip install regex`를 입력하면 설치할 수 있다.
- OS: Windows, OS X, Linux
- 인터넷 연결 필요
- 커맨드라인 인터페이스(CLI)에서 실행을 권장한다. 기타 환경에서 실행한 적 없음.

# PBP 크롤러 무작정 따라하기
여기 올려져있는 파일을 다운로드받고 실행하려면 평소에 안 쓰던 프로그램을 조금 써야 한다.

따라하기만 하면 길어도 10~20분 정도면 충분하니까 인내심을 갖도록 하자.

윈도우 환경에서 Anaconda로 파이썬(Python)을 미리 설치했다고 가정하고 진행하겠다.

## 1. Git, Git Bash 설치
[Git(다운로드 링크)](https://gitforwindows.org/)을 다운로드 받는다.

설치 방법은 [이 링크](http://dev-gabriel.tistory.com/21)를 참조.

굉장히 자세히 설명되어 있지만 그냥 막 다음-다음-다음만 선택해도 설치된다.

![git_for_windows_site](https://i.imgur.com/IIH9JEX.png "git_for_windows_site")

버전이 많은데 어지간해선 64bit, 최신 exe 파일을 받아서 실행하면 된다.
![git_for_windows_download](https://i.imgur.com/MsGrzqI.png "git_for_windows_download")


## 2. Git Bash 실행
잘 다음-다음-다음을 선택해서 Git을 설치하면 `Git Bash`라는 프로그램이 같이 설치됐을 것이다. 실행해주자.

처음 Git Bash를 실행하면 까만 화면이 뜬다. 여기서 당황하지 말고, 지금 이 문서가 있는 github 홈페이지로 돌아가보자.

![github_main_page](https://i.imgur.com/SUkCNhX.png "github_main_page")

오른쪽 위를 보면  __Clone or download__ 라고 써있는 초록색 버튼이 있다.

클릭하면 무슨 인터넷 사이트 주소 같은 게 나오는데 __통채로 잘 복사해준다__.

다시 Git Bash 프로그램으로 돌아가서 `cd Documents`라고 쓰고 엔터를 누르자.
- `cd Documents`는 `Documents`라는 경로로 이동하라는 뜻이다. 처음 시작 위치 밑에 존재하는 `Documents`는 `내 문서`다.

그 다음 `git clone [아까 복사한 주소]`를 입력하고 엔터를 쳐주면...

성공적으로 PBP 크롤러 등 파일을 복사해오게 된다.

윈도우 탐색기에서 `내 문서`로 들어가도 복사된게 보인다.


## 3. PBP 데이터 다운로드

다시 Git Bash로 돌아가 방금 경로로 들어간다. Git Bash를 껐다가 다시 실행했다면, `cd Documents/KBO_crawler`를 치고 엔터를 누르면 된다.
- git 사용법에 능숙해 다른 경로에 다운로드를 받았다면 해당 경로로 들어가도록 하자.

그리고 다음과 같이 입력하고 엔터를 친다. 2017년 3월 경기의 PBP 데이터를 다운로드하라는 뜻이다. 

`python pfx.py -d 2017 3`

- `Permission Denied`라는 에러가 나면서 실행이 안된다면, Git Bash를 관리자 권한으로 실행해보자. Git Bash 아이콘에 우클릭하면 아마도 '관리자 권한으로 실행하기'라는 메뉴가 보일 것이다.

- 윈도우에서 실행시 `bash: python: command not found`라는 에러가 난다면, 환경변수 추가가 필요하다. 윈도우 7 사용자는 [여기](http://bitboom.tistory.com/entry/Python-%EC%84%A4%EC%B9%98-%EB%B0%8F-%ED%99%98%EA%B2%BD%EB%B3%80%EC%88%98-%EC%84%A4%EC%A0%95)를, 윈도우 8 이상 사용자는 [여기](http://radiation.tistory.com/entry/%ED%99%98%EA%B2%BD%EB%B3%80%EC%88%98%EC%97%90-Python-%EC%B6%94%EA%B0%80%ED%95%98%EA%B8%B0)를 참조하여 환경변수를 설정하고 Git bash를 다시 실행해보자.

명령어 각각을 풀어서 설명하면 다음과 같다.

- `python pfx.py` : 파이썬으로 `pfx.py`를 실행하라는 뜻이다. `pfx.py`는 PBP 데이터를 읽어오고 해석하는 파이썬 파일이다.
- `-d` : PBP 데이터를 다운로드하라는 옵션을 지정한다. `-d` 밖에도 `-c`, `-p` 옵션이 존재한다. `-h`를 쓰면 도움말(usage)이 출력된다.
- `2017 3` : 다운로드 받을 데이터의 기간을 지정한다. 2017년 3월의 데이터를 다운로드하라는 뜻. 연도, 월을 최대 2개씩 지정할 수 있으며 이럴 경우 자동으로 연/월로 구분해 시작/종료 기간을 설정한다. 예를 들어 `2017 2018 3 10`을 입력하면 2017년 3월부터 2018년 10월까지의 데이터를 다운로드한다. 

1달에 90~100경기 정도가 있고 이를 다운로드하는데는 1분 정도가 걸린다. 3월 경기 수는 적어서 이보다 더 적다.

다운로드가 끝나면 `pbp_data`라는 폴더가 생성되고 그 아래에 연도/월 별로 `JSON` 확장자를 갖는 파일이 다운로드된다.


## 4. PBP 데이터 CSV 파일로 변환

JSON파일은 사람의 눈으로 직접 살펴보기 좋지 않다. 엑셀 등으로 쉽게 볼 수 있는 CSV 파일로 변환해보자.

이번에는 다음과 같이 입력해보자.

`python pfx.py -c 2017 3`

2017년 3월 경기의 PBP 데이터를 CSV파일로 변환하라는 뜻이다(JSON 파일이 다운로드되어있어야 한다).

정상적으로 진행된다면 JSON파일이 있는 경로에 CSV파일들이 생성될 것이다. 월별, 연도별 폴더에도 각각 월별/연도별로 정리된 CSV 파일이 생성된다.

연도별 파일은 위에서 지정한 월에 해당하는 데이터만 저장된다. 위와 같은 경우 `2017.csv`에는 3월 데이터만 저장된다. 시즌 전체 데이터를 합친 연도 파일을 생성하고 싶다면 `python pfx.py -c 2017` 이렇게 실행하도록 하자.

여기까지 왔으면 성공이다. ㅊㅋㅊㅋ
