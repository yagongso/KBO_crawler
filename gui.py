# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import traceback, sys, pathlib, datetime, time
import pandas as pd
from download import get_game_ids, get_game_data
from game_parse import game_status, header_row

def getTracebackStr():
    lines = traceback.format_exc().strip().split('\n')
    lines.reverse()
    rl = []
    for i in range(len(lines)):
        rl.append(lines[i].strip())
    return '\n'.join(rl)

now = datetime.datetime.now()

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
            yearfiles.sort(reverse=False)

            header_written = False

            for f in yearfiles:
                fp = f.open(encoding=enc)
                lines = fp.readlines()
                if len(lines) < 2:
                    fp.close()
                    continue
                if header_written == False:
                    yfile.write(lines[0])
                    header_written = True
                for line in lines[1:]:
                    yfile.write(line)
                fp.close()
                
            yfile.close()

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setEnabled(True)
        Dialog.resize(500, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        Dialog.setMinimumSize(QtCore.QSize(500, 300))
        Dialog.setMaximumSize(QtCore.QSize(500, 300))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        Dialog.setFont(font)
        self.startDateEdit = QtWidgets.QDateEdit(Dialog)
        self.startDateEdit.setGeometry(QtCore.QRect(20, 40, 110, 24))
        self.startDateEdit.setDate(QtCore.QDate(int(now.year), int(now.month), int(now.day)))
        self.startDateEdit.setObjectName("startDateEdit")
        self.startDateEdit.setCalendarPopup(True)
        self.endDateEdit = QtWidgets.QDateEdit(Dialog)
        self.endDateEdit.setGeometry(QtCore.QRect(260, 40, 110, 24))
        self.endDateEdit.setDate(QtCore.QDate(int(now.year), int(now.month), int(now.day)))
        self.endDateEdit.setObjectName("endDateEdit")
        self.endDateEdit.setCalendarPopup(True)
        self.startDateLabel = QtWidgets.QLabel(Dialog)
        self.startDateLabel.setGeometry(QtCore.QRect(20, 20, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.startDateLabel.setFont(font)
        self.startDateLabel.setObjectName("startDateLabel")
        self.endDateLabel = QtWidgets.QLabel(Dialog)
        self.endDateLabel.setGeometry(QtCore.QRect(260, 20, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.endDateLabel.setFont(font)
        self.endDateLabel.setObjectName("endDateLabel")
        self.noteLabel = QtWidgets.QLabel(Dialog)
        self.noteLabel.setGeometry(QtCore.QRect(20, 270, 451, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.noteLabel.setFont(font)
        self.noteLabel.setObjectName("noteLabel")
        self.progressBar = QtWidgets.QProgressBar(Dialog)
        self.progressBar.setGeometry(QtCore.QRect(30, 250, 441, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.checkDebug = QtWidgets.QCheckBox(Dialog)
        self.checkDebug.setGeometry(QtCore.QRect(20, 170, 131, 20))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.checkDebug.setFont(font)
        self.checkDebug.setObjectName("checkDebug")
        self.checkPlayoff = QtWidgets.QCheckBox(Dialog)
        self.checkPlayoff.setGeometry(QtCore.QRect(20, 140, 131, 20))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.checkPlayoff.setFont(font)
        self.checkPlayoff.setObjectName("checkPlayoff")
        self.checkJoinCSV = QtWidgets.QCheckBox(Dialog)
        self.checkJoinCSV.setGeometry(QtCore.QRect(20, 110, 341, 20))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.checkJoinCSV.setFont(font)
        self.checkJoinCSV.setObjectName("checkJoinCSV")
        self.checkSaveSource = QtWidgets.QCheckBox(Dialog)
        self.checkSaveSource.setGeometry(QtCore.QRect(20, 200, 341, 20))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.checkSaveSource.setFont(font)
        self.checkSaveSource.setObjectName("checkSaveSource")
        self.downloadButton = QtWidgets.QPushButton(Dialog)
        self.downloadButton.setGeometry(QtCore.QRect(400, 210, 81, 32))
        self.downloadButton = QtWidgets.QPushButton(Dialog)
        self.downloadButton.setGeometry(QtCore.QRect(400, 210, 81, 32))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.downloadButton.setFont(font)
        self.downloadButton.setObjectName("DownloadButton")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.savePathButton = QtWidgets.QPushButton(Dialog)
        self.savePathButton.setGeometry(QtCore.QRect(380, 160, 101, 32))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Gothic")
        self.savePathButton.setFont(font)
        self.savePathButton.setObjectName("savePathButton")
        self.pathDialog = QtWidgets.QFileDialog()
        self.pathDialog.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)

        self.retranslateUi(Dialog)
        self.downloadButton.clicked.connect(self.startDownload)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.dirname = None
        self.total_games = 0
        self.done = 0
        self.skipped = 0
        self.broken = 0
        self.savePathButton.clicked.connect(self.getSavePath)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("PBP", "PBP"))
        self.startDateLabel.setText(_translate("Dialog", "Game Date >="))
        self.endDateLabel.setText(_translate("Dialog", "Game Date <="))
        self.noteLabel.setText(_translate("Dialog", ""))
        self.checkDebug.setText(_translate("Dialog", "디버그 메시지 출력"))
        self.checkPlayoff.setText(_translate("Dialog", "포스트시즌 포함"))
        self.checkJoinCSV.setText(_translate("Dialog", "연도별 묶음 파일 생성(기존 파일은 삭제)"))
        self.checkSaveSource.setText(_translate("Dialog", "소스 파일 내용 저장(캐싱 용도)"))
        self.downloadButton.setText(_translate("Dialog", "다운로드!"))
        self.savePathButton.setText(_translate("Dialog", "저장 위치 선택"))

    def getSavePath(self):
        self.dirname = self.pathDialog.getExistingDirectory()

    def startDownload(self):
        now = datetime.datetime.now()
        _translate = QtCore.QCoreApplication.translate
        sd = self.startDateEdit.date()
        ed = self.endDateEdit.date()
        start_date = datetime.date(sd.year(), sd.month(), sd.day())
        end_date = datetime.date(ed.year(), ed.month(), ed.day())
        playoff = self.checkPlayoff.isChecked()
        save_path = self.dirname
        debug_mode = self.checkDebug.isChecked()
        join_csv = self.checkJoinCSV.isChecked()
        save_source = self.checkSaveSource.isChecked()

        sp = None
        if save_path is None:
            save_path = 'save'
            suffix = 1
            sp = pathlib.Path(save_path)
            while True:
                sp = pathlib.Path(save_path)
                if sp.exists() & sp.is_dir():
                    break
                try:
                    sp.mkdir()
                    self.noteLabel.setText(_translate("Dialog",
                                           f"알림: 저장경로 '{sp.stem}' 생성했습니다"))
                except FileExistsError:
                    save_path = 'save' + str(suffix)
                    suffix += 1
                    continue
                break
        else:
            sp = pathlib.Path(save_path)

        try:
            start_time = time.time()
            game_ids = get_game_ids(start_date, end_date, playoff)
            end_time = time.time()
            get_game_id_time = end_time - start_time
            self.total_games = len(game_ids)

            if self.total_games == 0:
                self.noteLabel.setText(_translate("Dialog",
                                       "다운로드할 경기가 없습니다."))
            else:
                self.noteLabel.setText(_translate("Dialog",
                                       f"다운로드 진행 중... (0/{self.total_games})"))
                enc = 'cp949' if sys.platform is 'win32' else 'utf-8'
                logfile = open('./log.txt', 'a', encoding=enc)
                logfile.write('\n\n')
                logfile.write('====================================\n')
                logfile.write(f"Current Time : {now.isoformat()}\n")
                logfile.write('====================================\n')

                self.skipped = 0
                self.broken = 0
                self.done = 0
                start_time = time.time()
                get_data_time = 0
                
                years = list(set([x[:4] for x in game_ids]))

                try:
                    for y in years:
                        y_path = sp / y

                        if not y_path.is_dir():
                            try:
                                y_path.mkdir()
                            except FileExistsError:
                                logfile.write(f'ERROR : path {y_path} exists, but not a directory')
                                logfile.write(f'\tclean path and try again')
                                self.noteLabel.setText(_translate("Dialog",
                                    f"알림: 저장경로 {str(y_path)}가 존재하지만 디렉토리가 아닙니다. 확인 후 재실행하세요."))
                                assert False

                    m = self.progressBar.maximum()
                    t = self.total_games

                    self.progressBar.setValue(0)
                    for gid in game_ids:
                        c = self.done + self.skipped + self.broken
                        self.progressBar.setValue(int(m*c/t))
                        self.noteLabel.setText(_translate("Dialog",
                                               f"다운로드 진행 중... ({c+1}/{t})"))
                        gid_to_date = datetime.date(int(gid[:4]),
                                                    int(gid[4:6]),
                                                    int(gid[6:8]))
                        if gid_to_date > now.date():
                            continue

                        if (sp / gid[:4] / f'{gid}.csv').exists():
                            self.skipped += 1
                            continue

                        ptime = time.time()
                        source_path = sp / gid[:4] / 'source'
                        if (source_path / f'{gid}_pitching.csv').exists() &\
                            (source_path / f'{gid}_batting.csv').exists() &\
                            (source_path / f'{gid}_relay.csv').exists():
                            game_data_dfs = []
                            game_data_dfs.append(pd.read_csv(str(source_path / f'{gid}_pitching.csv')))
                            game_data_dfs.append(pd.read_csv(str(source_path / f'{gid}_batting.csv')))
                            game_data_dfs.append(pd.read_csv(str(source_path / f'{gid}_relay.csv')))
                        else:
                            game_data_dfs = get_game_data(gid)

                        if game_data_dfs[0] is None:
                            logfile.write(game_data_dfs[-1])
                            if debug_mode == True:
                                print(game_data_dfs[-1])
                            assert False
                            self.noteLabel.setText(_translate("Dialog",
                                                   "ERROR: 로그 파일(log.txt)을 참조하세요."))

                        if save_source == True:
                            if not source_path.is_dir():
                                try:
                                    source_path.mkdir()
                                except FileExistsError:
                                    source_path = save_path / gid[:4]
                                    logfile.write(f'NOTE: {gid[:4]}/source exists but not a directory.')
                                    logfile.write(f'source files will be saved in {gid[:4]} instead.')

                            if not (source_path / f'{gid}_pitching.csv').exists():
                                game_data_dfs[0].to_csv(str(source_path / f'{gid}_pitching.csv'), index=False, encoding=enc)
                            if not (source_path / f'{gid}_batting.csv').exists():
                                game_data_dfs[1].to_csv(str(source_path / f'{gid}_batting.csv'), index=False, encoding=enc)
                            if not (source_path / f'{gid}_relay.csv').exists():
                                game_data_dfs[2].to_csv(str(source_path / f'{gid}_relay.csv'), index=False, encoding=enc)

                        get_data_time += time.time() - ptime
                        if game_data_dfs is not None:
                            gs = game_status()
                            gs.load(gid, game_data_dfs[0], game_data_dfs[1], game_data_dfs[2], log_file=logfile)
                            parse = gs.parse_game(debug_mode)
                            gs.save_game(sp / gid[:4])
                            if parse == True:
                                self.done += 1
                            else:
                                self.broken += 1
                        else:
                            self.broken += 1
                    if join_csv == True:
                        join_csvs(sp, start_date, end_date)

                    end_time = time.time()
                    self.progressBar.setValue(m)

                    parse_time = end_time - start_time - get_data_time
                    logfile.write('====================================\n')
                    logfile.write(f'Start date : {start_date.strftime("%Y%m%d")}\n')
                    logfile.write(f'End date : {end_date.strftime("%Y%m%d")}\n')
                    logfile.write(f'Successfully downloaded games : {self.done}\n')
                    logfile.write(f'Skipped games(already exists) : {self.skipped}\n')
                    logfile.write(f'Broken games(bad data) : {self.broken}\n')
                    logfile.write('====================================\n')
                    if debug_mode == True:
                        logfile.write(f'Elapsed {get_game_id_time:.2f} sec in get_game_ids\n')
                        logfile.write(f'Elapsed {(get_data_time):.2f} sec in get_game_data\n')
                        logfile.write(f'Elapsed {(parse_time):.2f} sec in parse_game\n')
                    logfile.write(f'Total {(parse_time+get_game_id_time+get_data_time):.2f} sec elapsed with {len(game_ids)} games\n')
                    self.noteLabel.setText(_translate("Dialog",
                                           f"총 {len(game_ids)} 경기 다운로드 완료."))

                    if logfile.closed is not True:
                        logfile.close()
                except:
                    if logfile.close is not True:
                        logfile.close()
                    assert False
        except:
            self.noteLabel.setText(_translate("Dialog",
                                    "ERROR: 로그 파일(log.txt)을 참조하세요."))
            log = open('log.txt', 'a')
            log.write('====================================\n')
            log.write(getTracebackStr())
            log.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
