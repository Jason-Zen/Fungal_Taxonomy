"""
==========================================
  Graphic user interface of MycoPhenoLib
==========================================
"""

import os
import sys
import traceback
from io import StringIO

import matplotlib
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QIcon, QTextCharFormat, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QHBoxLayout, \
    QCheckBox, QFileDialog, QPlainTextEdit, QFormLayout, QGroupBox, QLineEdit

from MPL.__init__ import MPLInfo
from MPL.Master import ExcelParser


class ThreadConsole(QThread):
    signal = pyqtSignal(str, str)

    def __init__(self, path, metr, ib, clf, sheet, pred, val, model, cls, idx):
        super().__init__()
        self.running = True
        self.path = path
        self.ib = ib
        self.metr = metr
        self.clf = clf
        self.sheet = sheet
        self.pred = pred
        self.val = val
        self.model = model
        self.cls = cls
        self.idx = idx

    def run(self):
        self.running = True
        try:
            parser = ExcelParser(ib=self.ib, excel_in=self.path, sheet_pred=self.sheet)

            if self.model is None:
                parser.train(opt=self.clf, metric=self.metr, stop_signal=self.stop_signal)

            if self.stop_signal():
                self.signal.emit("***** Terminated *****", "red")
                return

            if self.pred:
                parser.predict(val=self.val, model_path=self.model)

                if self.stop_signal():
                    self.signal.emit("***** Terminated *****", "red")
                    return

                if (self.cls is None) or (self.idx is None):
                    pass
                else:
                    parser.explain(cls=self.cls, idx=self.idx)

            self.signal.emit("[][][][][] Finished [][][][][]", 'lime')
            self.running = False

        except Exception:
            error_output = StringIO()
            traceback.print_exc(file=error_output)  # 打印异常回溯路径
            error_string = error_output.getvalue()
            self.signal.emit(f"Error:\n{error_string.strip()}", "red")
            return

    def stop(self):
        self.running = False
        self.signal.emit("Terminating ...", "orange")

    def stop_signal(self):
        return not self.running


class EmitStr(QObject):
    textWrit = pyqtSignal(str)

    def write(self, text):
        self.textWrit.emit(text)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.console = None
        self.path = None
        self.path_model = None
        self.metric = 'F1'
        self.ib = None
        self.clf = None
        self.sheet_predict = "TD"
        self.pred = None
        self.val = None
        self.cls = None
        self.idx = None

        self.font_groupbox = """
            QGroupBox {
                border: ; /* 边框宽度 样式(solid、dotted、dashed)  颜色 如2px solid #CCCCFF*/
                font-size: 10pt; /* 字体大小 */
                font-family: Calibri; /* 字体类型 */
                font-weight: bold; /* 加粗 */
            }
            QGroupBox::title {
                subcontrol-origin: ;
                subcontrol-position: ; /* 标题位置 */
                color: #28282B; /* 颜色 */
            }
        """
        self.font_button = """
                QPushButton {
                    background-color: #3498db; /* 设置背景色为深灰色 */
                    border: none; /* 移除边框 */
                    color: #fff; /* 文本颜色为白色 */
                    padding: 8px 12px; /* 设置内边距 */
                    text-align: center; /* 文本居中 */
                    text-decoration: none; /* 文本装饰（如下划线） */
                    display: inline-block; /* 将元素显示为行内块级元素 */
                    font-size: 14px; /* 设置字体大小 */
                    font-family: 'Arial', sans-serif; /* 设置字体类型 */
                    font-weight: bold; /* 设置字体加粗 */
                    margin: 5px 5px; /* 设置外边距 */
                    cursor: pointer; /* 鼠标样式 */
                    border-radius: 4px; /* 设置圆角 */
                    box-shadow: 0px 0px 8px rgba(0, 0, 0, 0.3); /* 添加阴影效果 */
                    transition: background-color 0.3s; /* 添加过渡效果 */
                    width: auto; /* 自动宽度 */
                }
                QPushButton:hover {
                    background-color: #2980b9; /* 悬停时的背景色 */
                }
                QPushButton:pressed {
                    background-color: #1f618d; /* 按下时的背景色 */
                }
                QPushButton:disabled {
                    background-color: #bdc3c7; /* 禁用时的背景色 */
                    color: #7f8c8d; /* 禁用时的文本颜色 */
                    cursor: not-allowed; /* 禁用时的鼠标样式 */
                }
            """

        # Module1: Training data
        self.label_file = QLabel('')
        button1 = QPushButton('Select data')
        button1.clicked.connect(self.mod1_file)
        self.button1_2 = QPushButton('Open output folder')
        self.button1_2.clicked.connect(self.mod1_folder)
        self.button1_2.setEnabled(False)
        # Sampling method
        label_sampling = QLabel('Sampling method:')
        self.opt_sampling = QComboBox()
        self.opt_sampling.addItems(['Normal sampling', 'Under sampling', 'Over sampling'])
        self.opt_sampling.setCurrentIndex(0)
        # self.opt_sampling.currentIndexChanged.connect(self.mod1_sampling)
        # Metric
        label_metric = QLabel('Metric for selecting the best model:')
        self.opt_metric = QComboBox()
        self.opt_metric.addItems(['Accuracy', 'Precision', 'Recall', 'F1'])
        self.opt_metric.setCurrentIndex(3)
        # self.opt_metric.currentIndexChanged.connect(self.mod1_metrics)
        # layout
        module1 = QGroupBox("Training datasets")
        layout1 = QFormLayout()
        layout1.addRow(QLabel('Current Excel file:'), self.label_file)
        layout1.addRow(button1, self.button1_2)
        layout1.addRow(label_sampling, self.opt_sampling)
        layout1.addRow(label_metric, self.opt_metric)
        module1.setLayout(layout1)
        module1.setStyleSheet(self.font_groupbox)

        # Module2: Data to be predicted
        label2_1 = QLabel(f"Excel has worksheet for prediction")
        self.opt2_1 = QCheckBox()
        self.opt2_1.stateChanged.connect(self.mod2_stat)
        label2_2 = QLabel("True labels in the 1st column")
        self.opt2_2 = QCheckBox()
        self.label_model = QLabel('')
        self.button2_1 = QPushButton('Select model')
        self.button2_1.setEnabled(False)
        self.button2_1.clicked.connect(self.mod2_file)
        self.button2_2 = QPushButton('Clear model')
        self.button2_2.setEnabled(False)
        self.button2_2.clicked.connect(self.mod2_file_clear)
        self.text2 = QLineEdit()
        self.text2.setEnabled(False)
        self.text2.textChanged.connect(self.mod2_text1)
        # layout
        module2 = QGroupBox("Data for prediction")
        layout2 = QFormLayout()
        layout2.addRow(self.opt2_1, label2_1)
        layout2.addRow(self.opt2_2, label2_2)
        layout2.addRow(QLabel("Name of the worksheet:"), self.text2)
        layout2.addRow(QLabel('Current model:'), self.label_model)
        layout2.addRow(self.button2_1, self.button2_2)
        module2.setLayout(layout2)
        module2.setStyleSheet(self.font_groupbox)

        # Module3: Data to be explained
        label3_1 = QLabel("Target class:")
        self.text3_1 = QLineEdit()
        self.text3_1.setEnabled(False)
        self.text3_1.textChanged.connect(self.mod3_text1)
        label3_2 = QLabel("Target index:")
        self.text3_2 = QLineEdit()
        self.text3_2.setEnabled(False)
        self.text3_2.textChanged.connect(self.mod3_text2)
        # layout
        module3 = QGroupBox("Prediction to be explained")
        layout3 = QFormLayout()
        layout3.addRow(label3_1, self.text3_1)
        layout3.addRow(label3_2, self.text3_2)
        module3.setLayout(layout3)
        module3.setStyleSheet(self.font_groupbox)

        # Module5: Classifiers to be tested
        label_clf1 = {}  # Classifier labels1
        label_clf2 = {}  # Classifier labels2
        self.opt_clf = {}  # Classifier checkboxes
        classifier_opt1 = [
            ('LR', 'Logistic Regression'),
            ('KNN', 'K Nearest Neighbors'),
            ('SVM', 'Support Vector Machines'),
            ('NN', 'Neural Networks'),
            ('AB', 'Adaboost'),
        ]
        classifier_opt2 = [
            ('DT', 'Decision Trees'),
            ('RF', 'Random Forests'),
            ('ET', 'Extra Trees'),
            ('ERT', 'Extremely Randomized Tree'),
            ('XGB', 'Extreme Gradient Boosting'),
            # ('GB', 'Gradient Boosting'),
        ]
        # Create label and checkboxes for classifiers
        for classifier, label in classifier_opt1:
            self.opt_clf[classifier] = QCheckBox()
            label_clf1[classifier] = QLabel(label)
        for classifier, label in classifier_opt2:
            self.opt_clf[classifier] = QCheckBox()
            label_clf2[classifier] = QLabel(label)
        # Set default checked classifiers
        check_default = ['DT', 'RF', 'ET', 'ERT']
        for clf_key in check_default:
            self.opt_clf[clf_key].setChecked(True)
        self.button5_1 = QPushButton('Select all')
        self.button5_1.clicked.connect(self.mod5_select)
        self.button5_2 = QPushButton('Unselect all')
        self.button5_2.clicked.connect(self.mod5_unselect)
        # layout
        module5 = QGroupBox("ML algorithms")
        layout5 = QHBoxLayout()
        layout5_1 = QFormLayout()
        layout5_1.addRow(self.button5_1)
        for clf_key, clf_label in label_clf1.items():
            layout5_1.addRow(self.opt_clf[clf_key], clf_label)
        layout5_2 = QFormLayout()
        layout5_2.addRow(self.button5_2)
        for clf_key, clf_label in label_clf2.items():
            layout5_2.addRow(self.opt_clf[clf_key], clf_label)
        layout5.addLayout(layout5_1)
        layout5.addLayout(layout5_2)
        module5.setLayout(layout5)
        module5.setStyleSheet(self.font_groupbox)

        # Module6: Buttons
        layout6 = QHBoxLayout()
        self.run_button = QPushButton('Run')
        self.run_button.clicked.connect(self.activate_run_button)
        self.run_button.setStyleSheet(self.font_button)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.activate_stop_button)
        self.stop_button.setStyleSheet(self.font_button)
        layout6.addWidget(self.run_button)
        layout6.addWidget(self.stop_button)

        # Module7: Display console
        self.m7_display = QPlainTextEdit()
        self.m7_display.setReadOnly(True)
        # self.console_output.setFixedSize(800, 650)
        self.m7_display.setStyleSheet("""
            QPlainTextEdit {
                background-color: black;
                color: white;
                font-family: 'Times New Roman';
                font-size: 10pt;
            }
        """)
        # layout
        layout7 = QVBoxLayout()
        layout7.addWidget(self.m7_display)

        # Module: Main
        layout_set = QVBoxLayout()
        layout_set.setSpacing(20)
        layout_set.addWidget(module1)
        layout_set.addWidget(module2)
        layout_set.addWidget(module3)
        layout_set.addWidget(module5)
        layout_set.addLayout(layout6)

        layout_main = QHBoxLayout()
        layout_main.addLayout(layout_set)
        layout_main.addLayout(layout7)
        layout_main.setStretch(0, 1)
        layout_main.setStretch(1, 3)

        # Window
        self.setLayout(layout_main)
        self.setWindowTitle(f'MycoPhenoLib - {MPLInfo.version} ({MPLInfo.date})')
        self.setWindowIcon(QIcon('ico.png'))  # 设置窗口图标

        # 重定向输出
        sys.stdout = EmitStr(textWrit=self.outwrite)
        sys.stderr = EmitStr(textWrit=self.outwrite)

    def outwrite(self, text):
        self.m7_display.insertPlainText(text)

    def mod1_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, 'Select', '', 'Excel file (*.xlsx);;All Files (*)')
            if file_path:
                self.label_file.setText(file_path)
                self.path = file_path
                self.button1_2.setEnabled(True)
        except Exception as e:
            self.threadconsole_print(str(e), 'red')

    def mod1_folder(self):
        folder_path = os.path.join(os.path.dirname(self.path), "Out_ML")  # 指定文件夹的路径
        if os.path.exists(folder_path):  # 检查路径是否存在
            os.startfile(folder_path)  # 在 Windows 上打开文件夹
        else:
            self.threadconsole_print("!!!!! Output folder does not exist !!!!!", 'orange')

    def mod2_stat(self):
        if self.opt2_1.isChecked():
            self.button2_1.setEnabled(True)
            self.text2.setEnabled(True)
        else:
            self.button2_1.setEnabled(False)
            self.text2.setEnabled(False)

    def mod2_file(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, 'Select', '', 'Model file (*.pkl)')
            if file_path:
                self.label_model.setText(file_path)
                self.path_model = file_path
                self.text3_1.setEnabled(True)
                self.text3_2.setEnabled(True)
                self.button2_2.setEnabled(True)
        except Exception as e:
            self.threadconsole_print(str(e), 'red')

    def mod2_file_clear(self):
        self.label_model.setText('')
        self.path_model = None
        self.text3_1.setEnabled(False)
        self.text3_2.setEnabled(False)
        self.text3_1.clear()
        self.text3_2.clear()

    def mod2_text1(self, text):
        self.sheet_predict = text

    def mod3_text1(self, text):
        if text == '':
            self.cls = None
        else:
            self.cls = text

    def mod3_text2(self, text):
        if text == '':
            self.idx = None
        else:
            self.idx = text

    def mod5_select(self):
        for k, v in self.opt_clf.items():
            v.setChecked(True)

    def mod5_unselect(self):
        for k, v in self.opt_clf.items():
            v.setChecked(False)

    def m6_input(self):
        command = self.text6.text()
        if command:
            self.m7_display.appendPlainText(command)

            self.text6.clear()

    def threadconsole_print(self, text, color='white'):
        if text == "[][][][][] Finished [][][][][]" or "***** Terminated *****" or "Error":
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        # 创建一个 QTextCursor 对象，它关联到 console_output 控件
        cursor = self.m7_display.textCursor()

        # 获取当前文本格式
        current_format = cursor.charFormat()

        # 创建一个新的文本格式对象，并设置字体颜色
        text_format = QTextCharFormat()
        text_format.setForeground(QColor(color))

        # 移动光标到文档末尾
        cursor.movePosition(cursor.End)

        # 应用新的文本格式
        cursor.setCharFormat(text_format)
        cursor.insertText(f'{text}\n')  # 插入文本

        # 恢复原始文本格式
        cursor.setCharFormat(current_format)

        self.m7_display.setTextCursor(cursor)  # 更新 QTextEdit 的光标位置
        self.m7_display.ensureCursorVisible()  # 确保光标（新内容）在视图中可见

    def options(self):
        # mod1_metrics: 直接从 self.sampling 获取当前选定的文本
        _metric = self.opt_metric.currentText()
        self.metric = _metric

        # mod1_sampling: 直接从 self.sampling 获取当前选定的文本
        _ib = self.opt_sampling.currentText()
        opt_ib = {
            'Normal sampling': None,
            'Under sampling': 'Under',
            'Over sampling': 'Over'
        }
        self.ib = opt_ib[_ib]

        # mod2_opt
        if self.opt2_1.isChecked():
            self.pred = True
            if self.opt2_2.isChecked():
                self.val = True
            else:
                self.val = False
        else:
            self.pred = False

    # 线程1控制方法
    def activate_run_button(self):
        self.m7_display.clear()

        # check input Excel
        if not self.path:
            self.threadconsole_print("!!!!! Please select the input file !!!!!", 'orange')
            return

        # check model file
        if not self.path_model:
            # check algorithms
            self.clf = [clf for clf, checkbox in self.opt_clf.items() if checkbox.isChecked()]
            if len(self.clf) == 0:
                self.threadconsole_print("!!!!! Please select ML algorithms to be tested !!!!!", 'orange')
                return

        self.options()

        if self.idx is not None:
            try:
                if not int(self.idx) > 0:
                    self.threadconsole_print("!!!!! Target index should be larger than 0 !!!!!", 'orange')
                    return
            except Exception:
                self.threadconsole_print("!!!!! Target index should be an integer !!!!!", 'orange')
                return

        self.console = ThreadConsole(
            self.path,
            self.metric,
            self.ib,
            self.clf,
            self.sheet_predict,
            self.pred,
            self.val,
            self.path_model,
            self.cls,
            self.idx
        )
        self.console.signal.connect(self.threadconsole_print)

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.console.start()

    def activate_stop_button(self):
        if self.console and self.console.isRunning():
            self.stop_button.setEnabled(False)
            self.console.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    matplotlib.use('pdf')

    GUI = MainWindow()
    GUI.resize(1500, 500)  # initial size
    GUI.show()

    sys.exit(app.exec_())
