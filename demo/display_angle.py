# --coding--:utf-8 --
from PyQt5.Qt import *


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("显示角度的应用")
        self.resize(500, 500)
        self.setup_ui()

    def setup_ui(self):
        self.find_file()

    def find_file(self):
        btn = QPushButton(self)
        btn.resize(200, 40)
        btn.move(20, 20)
        btn.setText("显示角度的按钮")
        textEdit = QTextEdit(self)
        textEdit.resize(200, 400)
        textEdit.move(80, 80)
        textEdit.setText("显示角度的文本框")

        def show_cao():
            print("已找到文件。。。。。。")

        def show_angel():
            with open('angle.txt', 'r', encoding="utf-8") as f:
                msg = f.read()
                textEdit.setPlainText(msg)

        btn.clicked.connect(show_angel)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    window = Window()
    window.show()

    sys.exit(app.exec_())