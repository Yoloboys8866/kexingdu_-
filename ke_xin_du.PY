import sys
import re
import time
from collections import deque
import numpy as np

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QLineEdit, QVBoxLayout, QHBoxLayout, QCheckBox, QMessageBox, QComboBox
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 确保中文显示正常
matplotlib.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    serial = None
    HAS_SERIAL = False


class SerialReader(QtCore.QThread):
    data_received = QtCore.pyqtSignal(str)
    connection_ready = QtCore.pyqtSignal()
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, port=None, baudrate=115200):
        super().__init__()
        self._running = False
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def run(self):
        self._running = True
        if not HAS_SERIAL:
            self.error_occurred.emit("pyserial模块未安装")
            return

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to {self.ser.name} at {self.baudrate}")
            self.connection_ready.emit()
            
            while self._running:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        print(f"Received: {line}")  # 调试信息
                        self.data_received.emit(line)
        except serial.SerialException as e:
            error_msg = f"无法打开端口 {self.port}: {str(e)}"
            print(f"Error: {e}")
            self.error_occurred.emit(error_msg)
            return
        except Exception as e:
            self.error_occurred.emit(f"串口读取错误: {str(e)}")

    def stop(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


class RealTimeCIRPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time CIR Quality Flag Viewer")

        # Available fields we know how to parse
        self.fields = [
            "D", "fom", "PD01", "PD02", "PD12", "azimuth", "elevation",                                              # initiator
        ]

        # 定义页面分组
        self.page_groups = [
            {"name": "距离和质量因数", "fields": ["D", "fom"]},
            {"name": "相位差", "fields": ["PD01", "PD02", "PD12"]},
            {"name": "角度", "fields": ["azimuth", "elevation"]}
        ]

        # === Left panel ===
        self.checkboxes = {}
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Select parameters to plot:"))
        for field in self.fields:
            cb = QCheckBox(field)
            cb.setChecked(field in ("D", "azimuth", "elevation"))  # default enabled
            # 添加信号槽连接，当用户选择/取消选择参数时更新图表
            cb.stateChanged.connect(self.on_checkbox_changed)
            self.checkboxes[field] = cb
            left_layout.addWidget(cb)

        # 添加端口选择
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.update_port_list()
        # 默认选择COM10（如果存在）
        index = self.port_combo.findText("COM10")
        if index >= 0:
            self.port_combo.setCurrentIndex(index)
        port_layout.addWidget(self.port_combo)
        left_layout.addLayout(port_layout)

        # 添加刷新端口按钮
        refresh_port_button = QPushButton("刷新端口")
        refresh_port_button.clicked.connect(self.update_port_list)
        left_layout.addWidget(refresh_port_button)

        # 添加全选按钮
        self.select_all_button = QPushButton("全选")
        self.select_all_button.clicked.connect(self.select_all)
        left_layout.addWidget(self.select_all_button)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_acquisition)
        self.is_running = False  # 添加运行状态标志
        left_layout.addWidget(self.start_button)

        # 添加状态标签
        self.status_label = QLabel("就绪")
        left_layout.addWidget(self.status_label)

        # === Matplotlib Figure with dynamic subplots ===
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        # === 分页控制 ===
        self.current_page = 0
        self.page_layout = QHBoxLayout()
        self.prev_button = QPushButton("上一页")
        self.prev_button.clicked.connect(self.prev_page)
        self.prev_button.setEnabled(False)
        self.page_label = QLabel(f"{self.page_groups[0]['name']}")
        self.next_button = QPushButton("下一页")
        self.next_button.clicked.connect(self.next_page)
        self.next_button.setEnabled(True)
        self.page_layout.addWidget(self.prev_button)
        self.page_layout.addWidget(self.page_label)
        self.page_layout.addWidget(self.next_button)

        # 主布局
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)
        right_layout.addLayout(self.page_layout)

        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(left_layout)
        self.main_layout.addLayout(right_layout)

        self.setLayout(self.main_layout)

        # Buffers
        self.max_samples = 500
        self.data_buffers = {f: deque(maxlen=self.max_samples) for f in self.fields}
        self.sample_indices = deque(maxlen=self.max_samples)

        self.reader = None
        
        # 初始化图表显示
        self.update_plots()

    def update_port_list(self):
        """更新可用的COM端口列表"""
        self.port_combo.clear()
        if HAS_SERIAL:
            try:
                ports = serial.tools.list_ports.comports()
                for port in ports:
                    self.port_combo.addItem(port.device)
            except Exception as e:
                print(f"获取端口列表失败: {e}")
                self.port_combo.addItem("COM10")
        else:
            self.port_combo.addItem("COM10")

    def select_all(self):
        # 检查是否有未选中的字段
        any_unchecked = any(not cb.isChecked() for cb in self.checkboxes.values())
        # 全部选中或取消全部选中
        for cb in self.checkboxes.values():
            cb.setChecked(any_unchecked)

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_page_display()
            self.update_plots()

    def next_page(self):
        if self.current_page < len(self.page_groups) - 1:
            self.current_page += 1
            self.update_page_display()
            self.update_plots()

    def update_page_display(self):
        # 更新页面标签和按钮状态
        self.page_label.setText(f"{self.page_groups[self.current_page]['name']}")
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < len(self.page_groups) - 1)

    def toggle_acquisition(self):
        """切换数据采集的开始/停止状态"""
        if self.is_running:
            # 当前正在运行，需要停止
            if self.reader:
                self.reader.stop()
                self.reader.wait()
            self.start_button.setText("Start")
            self.status_label.setText("已停止")
            self.is_running = False
        else:
            # 当前已停止，需要开始
            # 重置缓冲区
            for buf in self.data_buffers.values():
                buf.clear()
            self.sample_indices.clear()

            if self.reader:
                self.reader.stop()
                self.reader.wait()

            selected_port = self.port_combo.currentText()
            self.status_label.setText(f"连接到 {selected_port}...")
            
            self.reader = SerialReader(port=selected_port, baudrate=115200)
            self.reader.data_received.connect(self.handle_line)
            self.reader.connection_ready.connect(lambda: self.status_label.setText(f"已连接到 {selected_port}"))
            self.reader.error_occurred.connect(lambda msg: self.status_label.setText(msg))
            self.reader.start()
            
            self.start_button.setText("Stop")
            self.is_running = True

    def on_checkbox_changed(self):
        # 当用户选择或取消选择参数时更新图表
        self.update_plots()

    def handle_line(self, line):
        # Example: Cycle:1281, D:15.812494cm,PD01:114.607376, PD02:152.379532, ...
        # 尝试多种正则表达式模式来匹配不同格式的数据
        patterns = [
            r'([A-Za-z0-9]+):\s*(-?\d+(?:\.\d+)?)',  # 基本模式
            r'([A-Za-z0-9]+)\s*=\s*(-?\d+(?:\.\d+)?)',  # = 分隔的模式
            r'([A-Za-z0-9]+)\s*:\s*(-?\d+(?:\.\d+)?)(?:cm|mm|m)?'  # 带单位的模式
        ]
        
        values = {}
        for pattern in patterns:
            matches = re.findall(pattern, line)
            if matches:
                values = {k.strip(): float(v) for k, v in matches}
                break
        
        if not values:
            print(f"无法解析数据: {line}")
            return

        # 只要有数据就更新
        idx = self.sample_indices[-1] + 1 if self.sample_indices else 0
        self.sample_indices.append(idx)
        
        for f in self.fields:
            if f in values:
                self.data_buffers[f].append(values[f])
            else:
                # 使用NaN填充缺失值
                self.data_buffers[f].append(np.nan)
        
        # 更新图表
        self.update_plots()

    def update_plots(self):
        self.figure.clf()
        active_fields = [f for f, cb in self.checkboxes.items() if cb.isChecked()]
        
        # 获取当前页的字段组
        current_group = self.page_groups[self.current_page]
        current_group_fields = current_group["fields"]
        
        # 过滤当前组中已勾选的字段
        current_fields = [f for f in current_group_fields if f in active_fields]
        
        if not current_fields:
            # 即使没有选中字段，也要显示空图表和提示
            ax = self.figure.add_subplot(111)
            ax.set_title("没有选中要显示的参数")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Value")
            self.canvas.draw_idle()
            return

        n = len(current_fields)
        for i, f in enumerate(current_fields, 1):
            ax = self.figure.add_subplot(n, 1, i)
            
            # 确保x和y的长度相同
            x_data = list(self.sample_indices)
            y_data = list(self.data_buffers[f])
            
            # 截断较长的数据以匹配较短的数据
            min_length = min(len(x_data), len(y_data))
            x_data = x_data[:min_length]
            y_data = y_data[:min_length]
            
            # 设置图表标题和标签
            ax.set_title(f)
            ax.set_ylabel(f)
            if i == n:  # 最后一个图表添加x轴标签
                ax.set_xlabel("Sample index")
            
            if x_data and y_data:
                # 过滤掉NaN值
                valid_data = [(x, y) for x, y in zip(x_data, y_data) if not np.isnan(y)]
                if valid_data:
                    valid_x, valid_y = zip(*valid_data)
                    ax.plot(valid_x, valid_y, label=f)
                    # 设置合理的Y轴范围
                    y_min, y_max = min(valid_y), max(valid_y)
                    y_range = y_max - y_min
                    if y_range == 0:  # 所有值都相同
                        y_range = 1.0
                    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
            
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True)
            
        self.figure.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RealTimeCIRPlot()
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())