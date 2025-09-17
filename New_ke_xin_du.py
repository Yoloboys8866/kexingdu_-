import sys
import re
import time
from collections import deque
import numpy as np
from datetime import datetime

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QLineEdit, QVBoxLayout, QHBoxLayout, QCheckBox, QMessageBox, QComboBox,
    QFileDialog, QSplitter, QGroupBox
)

import matplotlib
# 修复：将Qt5Agg后端改为Qt6Agg以匹配PyQt5
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# 确保中文显示正常
matplotlib.rcParams['font.family'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
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

    def __init__(self, port=None, baudrate=115200, max_reconnect_attempts=3):
        super().__init__()
        self._running = False
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_count = 0

    def run(self):
        self._running = True
        if not HAS_SERIAL:
            self.error_occurred.emit("pyserial模块未安装")
            return

        while self._running:
            try:
                if not self.ser or not self.ser.is_open:
                    self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
                    print(f"Connected to {self.ser.name} at {self.baudrate}")
                    self.connection_ready.emit()
                    self.reconnect_count = 0  # 重置重连计数
                
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line:
                        # 仅在调试模式下打印所有接收的数据
                        # print(f"Received: {line}")
                        self.data_received.emit(line)
                else:
                    # 避免CPU占用过高
                    time.sleep(0.01)
                      
            except serial.SerialException as e:
                self.reconnect_count += 1
                if self.reconnect_count <= self.max_reconnect_attempts:
                    error_msg = f"连接断开，尝试重连 ({self.reconnect_count}/{self.max_reconnect_attempts})..."
                    print(f"Error: {e}, {error_msg}")
                    self.error_occurred.emit(error_msg)
                    time.sleep(1)  # 等待1秒后重连
                    continue
                else:
                    error_msg = f"无法打开端口 {self.port}: {str(e)}"
                    print(f"Error: {e}")
                    self.error_occurred.emit(error_msg)
                    break
            except Exception as e:
                self.error_occurred.emit(f"串口读取错误: {str(e)}")
                # 短暂暂停后继续运行，避免因小错误导致整个线程退出
                time.sleep(0.1)
                continue

    def stop(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()


class RealTimeCIRPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CIR数据实时可视化工具")
        
        # 设置应用程序样式
        self.setStyleSheet("""
            QWidget {
                font-size: 12px;
            }
            QPushButton {
                padding: 6px 12px;
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            QLabel {
                padding: 4px 0;
            }
        """)

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
        
        # 创建参数选择组
        param_group = QGroupBox("选择要显示的参数:")
        param_layout = QVBoxLayout()
        for field in self.fields:
            cb = QCheckBox(field)
            cb.setChecked(field in ("D", "azimuth", "elevation"))  # 默认启用的参数
            # 添加信号槽连接，当用户选择/取消选择参数时更新图表
            cb.stateChanged.connect(self.on_checkbox_changed)
            self.checkboxes[field] = cb
            param_layout.addWidget(cb)
        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # 添加端口选择
        port_group = QGroupBox("串口设置:")
        port_layout = QVBoxLayout()
        
        port_row_layout = QHBoxLayout()
        port_row_layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.update_port_list()
        # 默认选择COM10（如果存在）
        index = self.port_combo.findText("COM10")
        if index >= 0:
            self.port_combo.setCurrentIndex(index)
        port_row_layout.addWidget(self.port_combo)
        port_layout.addLayout(port_row_layout)

        # 添加波特率选择（新功能）
        baud_rate_layout = QHBoxLayout()
        baud_rate_layout.addWidget(QLabel("Baud Rate:"))
        self.baud_rate_combo = QComboBox()
        self.baud_rate_combo.addItems(["9600", "19200", "38400", "57600", "921600","115200", "230400"])
        self.baud_rate_combo.setCurrentText("115200")  # 设置默认波特率
        baud_rate_layout.addWidget(self.baud_rate_combo)
        port_layout.addLayout(baud_rate_layout)

        # 添加刷新端口按钮
        refresh_port_button = QPushButton("刷新端口")
        refresh_port_button.clicked.connect(self.update_port_list)
        port_layout.addWidget(refresh_port_button)
        port_group.setLayout(port_layout)
        left_layout.addWidget(port_group)

        # 数据控制组
        control_group = QGroupBox("数据控制:")
        control_layout = QVBoxLayout()
        
        # 添加全选按钮
        self.select_all_button = QPushButton("全选/取消全选")
        self.select_all_button.clicked.connect(self.select_all)
        control_layout.addWidget(self.select_all_button)

        # 添加保存数据按钮
        self.save_data_button = QPushButton("保存数据")
        self.save_data_button.clicked.connect(self.save_data)
        control_layout.addWidget(self.save_data_button)
        
        # 添加清除数据按钮
        self.clear_data_button = QPushButton("清除数据")
        self.clear_data_button.clicked.connect(self.clear_data)
        control_layout.addWidget(self.clear_data_button)
        
        # 添加自动缩放复选框
        self.auto_scale_checkbox = QCheckBox("自动缩放Y轴")
        self.auto_scale_checkbox.setChecked(True)
        control_layout.addWidget(self.auto_scale_checkbox)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # 添加开始/停止按钮
        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.start_button.clicked.connect(self.toggle_acquisition)
        self.is_running = False  # 添加运行状态标志
        left_layout.addWidget(self.start_button)

        # 添加状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("font-weight: bold; color: #2E7D32;")
        left_layout.addWidget(self.status_label)
        
        # 添加统计信息标签
        self.stats_label = QLabel("数据点: 0")
        self.stats_label.setStyleSheet("color: #1976D2;")
        left_layout.addWidget(self.stats_label)
        
        # 填充空间，将所有控件置顶
        left_layout.addStretch()

        # === Matplotlib Figure with dynamic subplots ===
        self.figure = Figure(figsize=(12, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        # === 分页控制 ===
        self.current_page = 0
        self.page_layout = QHBoxLayout()
        self.prev_button = QPushButton("上一页")
        self.prev_button.clicked.connect(self.prev_page)
        self.prev_button.setEnabled(False)
        self.page_label = QLabel(f"{self.page_groups[0]['name']}")
        self.page_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.page_label.setStyleSheet("font-weight: bold;")
        self.next_button = QPushButton("下一页")
        self.next_button.clicked.connect(self.next_page)
        self.next_button.setEnabled(len(self.page_groups) > 1)
        self.page_layout.addWidget(self.prev_button)
        self.page_layout.addWidget(self.page_label)
        self.page_layout.addWidget(self.next_button)

        # 右侧布局
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.canvas)
        right_layout.addLayout(self.page_layout)

        # 使用QSplitter实现可调整的左右布局
        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal)
        
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        # 设置初始大小比例
        splitter.setSizes([300, 900])

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Buffers
        self.max_samples = 500
        self.data_buffers = {f: deque(maxlen=self.max_samples) for f in self.fields}
        self.sample_indices = deque(maxlen=self.max_samples)
        self.timestamps = deque(maxlen=self.max_samples)  # 添加时间戳

        self.reader = None
        self.last_update_time = 0
        self.update_interval = 20  # 限制更新频率为每50毫秒一次（约20FPS）
        
        # 初始化图表显示
        self.update_plots()

    def update_port_list(self):
        """更新可用的COM端口列表"""
        self.port_combo.clear()
        if HAS_SERIAL:
            try:
                ports = serial.tools.list_ports.comports()
                if ports:
                    for port in ports:
                        # 显示端口和描述信息
                        self.port_combo.addItem(f"{port.device} - {port.description}", port.device)
                else:
                    self.port_combo.addItem("未检测到端口", "COM10")
                    QMessageBox.warning(self, "警告", "未检测到可用的COM端口！")
            except Exception as e:
                print(f"获取端口列表失败: {e}")
                self.port_combo.addItem("COM10", "COM10")
        else:
            self.port_combo.addItem("COM10", "COM10")

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
            self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.status_label.setText("已停止")
            self.status_label.setStyleSheet("font-weight: bold; color: #757575;")
            self.is_running = False
        else:
            # 当前已停止，需要开始
            # 重置缓冲区
            for buf in self.data_buffers.values():
                buf.clear()
            self.sample_indices.clear()
            self.timestamps.clear()

            if self.reader:
                self.reader.stop()
                self.reader.wait()

            # 获取实际的端口设备名
            selected_port = self.port_combo.currentData() or "COM10"
            # 获取用户选择的波特率（新功能）
            selected_baudrate = int(self.baud_rate_combo.currentText())
            display_port = self.port_combo.currentText()
            self.status_label.setText(f"连接到 {display_port}...")
            self.status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
            
            try:
                # 使用用户选择的波特率（新功能）
                self.reader = SerialReader(port=selected_port, baudrate=selected_baudrate)
                self.reader.data_received.connect(self.handle_line)
                self.reader.connection_ready.connect(lambda: self._on_connection_ready(display_port))
                self.reader.error_occurred.connect(lambda msg: self._on_error_occurred(msg))
                self.reader.start()
                
                self.start_button.setText("Stop")
                self.start_button.setStyleSheet("background-color: #F44336; color: white; font-weight: bold;")
                self.is_running = True
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法启动串口读取器: {str(e)}")
                self.status_label.setText("就绪")
                self.status_label.setStyleSheet("font-weight: bold; color: #2E7D32;")

    def _on_connection_ready(self, port_name):
        """连接成功后的回调"""
        self.status_label.setText(f"已连接到 {port_name}")
        self.status_label.setStyleSheet("font-weight: bold; color: #2E7D32;")

    def _on_error_occurred(self, message):
        """错误发生时的回调"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("font-weight: bold; color: #D32F2F;")

    def on_checkbox_changed(self):
        # 当用户选择或取消选择参数时更新图表
        self.update_plots()

    def handle_line(self, line):
        """改进的数据解析逻辑，支持更多格式和增强的错误处理"""
        try:
            # 预处理行，去除首尾空格和可能的换行符
            line = line.strip()
            if not line:
                return
                
            # 定义支持的分隔符模式
            patterns = [
                r'([A-Za-z0-9]+):\s*(-?\d+(?:\.\d+)?)',  # 基本模式: Key:Value
                r'([A-Za-z0-9]+)\s*=\s*(-?\d+(?:\.\d+)?)',  # = 分隔的模式: Key=Value
                r'([A-Za-z0-9]+)\s*:\s*(-?\d+(?:\.\d+)?)(?:cm|mm|m|°|deg)?',  # 带单位的模式
                r'([A-Za-z0-9]+)\s*=\s*(-?\d+(?:\.\d+)?)(?:cm|mm|m|°|deg)?'  # 带单位的等号模式
            ]
            
            values = {}
            # 尝试所有模式，直到找到匹配的
            for pattern in patterns:
                matches = re.findall(pattern, line)
                if matches:
                    for k, v in matches:
                        try:
                            values[k.strip()] = float(v)
                        except ValueError:
                            # 如果转换失败，记录但继续处理其他值
                            print(f"警告: 无法将值 '{v}' 转换为数字")
                    break
            
            if not values:
                # 只在调试模式下打印无法解析的行
                # print(f"无法解析数据: {line}")
                return
            
            # 添加时间戳和索引
            timestamp = time.time()
            idx = self.sample_indices[-1] + 1 if self.sample_indices else 0
            
            self.sample_indices.append(idx)
            self.timestamps.append(timestamp)
            
            for f in self.fields:
                if f in values:
                    self.data_buffers[f].append(values[f])
                else:
                    # 使用NaN填充缺失值
                    self.data_buffers[f].append(np.nan)
            
            # 更新统计信息
            self.stats_label.setText(f"数据点: {len(self.sample_indices)}")
            
            # 限制更新频率，避免UI卡顿
            current_time = time.time() * 1000  # 转换为毫秒
            if current_time - self.last_update_time > self.update_interval:
                self.update_plots()
                self.last_update_time = current_time
        except Exception as e:
            print(f"处理数据时发生错误: {str(e)}")
            # 不影响程序运行，只记录错误

    def update_plots(self):
        """优化的图表更新函数"""
        try:
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
                ax.set_xlabel("样本索引")
                ax.set_ylabel("值")
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
                field_names = {
                    "D": "距离 (cm)",
                    "fom": "质量因数",
                    "PD01": "相位差01",
                    "PD02": "相位差02",
                    "PD12": "相位差12",
                    "azimuth": "方位角 (°)",
                    "elevation": "仰角 (°)"
                }
                
                ax.set_title(field_names.get(f, f))
                ax.set_ylabel(field_names.get(f, f))
                if i == n:  # 最后一个图表添加x轴标签
                    ax.set_xlabel("样本索引")
                
                # 设置x轴为整数刻度
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                
                if x_data and y_data:
                    # 过滤掉NaN值
                    valid_data = [(x, y) for x, y in zip(x_data, y_data) if not np.isnan(y)]
                    if valid_data:
                        valid_x, valid_y = zip(*valid_data)
                        
                        # 使用更粗的线条和不同颜色
                        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                        color_idx = self.fields.index(f) % len(colors)
                        
                        ax.plot(valid_x, valid_y, label=f, linewidth=1.5, color=colors[color_idx])
                        
                        # 设置合理的X轴范围
                        ax.set_xlim(max(0, valid_x[-1] - self.max_samples), valid_x[-1] + 10)
                        
                        # 动态调整y轴范围
                        if self.auto_scale_checkbox.isChecked():
                            y_min, y_max = min(valid_y), max(valid_y)
                            y_range = y_max - y_min
                            if y_range == 0:  # 所有值都相同
                                y_range = 1.0
                            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
                
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.7)
                
            # 确保图表布局紧凑
            self.figure.tight_layout()
            # 使用draw_idle而不是draw，避免不必要的重绘
            self.canvas.draw_idle()
        except Exception as e:
            print(f"更新图表时发生错误: {str(e)}")
            # 显示错误消息但不中断程序
            QMessageBox.warning(self, "图表更新错误", f"更新图表时发生错误: {str(e)}")
            
    def save_data(self):
        """保存当前数据到CSV文件"""
        if not self.sample_indices:
            QMessageBox.warning(self, "警告", "没有数据可保存！")
            return
        
        # 创建默认文件名
        default_filename = f"CIR_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存数据", default_filename, "CSV文件 (*.csv);;所有文件 (*)")
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                # 写入表头
                headers = ['SampleIndex', 'Timestamp', 'DateTime'] + [f for f in self.fields if self.checkboxes[f].isChecked()]
                f.write(','.join(headers) + '\n')
                
                # 写入数据
                indices = list(self.sample_indices)
                timestamps = list(self.timestamps)
                
                for i in range(len(indices)):
                    # 准备数据行
                    row = [str(indices[i])]
                    
                    # 添加时间戳信息
                    if i < len(timestamps):
                        row.append(str(timestamps[i]))
                        # 添加人类可读的日期时间
                        dt = datetime.fromtimestamp(timestamps[i])
                        row.append(dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
                    else:
                        row.append('')
                        row.append('')
                    
                    # 添加各字段数据
                    for field in headers[3:]:  # 跳过前三个字段
                        if i < len(self.data_buffers[field]) and not np.isnan(self.data_buffers[field][i]):
                            row.append(str(self.data_buffers[field][i]))
                        else:
                            row.append('')
                    
                    # 写入CSV行
                    f.write(','.join(row) + '\n')
            
            QMessageBox.information(self, "成功", f"数据已保存到:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存数据失败: {str(e)}")
            print(f"保存数据错误: {str(e)}")
            
    def clear_data(self):
        """清除所有数据缓冲区"""
        # 弹出确认对话框
        reply = QMessageBox.question(
            self, "确认清除", 
            "确定要清除所有数据吗？此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 重置缓冲区
            for buf in self.data_buffers.values():
                buf.clear()
            self.sample_indices.clear()
            self.timestamps.clear()
            
            # 更新图表
            self.update_plots()
            
            # 更新状态
            self.status_label.setText("数据已清除")
            self.status_label.setStyleSheet("font-weight: bold; color: #757575;")
            self.stats_label.setText("数据点: 0")


if __name__ == "__main__":
    # 捕获未处理的异常
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print(f"未处理的异常: {exc_type.__name__}: {exc_value}")
        import traceback
        traceback.print_tb(exc_traceback)
        
        # 在UI中显示错误消息
        QMessageBox.critical(None, "程序错误", 
                            f"发生未处理的错误:\n{exc_type.__name__}: {exc_value}\n\n"
                            f"请查看控制台输出以获取详细信息。")
    
    sys.excepthook = handle_exception
    
    app = QApplication(sys.argv)
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    win = RealTimeCIRPlot()
    win.resize(1200, 800)
    win.show()
    
    # 添加优雅退出处理
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("程序被用户中断")
        sys.exit(0)