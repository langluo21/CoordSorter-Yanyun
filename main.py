import os
import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import QDateTime  # 仅用于时间戳格式化
from PyQt5.QtGui import QVector3D, QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QCheckBox, QComboBox, QListWidget, QTextEdit, 
                             QLineEdit, QFileDialog, QSplitter, QGroupBox, QDialog, QSizePolicy, 
                             QGridLayout, QStyle)
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.constraint_solver.pywrapcp import RoutingIndexManager, RoutingModel
from pyqtgraph.opengl import GLViewWidget, GLLinePlotItem, GLTextItem, MeshData, GLMeshItem


class CustomGLViewWidget(GLViewWidget):
    """自定义3D视图控件，增强了鼠标交互功能
    
    提供了平移和旋转锁定功能，使用左键平移视图，右键旋转视图
    """
    def __init__(self):
        """初始化3D视图控件，设置默认视图参数"""
        super().__init__()
        self._rotation_locked = False
        self._last_mouse_pos = None
        
        # 设置默认视图参数
        self.opts['center'] = QVector3D(0, 0, 0)  # 视图中心点
        self.opts['azimuth'] = 0                 # 水平旋转角度
        self.opts['elevation'] = 0               # 垂直仰角
        self.opts['distance'] = 10               # 观察距离
        self.opts['rotation'] = QVector3D(0, 0, 0) # 旋转向量
        self.opts['fov'] = 60                    # 视场角

    def mousePressEvent(self, ev):
        """鼠标按下事件处理"""
        self._last_mouse_pos = ev.pos()
        ev.accept()

    def mouseMoveEvent(self, ev):
        """鼠标移动事件处理，实现视图平移和旋转 - 优化版本"""
        if self._last_mouse_pos is None:
            return
        
        # 计算鼠标移动距离
        dx = ev.x() - self._last_mouse_pos.x()
        dy = ev.y() - self._last_mouse_pos.y()
        
        # 如果移动距离太小，忽略以减少不必要的更新
        if abs(dx) < 1 and abs(dy) < 1:
            return

        # 左键平移处理
        if ev.buttons() == Qt.LeftButton:
            # 计算平移速度（与视图距离成比例）
            speed = self.opts['distance'] * 0.002
            
            # 计算平移方向 - 使用缓存的方向向量
            azimuth_rad = np.radians(self.opts['azimuth'])
            # 使用预计算的sin和cos值
            cos_az = np.cos(azimuth_rad + np.pi/2)
            sin_az = np.sin(azimuth_rad + np.pi/2)
            
            right = QVector3D(cos_az, 0, sin_az)
            up = QVector3D(0, 1, 0)  # 固定向上方向
            
            # 计算平移向量并应用
            translate = (right * dx * speed) - (up * dy * speed)
            self.opts['center'] += translate

        # 右键旋转处理
        elif ev.buttons() == Qt.RightButton and not self._rotation_locked:
            # 优化：使用更平滑的旋转速度
            azimuth_delta = dx * 0.5
            elevation_delta = dy * 0.5
            
            self.opts['azimuth'] += azimuth_delta
            # 限制仰角范围在-90到90度之间
            self.opts['elevation'] = max(-90, min(90, self.opts['elevation'] + elevation_delta))

        self._last_mouse_pos = ev.pos()
        self.update()  # 触发视图更新
        ev.accept()


class PathOptimizerApp(QMainWindow):
    """路径优化应用主窗口
    
    提供3D路径可视化和多种路径优化算法，支持文件导入导出功能
    """
    def __init__(self):
        """初始化应用程序主窗口和所有组件"""
        super().__init__()
        
        # 数据相关属性
        self.original_filename = ""  # 存储原始文件名
        self.coordinates = []       # 存储坐标点列表 [(x, y, z, note), ...]
        self.optimized_path = []     # 存储优化后的路径索引
        
        # 视图相关属性
        self.gl_view = CustomGLViewWidget()  # 3D视图控件
        self.original_plot = None    # 原始路径线条对象
        self.optimized_plot = None   # 优化路径线条对象
        self.markers = None          # 坐标点标记对象（合并网格）
        self.selected_point_marker = None  # 选中点的高亮标记
        
        # UI控件引用
        self.chk_original = None     # 原始路径复选框
        
        # 点选择与闪烁效果
        self.blink_timer = QTimer()  # 闪烁计时器
        self.blink_timer.timeout.connect(self.blink_selected_point)
        self.blink_state = False     # 闪烁状态
        self.selected_point_index = None  # 当前选中的点索引
        
        # 初始化界面和视图
        self.init_ui()
        self.init_3d_view()
        self.reset_view()

    def init_ui(self):
        self.setWindowTitle("路径优化工具")
        self.setGeometry(100, 100, 1200, 800)
        
        # 设置全局样式 - 统一的小清新风格
        self.setStyleSheet("""
            /* 全局变量定义 */
            /* 主要颜色 */
            /* 主色调 - 柔和的蓝色 */
            /* 次要色调 - 清新的绿色 */
            /* 背景色 - 淡蓝色背景 */
            /* 文本颜色 - 深灰色 */
            /* 边框颜色 - 淡蓝色 */
            /* 悬停颜色 - 深蓝色 */
            
            /* 全局设置 */
            QMainWindow, QDialog {
                background-color: #f0f7ff;
                font-family: 'Microsoft YaHei', sans-serif;
            }
            
            /* 组件盒子样式 */
            QGroupBox {
                border: 1px solid #e0f0ff;
                border-radius: 10px;
                margin-top: 12px;
                background-color: white;
                padding: 10px;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
            QGroupBox::title {
                color: #4a9cff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                font-weight: bold;
            }
            
            /* 按钮样式 */
            QPushButton {
                background-color: #4a9cff;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 8px;
                font-weight: 500;
                border-bottom: 2px solid rgba(0, 0, 0, 0.1);
                box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #0984e3;
            }
            QPushButton:pressed {
                background-color: #0984e3;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
            
            /* 输入框样式 */
            QLineEdit {
                padding: 8px;
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            QLineEdit:focus {
                border: 1px solid #4a9cff;
                border-bottom: 2px solid #4a9cff;
            }
            
            /* 下拉框样式 */
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                min-height: 22px;
                color: #333;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 14px;
                border-left: none;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #4a9cff;
                margin-right: 6px;
            }
            QComboBox::down-arrow:on {
                top: 1px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #e0f0ff;
                selection-background-color: #e6f2ff;
                selection-color: #333;
                background-color: white;
                padding: 5px;
                border-radius: 8px;
            }
            
            /* 列表控件样式 */
            QListWidget {
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                min-height: 300px;
                min-width: 250px;
                font-size: 12px;
                padding: 8px;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #edf2f7;
                border-radius: 5px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background-color: #4a9cff;
                color: white;
                border-radius: 5px;
            }
            QListWidget::item:hover:!selected {
                background-color: #e6f2ff;
                border-radius: 5px;
            }
            
            /* 文本编辑框样式 */
            QTextEdit {
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                padding: 8px;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
                color: #4b6584;
                font-family: 'Microsoft YaHei', sans-serif;
                font-size: 12px;
                line-height: 1.5;
            }
            
            /* 标签样式 */
            QLabel {
                color: #4b6584;
                font-weight: 500;
                font-size: 13px;
            }
            
            /* 复选框样式 */
            QCheckBox {
                spacing: 6px;
                color: #4b6584;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #e0f0ff;
                border-radius: 4px;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #4a9cff;
                border-color: #4a9cff;
            }
            
            /* 分割器样式 */
            QSplitter::handle {
                background-color: transparent;
                width: 2px;
            }
            
            /* 滚动条样式 */
            QScrollBar:vertical {
                border: none;
                background: #f0f7ff;
                width: 8px;
                margin: 0px 0px 0px 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #c0d6f0;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4a9cff;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #f0f7ff;
                height: 8px;
                margin: 0px 0px 0px 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #c0d6f0;
                min-width: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #4a9cff;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)

        # 主布局
        main_splitter = QSplitter(Qt.Horizontal)
        # 三维视图区域（包含工具栏和视图）
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建浮动工具栏
        toolbar = QWidget()
        toolbar.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                border: 1px solid #e0f0ff;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
            QPushButton {
                background-color: #4a9cff;
                color: white;
                border: none;
                padding: 4px 8px;  /* 减小内边距 */
                border-radius: 8px;
                font-weight: bold;
                min-width: 60px;  /* 减小最小宽度 */
                min-height: 20px; /* 减小最小高度 */
                margin: 0 3px;    /* 减小左右间距 */
                border-bottom: 2px solid rgba(0, 0, 0, 0.1);
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #0984e3;
            }
            QPushButton:pressed {
                background-color: #0984e3;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
            }
            QPushButton#btn_import {
                min-width: 60px;  /* 减小导入按钮最小宽度 */
                min-height: 20px; /* 减小导入按钮最小高度 */
            }
        """)
        
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 5, 10, 5)
        toolbar_layout.setSpacing(8)  # 设置按钮之间的间距
        
        # 添加导入文件按钮到工具栏（与其他按钮保持一致的样式）
        self.btn_import = QPushButton("导入文件")
        self.btn_import.setIcon(self.style().standardIcon(QStyle.SP_FileDialogStart))
        
        # 添加视图控制按钮到工具栏
        self.btn_reset = QPushButton("重置视角")
        self.btn_top = QPushButton("俯视图")
        self.btn_rotation_lock = QPushButton("旋转锁定")
        
        # 按顺序添加所有按钮
        toolbar_layout.addWidget(self.btn_import)
        toolbar_layout.addWidget(self.btn_reset)
        toolbar_layout.addWidget(self.btn_top)
        toolbar_layout.addWidget(self.btn_rotation_lock)
        toolbar_layout.addStretch()
        
        # 添加工具栏和3D视图到容器
        view_layout.addWidget(toolbar)
        view_layout.addWidget(self.gl_view, 1)  # 1表示拉伸因子
        
        # 将视图容器添加到主分割器
        main_splitter.addWidget(view_container)

        # 右侧控制面板
        control_panel = QWidget()
        control_panel.setStyleSheet("""
            QWidget {
                background-color: #f0f7ff;
                border-radius: 10px;
            }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(6)  # 减小主布局的间距
        layout.setContentsMargins(8, 8, 8, 8)  # 减小主布局的内边距

        # 坐标列表 - 直接添加到主布局
        list_container = QWidget()
        list_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
        """)
        list_container_layout = QVBoxLayout(list_container)
        list_container_layout.setContentsMargins(6, 6, 6, 6)  # 减小列表容器的内边距
        list_container_layout.setSpacing(4)  # 减小列表容器内部的间距
        
        # 删除了list_header标签控件
        
        self.list_coords = QListWidget()
        # 设置列表控件的大小策略为扩展
        self.list_coords.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_coords.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                min-height: 320px;
                min-width: 250px;
                font-size: 12px;
                padding: 6px;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #edf2f7;
                margin: 2px 0;
                border-radius: 5px;
            }
            QListWidget::item:selected {
                background-color: #4a9cff;
                color: white;
                border-radius: 5px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
            QListWidget::item:hover:!selected {
                background-color: #e6f2ff;
                border-radius: 5px;
            }
        """)
        list_container_layout.addWidget(self.list_coords)
        
        # 添加到布局并设置拉伸因子为3，使其占据更多空间
        layout.addWidget(list_container, 3)

        # 显示规则
        display_group = QGroupBox("")
        display_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #e0f0ff;
                border-radius: 10px;
                margin-top: 8px;
                background-color: white;
                padding: 6px;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
            }
            QGroupBox::title {
                color: #4a9cff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                font-weight: bold;
            }
        """)
        display_layout = QHBoxLayout()
        display_layout.setSpacing(6)  # 减小显示规则部分的间距
        display_layout.setContentsMargins(5, 3, 5, 3)  # 减小显示规则部分的内边距
        
        self.chk_original = QCheckBox("原始路径")
        self.chk_optimized = QCheckBox("优化路径")
        self.chk_markers = QCheckBox("显示标点")
        
        for chk in [self.chk_original, self.chk_optimized, self.chk_markers]:
            chk.setStyleSheet("""
                QCheckBox {
                    font-size: 13px;
                    color: #4b6584;
                    spacing: 6px;
                    background-color: transparent;
                }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #e0f0ff;
                    border-radius: 3px;
                    background-color: transparent;
                }
                QCheckBox::indicator:checked {
                    background-color: #4a9cff;
                    border-color: #4a9cff;
                }
            """)
            
        display_layout.addWidget(self.chk_original)
        display_layout.addWidget(self.chk_optimized)
        display_layout.addWidget(self.chk_markers)
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 优化设置
        opt_group = QGroupBox("")
        opt_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #e0f0ff;
                border-radius: 10px;
                margin-top: 8px;
                background-color: white;
                padding: 6px;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
            }
            QGroupBox::title {
                color: #4a9cff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                font-weight: bold;
            }
        """)
        opt_layout = QVBoxLayout()
        opt_layout.setSpacing(4)  # 减小优化设置部分的间距
        opt_layout.setContentsMargins(5, 5, 5, 5)  # 减小优化设置部分的内边距
        
        # 移除优化算法标签，但保留下拉框功能
        
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems([
            "平面优化 (计算高效)",
            "智能优化 (OR-Tools)",
            "快速优化 (最近邻算)",
            "全局优化 (模拟退火)"
        ])
        self.combo_algorithm.setStyleSheet("""
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: transparent;
                min-height: 22px;
                color: #333;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 14px;
                border-left: none;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #4a9cff;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #e0f0ff;
                selection-background-color: #e6f2ff;
                selection-color: #333;
                background-color: transparent;
                padding: 5px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
        """)
        
        btn_optimize = QPushButton("开始优化")
        btn_optimize.setStyleSheet("""
            QPushButton {
                background-color: #4a9cff;
                color: white;
                font-weight: bold;
                min-height: 30px;
                padding: 6px 12px;
                border-radius: 8px;
                border-bottom: 2px solid rgba(0, 0, 0, 0.1);
                box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
            }
            QPushButton:hover {
                background-color: #0984e3;
                color: white;
            }
            QPushButton:pressed {
                background-color: #0984e3;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
            }
        """)
        
        # 直接添加下拉框，不添加标签
        opt_layout.addWidget(self.combo_algorithm)
        opt_layout.addSpacing(5)
        opt_layout.addWidget(btn_optimize)
        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)

        # 日志显示
        log_container = QWidget()
        log_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
        """)
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(6, 6, 6, 6)  # 减小列表容器的内边距
        log_layout.setSpacing(4)  # 减小列表容器内部的间距
        
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.log_view.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                min-height: 320px;
                min-width: 250px;
                font-size: 12px;
                padding: 6px;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
                color: #4b6584;
                font-family: 'Microsoft YaHei', sans-serif;
            }
        """)
        
        log_layout.addWidget(self.log_view)
        
        # 添加到布局并设置拉伸因子为3，使其占据更多空间
        layout.addWidget(log_container, 3)

        # 导出设置
        export_group = QGroupBox("")
        export_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #e0f0ff;
                border-radius: 10px;
                margin-top: 12px;
                background-color: white;
                padding: 10px;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
            }
            QGroupBox::title {
                color: #4a9cff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                font-weight: bold;
            }
        """)
        export_layout = QGridLayout()
        export_layout.setVerticalSpacing(8)
        export_layout.setHorizontalSpacing(10)
        export_layout.setContentsMargins(8, 8, 8, 8)
        self.edit_filename = QLineEdit("待导入文件")
        self.edit_filename.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                border-bottom: 1px solid #e0f0ff;
            }
        """)
        
        self.combo_encoding = QComboBox()
        self.combo_encoding.addItems(["ANSI", "GB18030", "UTF-8"])
        self.combo_encoding.setStyleSheet("""
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                min-height: 22px;
                color: #333;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 14px;
                border-left: none;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #4a9cff;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #e0f0ff;
                selection-background-color: #e6f2ff;
                selection-color: #333;
                background-color: transparent;
                padding: 5px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
        """)
        
        self.combo_format = QComboBox()
        self.combo_format.addItems(["TXT", "INI"])
        self.combo_format.setStyleSheet("""
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #e0f0ff;
                border-radius: 8px;
                background-color: white;
                min-height: 22px;
                color: #333;
                border-bottom: 1px solid #e0f0ff;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 14px;
                border-left: none;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #4a9cff;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #e0f0ff;
                selection-background-color: #e6f2ff;
                selection-color: #333;
                background-color: transparent;
                padding: 5px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
        """)
        
        btn_save = QPushButton("导出文件")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #ff7675;
                color: white;
                font-weight: bold;
                min-height: 28px;
                padding: 6px 10px;
                border-radius: 8px;
                border-bottom: 2px solid rgba(0, 0, 0, 0.1);
                margin-top: 5px;
            }
            QPushButton:hover {
                background-color: #e84393;
            }
            QPushButton:pressed {
                background-color: #e84393;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            }
        """)

        # 创建标签并设置样式
        file_name_label = QLabel("文件名称：")
        file_encoding_label = QLabel("文件编码：")
        file_format_label = QLabel("文件格式：")
        
        for label in [file_name_label, file_encoding_label, file_format_label]:
            label.setStyleSheet("""
                QLabel {
                    color: #4b6584;
                    font-weight: 500;
                    font-size: 13px;
                }
            """)
        
        export_layout.addWidget(file_name_label, 0, 0)
        export_layout.addWidget(self.edit_filename, 0, 1)
        export_layout.addWidget(file_encoding_label, 1, 0)
        export_layout.addWidget(self.combo_encoding, 1, 1)
        export_layout.addWidget(file_format_label, 2, 0)
        export_layout.addWidget(self.combo_format, 2, 1)
        export_layout.addWidget(btn_save, 3, 0, 1, 2)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        control_panel.setLayout(layout)
        main_splitter.addWidget(control_panel)
        self.setCentralWidget(main_splitter)

        # 设置分割比例
        main_splitter.setStretchFactor(0, 7)  # 3D视图占70%
        main_splitter.setStretchFactor(1, 3)  # 控制面板占30%
        
        # 设置分割器样式
        main_splitter.setStyleSheet("""
            QSplitter {
                background-color: #f0f7ff;
            }
            QSplitter::handle {
                background-color: transparent;
                width: 2px;
                margin: 2px 2px;
                border-radius: 1px;
            }
        """)

        # 信号连接
        self.chk_original.stateChanged.connect(self.toggle_original_path)
        self.chk_optimized.stateChanged.connect(self.toggle_optimized_path)
        self.chk_markers.stateChanged.connect(self.toggle_markers)
        btn_optimize.clicked.connect(self.run_optimization)
        btn_save.clicked.connect(self.export_file)
        self.btn_reset.clicked.connect(self.reset_view)
        self.btn_top.clicked.connect(self.set_top_view)
        self.btn_rotation_lock.clicked.connect(self.lock_rotation)
        self.list_coords.itemSelectionChanged.connect(self.on_coord_selected)
        self.btn_import.clicked.connect(self.import_file)

    def init_3d_view(self):
        """初始化三维视图（仿手绘地图风格）"""
        self.gl_view.setBackgroundColor('w')  # 白色背景
        x_range = (-5500, 2000)
        y_range = (-4000, 3000)
        grid_step = 500  # 保持网格间距500单位

        # 创建淡蓝灰色半透明网格系统
        grid_color = (0.27, 0.51, 0.71, 0.2)  # 更淡的RGBA蓝色系

        # 水平网格线 (X轴方向)
        for y in np.arange(y_range[0], y_range[1] + grid_step, grid_step):
            line = GLLinePlotItem(
                pos=np.array([[x_range[0], y, 0], [x_range[1], y, 0]]),
                color=grid_color,
                width=1.2,  # 细线宽
                antialias=True
            )
            self.gl_view.addItem(line)

            # 在右侧边缘标注Y坐标
            self.gl_view.addItem(GLTextItem(
                pos=(x_range[1] + 50, y, 0),
                text=str(int(y)),
                color=(0.3, 0.3, 0.3, 0.6),  # 更淡的灰色半透明
                font=QFont('Arial', 9)
            ))

        # 垂直网格线 (Y轴方向)
        for x in np.arange(x_range[0], x_range[1] + grid_step, grid_step):
            line = GLLinePlotItem(
                pos=np.array([[x, y_range[0], 0], [x, y_range[1], 0]]),
                color=grid_color,
                width=1.2,
                antialias=True
            )
            self.gl_view.addItem(line)

            # 在上方边缘标注X坐标
            self.gl_view.addItem(GLTextItem(
                pos=(x, y_range[1] + 50, 0),
                text=str(int(x)),
                color=(0.3, 0.3, 0.3, 0.6),
                font=QFont('Arial', 9)
            ))

        # 添加方向指示标签（仿地图图例）
        legend_config = {
            'color': (0.2, 0.2, 0.2, 0.7),
            'font': QFont('Arial', 11, QFont.Bold)
        }
        self.gl_view.addItem(GLTextItem(
            pos=(x_range[1] - 500, y_range[1] + 200, 0),
            text="→ 东向坐标",
         ** legend_config
        ))
        self.gl_view.addItem(GLTextItem(
            pos=(x_range[0] + 200, y_range[1] - 300, 0),
            text="↑ 北向坐标",
        ** legend_config
        ))

    def reset_view(self):
        """重置到默认视角"""
        # 计算中心点
        center_x = (2000 + (-5500)) / 2  # x_range的中点
        center_y = (3000 + (-4000)) / 2  # y_range的中点
        
        self.gl_view.setCameraPosition(
            pos=QVector3D(center_x+1000, center_y, 0),  # 相机位置在中心点下方
            distance=6000,  # 观察距离足够看到整个区域
            elevation=60,   # 俯角30度
            azimuth=0    # 水平旋转-45度
        )
        self.log("视图已重置到默认视角")

    def set_top_view(self):
        """设置为俯视图"""
        self.gl_view.setCameraPosition(
            distance=7000,
            elevation=90,
            azimuth=0
        )
        self.log("已切换到俯视图")

    def lock_rotation(self):
        """切换旋转锁定状态"""
        try:
            is_locked = getattr(self.gl_view, '_rotation_locked', False)
            if is_locked:
                self.gl_view._rotation_locked = False
                self.btn_rotation_lock.setText("旋转锁定")
                self.log("旋转锁定已禁用")
            else:
                self.gl_view._rotation_locked = True
                self.btn_rotation_lock.setText("解锁旋转")
                self.log("旋转锁定已启用")
        except Exception as e:
            self.log(f"旋转锁定操作失败: {str(e)}")

    # 移除了未使用的lock_pan、setModelview和viewMatrix方法

    def log(self, message):
        """记录日志"""
        
        # 只显示分钟和秒数
        timestamp = QDateTime.currentDateTime().toString("mm:ss")
        self.log_view.append(f"[{timestamp}] {message}")

    def import_file(self):
        """导入坐标文件 - 优化版本"""
        # 先清理3D视图中的所有对象
        self._clear_3d_view()
        
        # 打开文件对话框
        filename, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "文本文件 (*.txt;*.ini)")
        if not filename:
            return
            
        try:
            # 记录原始文件名（不含扩展名）
            self.original_filename = os.path.splitext(os.path.basename(filename))[0]
            # 更新导出文件名输入框
            self.edit_filename.setText(f"{self.original_filename}-优化版")
            
            # 解析文件内容
            self._parse_coordinate_file(filename)
            
            # 更新UI和3D视图
            self.update_coord_list()
            self.plot_points()
            self.log(f"成功导入文件: {filename}，共{len(self.coordinates)}个坐标点")

        except Exception as e:
            self.log(f"文件导入失败: {str(e)}")
            # 只在调试模式下显示详细错误信息
            import traceback
            self.log(traceback.format_exc())
    
    def _clear_3d_view(self):
        """清理3D视图中的所有对象 - 从import_file中提取的辅助方法"""
        # 清除原始路径
        if self.original_plot:
            self.gl_view.removeItem(self.original_plot)
            self.original_plot = None

        # 清除优化路径
        if self.optimized_plot:
            self.gl_view.removeItem(self.optimized_plot)
            self.optimized_plot = None

        # 清除标记点
        if self.markers:
            self.gl_view.removeItem(self.markers)
            self.markers = None
            
        # 清除选中点高亮标记
        if self.selected_point_marker:
            self.gl_view.removeItem(self.selected_point_marker)
            self.selected_point_marker = None

        # 清理所有其他GLMeshItem对象
        items_to_remove = [item for item in self.gl_view.items if isinstance(item, GLMeshItem)]
        for item in items_to_remove:
            self.gl_view.removeItem(item)
    
    def _parse_coordinate_file(self, filename):
        """解析坐标文件 - 从import_file中提取的辅助方法"""
        self.coordinates = []
        with open(filename, 'r', encoding='ansi') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        z = float(parts[2].strip())
                        note = parts[3].strip()
                        self.coordinates.append((x, y, z, note))
                    except ValueError:
                        self.log(f"第{line_num}行数据格式错误: {line}")

    def update_coord_list(self):
        """更新坐标列表显示"""
        self.list_coords.clear()
        for idx, coord in enumerate(self.coordinates):
            self.list_coords.addItem(f"[{idx + 1}] {coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f} - {coord[3]}")

    def plot_points(self):
        """绘制三维点 - 优化版本"""
        # 清除旧图形
        if self.original_plot:
            self.gl_view.removeItem(self.original_plot)
        
        # 清除旧标点
        if self.markers:
            self.gl_view.removeItem(self.markers)
            self.markers = None

        if not self.coordinates:
            return
            
        # 限制最大点数
        MAX_POINTS = 3000
        if len(self.coordinates) > MAX_POINTS:
            self.log(f"警告：点数超过{MAX_POINTS}，将只显示前{MAX_POINTS}个点")
            self.coordinates = self.coordinates[:MAX_POINTS]

        # 提前计算点坐标数组，避免重复计算
        points = np.array([[x, y, z] for x, y, z, _ in self.coordinates])
        
        # 绘制路径线
        self.original_plot = GLLinePlotItem(
            pos=points,
            color=(0.9, 0.3, 0.3, 0.35),  # 柔和的红色，降低透明度
            width=2,
            antialias=True
        )
        self.gl_view.addItem(self.original_plot)
        
        # 自动勾选原始路径复选框
        if self.chk_original and not self.chk_original.isChecked():
            self.chk_original.setChecked(True)

        # 只在需要时创建标点
        if self.chk_markers.isChecked():
            self._create_markers(points)
    
    def _create_markers(self, points):
        """创建标记点 - 从plot_points中提取的辅助方法"""
        # 优化：降低球体精度，减少顶点数量
        sphere_data = MeshData.sphere(rows=5, cols=10)  # 进一步降低球体精度
        base_verts = sphere_data.vertexes()
        base_faces = sphere_data.faces()
        
        # 预分配数组空间，避免频繁扩展
        n_points = len(points)
        n_verts_per_sphere = len(base_verts)
        n_faces_per_sphere = len(base_faces)
        
        vertices = np.zeros((n_points * n_verts_per_sphere, 3))
        faces = np.zeros((n_points * n_faces_per_sphere, 3), dtype=int)
        colors = np.zeros((n_points * n_verts_per_sphere, 4))
        
        scale = 15  # 球体大小
        
        # 批量处理所有点，避免Python循环
        for i in range(n_points):
            # 计算顶点偏移
            vert_offset = i * n_verts_per_sphere
            face_offset = i * n_faces_per_sphere
            
            # 设置顶点位置
            vertices[vert_offset:vert_offset+n_verts_per_sphere] = base_verts * scale + points[i]
            
            # 设置面索引
            faces[face_offset:face_offset+n_faces_per_sphere] = base_faces + vert_offset
            
            # 设置颜色
            colors[vert_offset:vert_offset+n_verts_per_sphere] = [0.9, 0.3, 0.3, 0.5]
        
        # 创建合并的网格对象
        merged_mesh = MeshData(
            vertexes=vertices,
            faces=faces,
            vertexColors=colors
        )
        
        self.markers = GLMeshItem(
            meshdata=merged_mesh,
            smooth=True,
            glOptions='translucent'
        )
        self.gl_view.addItem(self.markers)

    def toggle_original_path(self, state):
        """切换原始路径显示"""
        if self.original_plot:
            self.original_plot.setVisible(state == Qt.Checked)

    def toggle_optimized_path(self, state):
        """切换优化路径显示"""
        if self.optimized_plot:
            self.optimized_plot.setVisible(state == Qt.Checked)

    def toggle_markers(self, state):
        """切换坐标点标记显示"""
        if state == Qt.Checked:
            # 如果选中了复选框但markers不存在，重新创建标记点
            if not self.markers and self.coordinates:
                self.plot_points()
            # 如果markers存在，设置为可见
            elif self.markers:
                self.markers.setVisible(True)
        else:
            # 如果取消选中复选框且markers存在，设置为不可见
            if self.markers:
                self.markers.setVisible(False)
            # 停止闪烁
            self.blink_timer.stop()
            if self.selected_point_marker:
                self.gl_view.removeItem(self.selected_point_marker)
                self.selected_point_marker = None

    def on_coord_selected(self):
        """处理坐标点选择事件"""
        if not self.coordinates or not self.chk_markers.isChecked():
            return

        # 停止之前的闪烁
        self.blink_timer.stop()
        if self.selected_point_marker:
            self.gl_view.removeItem(self.selected_point_marker)
            self.selected_point_marker = None

        # 获取选中项
        current_item = self.list_coords.currentItem()
        if current_item:
            # 从项目文本中提取索引（格式：[1] x, y, z - note）
            index = int(current_item.text().split(']')[0].strip('[')) - 1
            self.selected_point_index = index
            
            # 创建高亮标记
            point = self.coordinates[index][:3]  # 获取x, y, z坐标
            self.create_highlight_marker(point)
            
            # 开始闪烁
            self.blink_timer.start(500)  # 每500毫秒闪烁一次

    def create_highlight_marker(self, point):
        """创建高亮标记"""
        if self.selected_point_marker:
            self.gl_view.removeItem(self.selected_point_marker)

        # 创建较大的深灰色球体作为高亮标记
        sphere = MeshData.sphere(rows=10, cols=20)
        self.selected_point_marker = GLMeshItem(
            meshdata=sphere,
            smooth=True,
            color=(0.2, 0.2, 0.2, 0.6),  # 改为深灰色，降低不透明度
            glOptions='translucent'
        )
        self.selected_point_marker.scale(25, 25, 25)  # 比普通标记大一倍
        self.selected_point_marker.translate(point[0], point[1], point[2])
        self.gl_view.addItem(self.selected_point_marker)

    def blink_selected_point(self):
        """闪烁效果"""
        if self.selected_point_marker:
            self.blink_state = not self.blink_state
            self.selected_point_marker.setVisible(self.blink_state)

    def _check_coordinates(self):
        """检查坐标点是否足够进行优化"""
        if len(self.coordinates) < 2:
            self.log("错误：需要至少2个坐标点才能进行优化")
            return False
        return True
        
    def _update_progress_label(self, progress, text):
        """更新进度对话框的文本"""
        if progress:
            label = progress.findChild(QLabel)
            if label:
                label.setText(text)
                QApplication.processEvents()
    
    def _draw_optimized_path(self, path, color=(0, 1, 0, 0.4)):
        """绘制优化后的路径
        
        Args:
            path: 路径点索引列表
            color: 路径颜色，默认为半透明绿色
        """
        # 移除旧路径
        if self.optimized_plot:
            self.gl_view.removeItem(self.optimized_plot)
        
        # 获取路径点坐标
        points = np.array([self.coordinates[i][:3] for i in path])
        
        # 创建并添加新路径
        self.optimized_plot = GLLinePlotItem(
            pos=points,
            color=color,
            width=3,
            antialias=True
        )
        self.gl_view.addItem(self.optimized_plot)
        
        # 自动勾选优化路径复选框
        if self.chk_optimized and not self.chk_optimized.isChecked():
            self.chk_optimized.setChecked(True)
    
    def _calculate_distance_3d(self, points, path):
        """计算3D路径的总距离
        
        Args:
            points: 坐标点数组
            path: 路径点索引列表
            
        Returns:
            float: 路径总距离
        """
        return sum(np.linalg.norm(points[path[i]] - points[path[i-1]]) 
                  for i in range(1, len(path)))
    
    def _log_optimization_result(self, path, distance, algorithm_name=""):
        """记录优化结果
        
        Args:
            path: 优化后的路径
            distance: 路径总距离
            algorithm_name: 算法名称
        """
        prefix = f"{algorithm_name}" if algorithm_name else "优化"
        self.log(f"{prefix}完成！总距离: {distance:.2f} 米")
        self.log(f"优化路径顺序: {[i + 1 for i in path]}")
    
    def run_optimization(self):
        """运行路径优化 - 优化版本"""
        # 检查是否有足够的坐标点
        if not self._check_coordinates():
            return
            
        # 创建进度对话框
        progress = self._create_progress_dialog("正在优化中")
        
        try:
            # 获取当前选择的优化算法
            algorithm = self.combo_algorithm.currentText()
            
            # 根据选择的算法执行相应的优化函数
            if algorithm == "平面优化 (计算高效)":
                self.optimize_xy_only(progress)
            elif algorithm == "智能优化 (OR-Tools)":
                self.optimize_with_ortools(progress)
            elif algorithm == "快速优化 (最近邻算)":
                self.optimize_with_nearest_neighbor(progress)
            elif algorithm == "全局优化 (模拟退火)":
                self.optimize_with_simulated_annealing(progress)
            else:
                self.log(f"未知的优化算法: {algorithm}")
        except Exception as e:
            self.log(f"优化过程中发生错误: {str(e)}")
        finally:
            # 关闭进度对话框
            progress.close()
    
    def _create_progress_dialog(self, message):
        """创建进度对话框 - 从run_optimization中提取的辅助方法"""
        # 创建无标题栏的简洁提示对话框
        progress = QDialog(self)
        # 设置无边框窗口
        progress.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        progress.setWindowModality(Qt.WindowModal)
        progress.setFixedSize(300, 100)
        progress.setStyleSheet("""
            QDialog {
                background-color: #f0f7ff;
                border-radius: 10px;
                border: 1px solid #e0f0ff;
                border-bottom: 2px solid #e0f0ff;
                border-right: 2px solid #e0f0ff;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
            QLabel {
                color: #4b6584;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        # 创建垂直布局
        layout = QVBoxLayout(progress)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 添加提示文本
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # 显示对话框
        progress.show()
        # 处理所有挂起的Qt事件，确保对话框立即显示
        QApplication.processEvents()
        
        return progress

    def optimize_with_ortools(self, progress=None):
        """使用OR-Tools进行3D路径优化
        
        Args:
            progress: 进度对话框对象，可选
        """
        if not self._check_coordinates():
            return

        self.log("开始OR-Tools 3D路径优化...")
        QApplication.processEvents()  # 更新界面
        
        # 更新提示信息
        self._update_progress_label(progress, "正在计算3D距离矩阵...")

        try:
            # 创建距离矩阵 - 使用缓存的点数组避免重复计算
            locations = np.array([[x, y, z] for x, y, z, _ in self.coordinates])
            manager = RoutingIndexManager(len(locations), 1, 0)
            routing = RoutingModel(manager)

            # 优化距离计算回调函数 - 使用向量化操作
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                # 使用预计算的欧几里得距离，并缩放为整数
                return int(np.linalg.norm(locations[from_node] - locations[to_node]) * 100)

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # 设置搜索参数
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.time_limit.seconds = 30

            # 求解
            self._update_progress_label(progress, "正在求解优化路径...")
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                # 提取优化路径
                index = routing.Start(0)
                path = []
                while not routing.IsEnd(index):
                    path.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                
                self.optimized_path = path
                
                # 绘制优化路径 - 使用辅助函数
                self._draw_optimized_path(path, color=(0, 0.8, 0, 0.35))

                # 计算并记录总距离
                total_distance = solution.ObjectiveValue() / 100
                self._log_optimization_result(path, total_distance, "OR-Tools 3D优化")
            else:
                self.log("优化失败，请检查输入数据")

        except Exception as e:
            self.log(f"OR-Tools优化过程中发生错误: {str(e)}")

    def optimize_xy_only(self, progress=None):
        """仅考虑X-Y平面的优化
        
        Args:
            progress: 进度对话框对象，可选
        """
        if not self._check_coordinates():
            return
            
        self.log("开始平面优化...")
        self._update_progress_label(progress, "正在计算2D距离矩阵...")
        
        try:
            # 创建2D坐标数组 - 只提取x和y坐标
            locations_2d = np.array([[x, y] for x, y, _, _ in self.coordinates])
            manager = RoutingIndexManager(len(locations_2d), 1, 0)
            routing = RoutingModel(manager)

            # 优化的2D距离计算 - 使用向量化操作
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                # 直接计算欧几里得距离，避免额外的函数调用
                return int(np.linalg.norm(locations_2d[from_node] - locations_2d[to_node]) * 100)

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            # 设置搜索参数
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.time_limit.seconds = 30

            # 求解
            self._update_progress_label(progress, "正在求解平面优化路径...")
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                # 提取优化路径
                index = routing.Start(0)
                path = []
                while not routing.IsEnd(index):
                    path.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                
                self.optimized_path = path
                
                # 绘制优化路径
                self._draw_optimized_path(path)

                # 计算并记录总距离
                total_distance = solution.ObjectiveValue() / 100
                self._log_optimization_result(path, total_distance, "平面优化")
            else:
                self.log("平面优化失败，请检查输入数据")

        except Exception as e:
            self.log(f"平面优化过程中发生错误: {str(e)}")

    def optimize_with_nearest_neighbor(self, progress=None):
        """使用最近邻算法进行路径优化
        
        Args:
            progress: 进度对话框对象，可选
        """
        if not self._check_coordinates():
            return

        try:
            self.log("开始最近邻算法优化...")
            self._update_progress_label(progress, "正在计算最近邻路径...")
            
            # 创建坐标点数组 - 预计算以提高性能
            n = len(self.coordinates)
            points = np.array([[x, y, z] for x, y, z, _ in self.coordinates])
            
            # 初始化路径
            unvisited = list(range(1, n))  # 0是起点
            path = [0]
            current = 0

            # 贪心选择最近的点 - 优化版本
            while unvisited:
                # 使用向量化操作计算距离 - 性能优化
                current_point = points[current]
                # 一次性计算所有距离，避免循环
                distances = np.array([np.linalg.norm(current_point - points[i]) for i in unvisited])
                # 找到最近的点
                min_idx = np.argmin(distances)
                nearest_idx = unvisited[min_idx]
                path.append(nearest_idx)
                # 使用pop而不是remove以提高性能
                unvisited.pop(min_idx)
                current = nearest_idx

            self.optimized_path = path
            
            # 绘制优化路径 - 使用辅助函数
            self._draw_optimized_path(path)

            # 计算总距离并记录结果
            total_distance = self._calculate_distance_3d(points, path)
            self._log_optimization_result(path, total_distance, "最近邻算法")

        except Exception as e:
            self.log(f"最近邻优化过程中发生错误: {str(e)}")

    def optimize_with_simulated_annealing(self, progress=None):
        """使用模拟退火算法进行全局优化
        
        Args:
            progress: 进度对话框对象，可选
        """
        if not self._check_coordinates():
            return

        try:
            self.log("开始模拟退火优化...")
            self._update_progress_label(progress, "正在进行模拟退火全局优化...")
            
            # 预计算坐标点数组
            points = np.array([[x, y, z] for x, y, z, _ in self.coordinates])
            n = len(points)

            # 优化的距离计算函数 - 使用缓存避免重复计算
            def calculate_distance(path):
                # 计算路径总距离（包括回到起点的距离）
                path_distance = self._calculate_distance_3d(points, path)
                # 添加从终点回到起点的距离
                return path_distance + np.linalg.norm(points[path[-1]] - points[path[0]])

            # 初始化参数 - 使用更高效的冷却策略
            initial_temp = 100.0
            final_temp = 0.01
            alpha = 0.95  # 冷却速率
            iterations_per_temp = max(100, n * 2)  # 根据点数调整迭代次数

            # 初始解
            current_path = list(range(n))
            current_distance = calculate_distance(current_path)
            best_path = current_path.copy()
            best_distance = current_distance
            
            # 模拟退火主循环 - 优化版本
            temp = initial_temp
            iteration_count = 0
            max_iterations = int(np.log(final_temp / initial_temp) / np.log(alpha)) * iterations_per_temp
            
            while temp > final_temp:
                for iter_idx in range(iterations_per_temp):
                    # 更新进度信息
                    if iter_idx % 10 == 0:
                        progress_pct = min(100, int(100 * iteration_count / max_iterations))
                        self._update_progress_label(progress, f"模拟退火优化中...({progress_pct}%)")
                        iteration_count += 1
                    
                    # 使用2-opt邻域操作 - 随机选择两个点进行路径段反转
                    i, j = sorted(np.random.randint(0, n, 2))
                    if i == j:  # 避免无效操作
                        continue
                        
                    # 创建新路径 - 使用切片反转以提高效率
                    new_path = current_path.copy()
                    new_path[i:j+1] = reversed(new_path[i:j+1])
                    
                    # 计算新路径距离
                    new_distance = calculate_distance(new_path)

                    # 计算接受概率 - Metropolis准则
                    delta = new_distance - current_distance
                    if delta < 0 or np.random.random() < np.exp(-delta / temp):
                        current_path = new_path  # 无需复制，直接赋值
                        current_distance = new_distance
                        # 更新最优解
                        if current_distance < best_distance:
                            best_path = current_path.copy()
                            best_distance = current_distance

                # 降温
                temp *= alpha

            # 保存优化路径
            self.optimized_path = best_path

            # 绘制优化路径 - 使用辅助函数
            self._draw_optimized_path(best_path)

            # 记录优化结果
            self._log_optimization_result(best_path, best_distance, "模拟退火算法")

        except Exception as e:
            self.log(f"模拟退火优化过程中发生错误: {str(e)}")

    def export_file(self):
        """导出优化结果"""
        if not self.optimized_path:
            self.log("错误：请先进行路径优化")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "保存文件",
            self.edit_filename.text(),
            f"{self.combo_format.currentText()}文件 (*.{self.combo_format.currentText().lower()})"
        )

        if filename:
            try:
                encoding = self.combo_encoding.currentText()
                with open(filename, 'w', encoding=encoding) as f:
                    for idx in self.optimized_path:
                        x, y, z, note = self.coordinates[idx]
                        f.write(f"{x:.6f},{y:.6f},{z:.6f},{note}\n")
                self.log(f"成功导出文件: {filename}")
            except Exception as e:
                self.log(f"文件导出失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PathOptimizerApp()
    window.show()
    sys.exit(app.exec_())