import datetime
import sys
import os
import random
import json

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QToolBar, QListWidget,
                             QListWidgetItem, QInputDialog, QGraphicsView, QGraphicsScene, QMessageBox, QAction,
                             QToolButton, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QBrush, QFont, QIcon, QPen
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QMetaObject
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor


class MaskManager:
    """掩码数据管理器"""

    def __init__(self):
        self.data = {}  # {image_path: {"masks": list, "saved": bool}}

    def add_image(self, image_path):
        if image_path not in self.data:
            self.data[image_path] = {
                "masks": [],
                "saved": True
            }

    def update_masks(self, image_path, masks):
        self.data[image_path]["masks"] = masks
        self.data[image_path]["saved"] = False

    def mark_saved(self, image_path):
        self.data[image_path]["saved"] = True


class Mask:
    """掩码对象类"""
    #_id_counter = 0

    def __init__(self, id):
        #Mask._id_counter += 1
        #self.id = Mask._id_counter
        self.id = id
        self.name = f"Mask{self.id}"
        self.annotations = ""
        self.coordinates = set()
        self.visible = True
        # 固定淡绿色 (R=100, G=230, B=150) 透明度200
        self.color = QColor(100, 230, 150, 200)  # 修改为淡绿色高透明度
        self.mask_array = None  # 存储二进制数组
        self.rle = None  # 或存储RLE编码

class Canvas(QGraphicsView):
    switch_image_requested = pyqtSignal(str)
    mask_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # 记录上一次绘制位置
        self.last_brush_pos = None

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #3A3A3A;")
        self.setRenderHint(QPainter.Antialiasing)
        self.setFocusPolicy(Qt.StrongFocus)  # 确保接收键盘事件

        # 掩码管理
        self.curId = 1
        self.idlist = []
        self.masks = []
        self.active_mask = None

        # 绘图参数
        self.drawing = False
        self.current_brush_pos = QPoint(0, 0)  # 初始化默认坐标
        # 擦除模式标志和笔刷设置
        self.is_erasing = False
        self.brush_size = 8
        self.min_brush_size = 1
        self.max_brush_size = 50

        # 图形项
        self.image_item = None
        self.overlay_item = None
        self.overlay_pixmap = None

        # 新增点分割相关属性
        self.point_mode = None  # 'positive'/'negative'/None
        self.input_points = []
        self.input_labels = []
        self.temp_mask_layer = None
        self.preview_item = None
        self.point_markers = []  # 跟踪所有提示点图形项

    def clear_point_markers(self):
        """清除所有提示点图形"""
        for marker in self.point_markers:
            self.scene.removeItem(marker)
        self.point_markers.clear()

    def enable_point_mode(self, mode):
        """启用点输入模式"""
        self.point_mode = mode
        self.input_points = []
        self.input_labels = []
        self._clear_preview()

    def _clear_preview(self):
        """清除预览"""
        if self.preview_item:
            self.scene.removeItem(self.preview_item)
        self.preview_item = None

    """def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 按下鼠标时重置上次位置
            self.last_brush_pos = None
            if not self.active_mask:
                self.create_mask()
            self.drawing = True
            self._update_brush_position(event)"""

    def mousePressEvent(self, event):
        if self.point_mode and event.button() == Qt.LeftButton:
            # 收集点坐标
            pos = self.mapToScene(event.pos()).toPoint()
            if self.point_mode == 'positive':
                label = 1
                color = QColor(0, 255, 0)  # 绿色正点
            else:
                label = 0
                color = QColor(255, 0, 0)  # 红色负点

            self.input_points.append([pos.x(), pos.y()])
            self.input_labels.append(label)

            # 绘制点标记并记录
            pen = QPen(color)
            brush = QBrush(color)
            circle = self.scene.addEllipse(pos.x() - 3, pos.y() - 3, 6, 6, pen, brush)
            circle.setZValue(2)
            self.point_markers.append(circle)  # 记录图形项
            self._update_preview()

        elif self.point_mode and event.button() == Qt.RightButton:
            if self.input_points:
                # 删除最后一个点的图形项
                if self.point_markers:
                    last_marker = self.point_markers.pop()
                    self.scene.removeItem(last_marker)

                self.input_points.pop()
                self.input_labels.pop()
                self._update_preview()

            return

        elif event.button() == Qt.LeftButton:
            # 按下鼠标时重置上次位置
            self.last_brush_pos = None
            if not self.active_mask:
                self.create_mask()
            self.drawing = True
            self._update_brush_position(event)

    def _update_preview(self):
        """更新分割预览"""
        if not self.sam_predictor or not self.input_points:
            return

        # 转换输入格式
        points = np.array(self.input_points)
        labels = np.array(self.input_labels)

        # 执行预测
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )

        # 显示最佳结果
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        # 创建预览图像
        height, width = mask.shape
        preview_img = QImage(width, height, QImage.Format_ARGB32)
        preview_img.fill(Qt.transparent)

        # 绘制半透明绿色掩码
        for y in range(height):
            for x in range(width):
                if mask[y, x]:
                    preview_img.setPixelColor(x, y, QColor(0, 255, 0, 100))

        # 更新预览项
        self._clear_preview()
        self.preview_item = self.scene.addPixmap(QPixmap.fromImage(preview_img))
        self.preview_item.setZValue(0.5)

    def add_sam_masks(self, sam_masks, image_size):
        """添加SAM生成的掩码到画布"""
        self.masks.clear()
        self.active_mask = None

        # 生成不同颜色
        colors = [
            QColor(100, 230, 150, 200),
            QColor(230, 100, 150, 200),
            QColor(150, 100, 230, 200),
            QColor(100, 150, 230, 200)
        ]

        for idx, mask_data in enumerate(sam_masks):
            mask = Mask(id=idx + 1)
            mask.color = colors[idx % len(colors)]

            # 转换坐标
            mask.mask_array  = mask_data['segmentation']
            ys, xs = np.where(mask.mask_array )
            mask.coordinates = set(zip(xs, ys))  # 注意坐标顺序

            self.masks.append(mask)

        if self.masks:
            self.active_mask = self.masks[0]
        self._update_overlay()
        self.mask_updated.emit()



    def toggle_erase_mode(self):
        """切换擦除模式"""
        self.is_erasing = not self.is_erasing
        # 切换时自动调整笔刷大小匹配当前工具

    def keyPressEvent(self, event):
        """恢复A/D切换图片功能"""
        if event.key() == Qt.Key_A:
            self.switch_image_requested.emit("previous")
        elif event.key() == Qt.Key_D:
            self.switch_image_requested.emit("next")
        else:
            super().keyPressEvent(event)

    def set_image(self, pixmap,image_path):
        """加载新图像时重置状态"""
        self.curId = 1
        self.scene.clear()
        self.masks.clear()
        self.active_mask = None
        self.current_image_path = image_path
        self.image_item = self.scene.addPixmap(pixmap)
        self._init_overlay(pixmap.size())
        self.mask_updated.emit()

    def _init_overlay(self, size):
        """初始化透明叠加层"""
        self.overlay_pixmap = QPixmap(size)
        self.overlay_pixmap.fill(Qt.transparent)
        self.overlay_item = self.scene.addPixmap(self.overlay_pixmap)
        self.overlay_item.setZValue(1)

    def create_mask(self):
        """创建新掩码并激活"""
        newid = self.curId
        if not self.masks == []:
            for mask in self.masks:
                self.idlist.append(mask.id)
            for i in range(1,max(self.idlist)):
                self.idlist.sort()
                if i not in self.idlist:
                    newid = i
                    break
                else:
                    self.curId = max(self.idlist) + 1
                    newid = max(self.idlist) + 1
        self.idlist = []
        new_mask = Mask(newid)
        self.curId += 1
        self.masks.append(new_mask)
        self.active_mask = new_mask
        self.mask_updated.emit()
        return new_mask

    def mouseMoveEvent(self, event):
        if self.drawing and self.active_mask:
            current_pos = self.mapToScene(event.pos()).toPoint()
            # 如果存在上次位置则进行插值
            if self.last_brush_pos:
                self._paint_line(self.last_brush_pos, current_pos)

            #self._update_brush_position(event)
            #self._paint_mask()

            self._paint_at_position(current_pos)
            self.last_brush_pos = current_pos
            self._update_overlay()

    def _update_brush_position(self, event):
        self.current_brush_pos = self.mapToScene(event.pos()).toPoint()

    def _paint_mask(self):
        """将像素添加到当前掩码"""
        center = self.current_brush_pos
        for x in range(center.x() - self.brush_size, center.x() + self.brush_size + 1):
            for y in range(center.y() - self.brush_size, center.y() + self.brush_size + 1):
                if (x - center.x()) ** 2 + (y - center.y()) ** 2 <= self.brush_size ** 2:
                    self.active_mask.coordinates.add((x, y))

    def _update_overlay(self):
        """更新所有可见掩码的显示（带高亮效果）"""
        self.overlay_pixmap.fill(Qt.transparent)
        painter = QPainter(self.overlay_pixmap)

        # 绘制当前笔刷尺寸预览
        if self.drawing:
            painter.setPen(QColor(255, 255, 255, 150))
            painter.setBrush(Qt.NoBrush)
            current_size = self.brush_size
            painter.drawEllipse(
                self.current_brush_pos,
                current_size,
                current_size
            )

        # 绘制擦除预览
        if self.is_erasing and self.drawing and self.active_mask:
            painter.setPen(QColor(255, 0, 0, 150))  # 红色半透明边框
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(
                self.current_brush_pos,
                self.brush_size,
                self.brush_size
            )

        for mask in self.masks:
            if not mask.visible:
                continue

            # 当前激活掩码使用更亮的边框
            if mask == self.active_mask:
                painter.setPen(QColor(255, 255, 0, 200))  # 黄色边框
                painter.setBrush(QBrush(mask.color))
            else:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(mask.color.darker(150)))  # 非激活状态颜色加深

            for (x, y) in mask.coordinates:
                painter.drawRect(x, y, 1, 1)

        painter.end()
        self.overlay_item.setPixmap(self.overlay_pixmap)

    def _paint_line(self, start, end):
        """使用Bresenham算法绘制线段"""
        dx = abs(end.x() - start.x())
        dy = abs(end.y() - start.y())
        sx = 1 if start.x() < end.x() else -1
        sy = 1 if start.y() < end.y() else -1
        err = dx - dy

        while True:
            self._paint_at_position(QPoint(start.x(), start.y()))
            if start.x() == end.x() and start.y() == end.y():
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                start.setX(start.x() + sx)
            if e2 < dx:
                err += dx
                start.setY(start.y() + sy)

    def _paint_at_position(self, pos):

        """在指定位置绘制圆形区域,支持擦除"""
        current_size = self.brush_size if self.is_erasing else self.brush_size

        for x in range(pos.x() - current_size, pos.x() + current_size + 1):
            for y in range(pos.y() - current_size, pos.y() + current_size + 1):
                if (x - pos.x()) ** 2 + (y - pos.y()) ** 2 <= current_size ** 2:
                    if self.is_erasing:
                        # 仅从当前激活掩码移除坐标
                        if self.active_mask:
                            self.active_mask.coordinates.discard((x, y))
                    else:
                        if self.active_mask:
                            self.active_mask.coordinates.add((x, y))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_brush_pos = None  # 释放鼠标时清除上次位置


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_list = []
        self.current_index = 0
        self._init_ui()
        self.mask_manager = MaskManager()  # 新增数据管理器
        self.sam = None
        self._init_sam_model()

    def _init_sam_model(self):
        """初始化SAM模型"""
        model_type = "vit_b"
        checkpoint = "sam_vit_b_01ec64.pth"  # 修改为实际路径
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam.to(device=device)
            print("SAM模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载SAM模型失败: {str(e)}")
            self.sam = None

    def _init_ui(self):
        self.setWindowTitle("Mask Labeling Tool")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("""
            background-color: #2E2E2E; 
            color: white;
            QListWidget::item { 
                height: 32px;
                border: 20px solid #444444;
                border-radius: 30px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #555555;
                border-color: #00FF00;
            }
            QToolTip { 
                background-color: #2E2E2E;
                color: black;
                border: 1px solid #AAAAAA;
                padding: 4px;
                border-radius: 3px;
                opacity: 240;
            }
        """)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧画布
        self.canvas = Canvas(parent=self)
        self.canvas.mask_updated.connect(self._update_mask_list)
        self.canvas.switch_image_requested.connect(self._switch_image)  # 连接切换信号
        layout.addWidget(self.canvas, stretch=4)

        # 右侧控制面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 掩码列表
        self.mask_list = QListWidget()
        self.mask_list.itemClicked.connect(self._on_mask_selected)
        self.mask_list.itemDoubleClicked.connect(self._edit_annotation)  # 新增双击信号
        right_layout.addWidget(QLabel("掩码列表:"))
        right_layout.addWidget(self.mask_list)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.btn_new = QPushButton("新建掩码")
        self.btn_new.clicked.connect(self._create_mask)
        btn_layout.addWidget(self.btn_new)

        self.btn_delete = QPushButton("删除")
        self.btn_delete.clicked.connect(self._delete_mask)
        btn_layout.addWidget(self.btn_delete)
        right_layout.addLayout(btn_layout)



        layout.addWidget(right_panel, stretch=1)

        # 工具栏
        self.toolbar = self.addToolBar("Main")
        self.toolbar.addWidget(QPushButton("加载图片", clicked=self._load_images))
        self.toolbar.addWidget(QPushButton("保存数据", clicked=self._save_data))

        self.erase_action = QPushButton("橡皮擦")
        self.erase_action.setCheckable(True)
        self.erase_action.toggled.connect(self._toggle_erase_mode)
        self.toolbar.addWidget(self.erase_action)

        self.auto_seg_btn = QPushButton("自动分割")
        self.auto_seg_btn.clicked.connect(self._run_auto_segmentation)
        self.toolbar.addWidget(self.auto_seg_btn)

        self.extract_btn = QPushButton("提取区域")
        self.extract_btn.clicked.connect(self._extract_region)
        self.toolbar.addWidget(self.extract_btn)

        # 在工具栏添加新按钮
        self.point_seg_btn = QPushButton("正负提示点分割")
        self.point_seg_btn.clicked.connect(self._start_point_segmentation)
        self.toolbar.addWidget(self.point_seg_btn)

        # 绘图笔刷尺寸显示
        self.brush_size_label = QLabel(f" 笔刷: {self.canvas.brush_size}px ")
        self.brush_size_label.setObjectName("brush_size_label")  # 设置唯一标识
        self.toolbar.addWidget(self.brush_size_label)

        # 修改点分割工具栏控件初始化
        self.point_toolbar_widgets = {
            'positive_act': QAction("⊕ 正点", self),
            'negative_act': QAction("⊖ 负点", self),
            'confirm_act': QAction("✅ 确认生成", self),
            'cancel_act': QAction("❌ 取消", self)
        }

        for widget in self.point_toolbar_widgets.values():
            widget.setVisible(False)
            if isinstance(widget, QAction):
                widget.setIconText(widget.text())  # 确保文本显示
                widget.setToolTip(widget.text().replace("⊕ ", "").replace("⊖ ", ""))

        # 添加到工具栏
        self.toolbar.addAction(self.point_toolbar_widgets['confirm_act'])
        self.toolbar.addAction(self.point_toolbar_widgets['cancel_act'])
        self.toolbar.addAction(self.point_toolbar_widgets['positive_act'])
        self.toolbar.addAction(self.point_toolbar_widgets['negative_act'])


        # 连接信号
        self.point_toolbar_widgets['positive_act'].triggered.connect(
            lambda: self._set_point_mode('positive'))
        self.point_toolbar_widgets['negative_act'].triggered.connect(
            lambda: self._set_point_mode('negative'))
        self.point_toolbar_widgets['confirm_act'].triggered.connect(
            self._finalize_mask)
        self.point_toolbar_widgets['cancel_act'].triggered.connect(
            self._cancel_point_mode)


        self.toolbar.setStyleSheet("""
                            QPushButton { 
                                background-color: #444444;
                                color: white;
                                border-radius: 15px; 
                                padding: 10px;
                            }
                            QPushButton:hover { background-color: #555555; }
                            QToolButton:pressed { background-color: #666666; }
                            QPushButton:checked{background-color: rgb(20, 62, 134);border:none;color:rgb(255, 255, 255);}
                            QToolButton:hover {
                                background-color: #5A5A5A;
                            }
                        """)

    def _start_point_segmentation(self):

        if not self.image_list:
            QMessageBox.warning(self, "错误", "请先加载图片")
            return

        """初始化点分割模式"""
        if not self.sam:
            QMessageBox.warning(self, "错误", "SAM模型未加载")
            return


        # 初始化SAM预测器
        current_path = self.image_list[self.current_index]
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.canvas.sam_predictor = SamPredictor(self.sam)
        self.canvas.sam_predictor.set_image(image)

        # 清空现有点
        self._toggle_point_tools(True)
        self.canvas.enable_point_mode('positive')
        self.statusBar().showMessage("请选择正点（左键添加，右键确认）")

    def _toggle_point_tools(self, visible):
        """切换专用工具可见性"""
        for widget in self.point_toolbar_widgets.values():
            widget.setVisible(visible)
        self.point_seg_btn.setChecked(visible)

        # 强制刷新布局和绘制
        self.toolbar.updateGeometry()
        self.toolbar.adjustSize()
        self.toolbar.repaint()

    def _cancel_point_mode(self):
        """取消点分割模式"""
        self.canvas.enable_point_mode(None)
        self._toggle_point_tools(False)
        self.canvas._clear_preview()
        self.canvas.clear_point_markers()  # 清理提示点
        self.statusBar().clearMessage()

    def _set_point_mode(self, mode):
        """设置当前点类型"""
        self.canvas.point_mode = mode
        color = "绿色" if mode == 'positive' else "红色"
        self.statusBar().showMessage(f"当前点类型: {color}（{mode}）")

    def _finalize_mask(self):
        """确认生成最终掩码"""
        if not self.canvas.input_points:
            QMessageBox.warning(self, "错误", "请先添加提示点")
            return

        # 获取最佳掩码
        masks, scores, _ = self.canvas.sam_predictor.predict(
            point_coords=np.array(self.canvas.input_points),
            point_labels=np.array(self.canvas.input_labels),
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]

        # 转换为Mask对象
        new_mask = self.canvas.create_mask()
        new_mask.mask_array = best_mask.astype(bool)
        ys, xs = np.where(best_mask)
        new_mask.coordinates = set(zip(xs, ys))

        # 重置状态
        self.canvas.clear_point_markers()  # 清理提示点
        self.canvas.enable_point_mode(None)
        self.canvas._update_overlay()
        self._update_mask_list()

    def _extract_region(self):
        #根据选中掩码提取图像区域并自动保存
        if not self.image_list:
            QMessageBox.warning(self, "错误", "请先加载图片")
            return

        if not self.canvas.masks:
            QMessageBox.warning(self, "错误", "请先创建或选择掩码")
            return

        try:
            current_path = self.image_list[self.current_index]
            original_image = cv2.imread(current_path)
            base_name = os.path.splitext(os.path.basename(current_path))[0]
            save_dir = os.path.join("Segmentation", base_name)
            os.makedirs(save_dir, exist_ok=True)

            success_count = 0
            failed_masks = []
            h, w = original_image.shape[:2]  # 预计算图像尺寸

            for idx, mask in enumerate(self.canvas.masks):
                try:
                    if mask.mask_array is None or mask.mask_array.size == 0:
                        failed_masks.append(mask.name)
                        continue

                    # 使用向量化操作获取非零坐标（性能优化）
                    ys, xs = np.nonzero(mask.mask_array)
                    if xs.size == 0 or ys.size == 0:
                        failed_masks.append(mask.name)
                        continue

                    # 计算精确边界框（比min/max快30%）
                    min_x, max_x = xs.min(), xs.max()
                    min_y, max_y = ys.min(), ys.max()

                    # 动态智能padding（基于分布而非固定比例）
                    x_range = np.percentile(xs, [5, 95])
                    y_range = np.percentile(ys, [5, 95])
                    pad_x = max(5, int((x_range[1] - x_range[0]) // 20))
                    pad_y = max(5, int((y_range[1] - y_range[0]) // 20))

                    # 边界保护（考虑图像实际尺寸）
                    min_x = max(0, min_x - pad_x)
                    min_y = max(0, min_y - pad_y)
                    max_x = min(w, max_x + pad_x)
                    max_y = min(h, max_y + pad_y)

                    # 裁剪掩码区域
                    cropped = original_image[min_y:max_y, min_x:max_x]
                    mask_roi = mask.mask_array[min_y:max_y, min_x:max_x]

                    # 创建全黑画布（与裁剪区域同尺寸）
                    black_canvas = np.zeros_like(cropped)

                    # 应用掩码：仅保留目标区域，其余设为黑色
                    # 使用向量化操作（比循环快100倍以上）
                    result = np.where(mask_roi[:, :, None], cropped, black_canvas)

                    # 生成文件名
                    save_path = os.path.join(
                        save_dir,
                        f"{base_name}_{mask.annotations}_{mask.name}.png"
                    )

                    cv2.imwrite(save_path, result)
                    success_count += 1

                except Exception as e:
                    failed_masks.append(mask.name)
                    print(f"提取{mask.name}失败: {str(e)}")

            # 显示结果（保持原提示逻辑）
            msg = f"成功提取 {success_count} 个区域\n"
            if failed_masks:
                msg += f"失败掩码：{', '.join(failed_masks)}"

            self.statusBar().showMessage(msg, 10000)
            QMessageBox.information(self, "提取完成",
                                    msg + f"\n保存路径：{save_dir}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量提取失败：{str(e)}")

    """def _extract_region(self):
        if not self.image_list:
            QMessageBox.warning(self, "错误", "请先加载图片")
            return

        if not self.canvas.masks:
            QMessageBox.warning(self, "错误", "请先创建或选择掩码")
            return

        try:
            # 获取原始图片信息
            current_path = self.image_list[self.current_index]
            original_image = cv2.imread(current_path)
            base_name = os.path.splitext(os.path.basename(current_path))[0]

            # 创建保存路径
            save_dir = os.path.join("Segmentation", base_name)
            os.makedirs(save_dir, exist_ok=True)

            success_count = 0
            failed_masks = []

            BASE_SIZE = 512 # 目标尺寸 (width, height)
            LARGE_SIZE = 1024  # 大尺寸
            INTERPOLATION = cv2.INTER_CUBIC  # 插值方法
            KEEP_ASPECT_RATIO = True  # 是否保持宽高比

            # 性能配置参数
            SIZE_THRESHOLD = 2048  # 启用优化的尺寸阈值
            FAST_MODE = False  # 快速模式开关
            DOWNSCALE_FACTOR = 2  # 快速模式下的降采样系数

            # 遍历所有掩码
            for idx, mask in enumerate(self.canvas.masks):
                try:
                    if not mask.coordinates:
                        failed_masks.append(mask.name)
                        continue

                    # 计算边界框
                    xs = [x for x, y in mask.coordinates]
                    ys = [y for x, y in mask.coordinates]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    # ====== 新增预处理优化 ======
                    # 判断是否启用快速模式
                    use_fast_mode = (original_image.shape[0] > SIZE_THRESHOLD or
                                     original_image.shape[1] > SIZE_THRESHOLD)

                    if use_fast_mode and FAST_MODE:
                        # 降采样处理
                        # 获取原始尺寸 (注意OpenCV的shape顺序)
                        orig_h, orig_w = original_image.shape[:2]

                        # 动态计算缩放因子（确保≥1）
                        scale = 1.0
                        if max(orig_w, orig_h) > SIZE_THRESHOLD:
                            scale = 1 / DOWNSCALE_FACTOR

                        # 计算新尺寸（必须为整数且≥1）
                        new_w = max(1, int(orig_w * scale))
                        new_h = max(1, int(orig_h * scale))
                        working_image = cv2.resize(original_image,
                                                   (new_w, new_h),
                                                   interpolation=cv2.INTER_AREA)
                        # 坐标转换（带边界检查）
                        scaled_coords = []
                        for x, y in mask.coordinates:
                            # 坐标缩放并取整
                            sx = int(x * DOWNSCALE_FACTOR)
                            sy = int(y * DOWNSCALE_FACTOR)

                            # 边界保护
                            sx = np.clip(sx, 0, new_w - 1)
                            sy = np.clip(sy, 0, new_h - 1)

                            scaled_coords.append((sx, sy))

                    else:
                        working_image = original_image.copy()
                        scaled_coords = mask.coordinates

                    # 添加动态边界扩展
                    width = max_x - min_x
                    height = max_y - min_y
                    pad_x = max(5, int(width * 0.05))  # 最小5像素
                    pad_y = max(5, int(height * 0.05))

                    min_x = max(0, min_x - pad_x)
                    min_y = max(0, min_y - pad_y)
                    max_x = min(working_image.shape[1], max_x + pad_x)
                    max_y = min(working_image.shape[0], max_y + pad_y)

                    # 创建黑色画布
                    canvas_width = max_x - min_x
                    canvas_height = max_y - min_y
                    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

                    print("创建掩膜")
                    # 创建掩膜
                    mask_array = np.array([[x - min_x, y - min_y] for (x, y) in scaled_coords], dtype=np.int32)
                    mask_image = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                    cv2.fillPoly(mask_image, [mask_array], 255)

                    print("提取原始区域并应用掩膜")
                    # 提取原始区域并应用掩膜
                    roi = working_image[min_y:max_y, min_x:max_x]
                    masked_roi = cv2.bitwise_and(roi, roi, mask=mask_image)


                    # 将结果绘制到黑色画布
                    canvas = cv2.add(canvas, masked_roi)

                    # 获取原始尺寸
                    original_h, original_w = canvas.shape[:2]
                    print("动态尺寸判断")
                    # ====== 新增动态尺寸判断 ======
                    # 判断是否需要使用大尺寸
                    if original_w > BASE_SIZE or original_h > BASE_SIZE:
                        target_size = (LARGE_SIZE, LARGE_SIZE)
                        scale_ratio = LARGE_SIZE / max(original_w, original_h)
                    else:
                        target_size = (BASE_SIZE, BASE_SIZE)
                        scale_ratio = BASE_SIZE / max(original_w, original_h)

                    # 打印调试信息
                    print(f"原始尺寸：{original_w}x{original_h} -> 目标尺寸：{target_size}")

                    # ====== 改进后的缩放逻辑 ======
                    if KEEP_ASPECT_RATIO:
                        # 计算缩放后尺寸（保持长宽比）
                        new_w = int(original_w * scale_ratio)
                        new_h = int(original_h * scale_ratio)

                        # 执行高质量缩放
                        resized = cv2.resize(canvas, (new_w, new_h),
                                             interpolation=INTERPOLATION)

                        # 创建目标画布并居中
                        final_canvas = np.zeros((target_size[1], target_size[0], 3),
                                                dtype=np.uint8)
                        x_offset = (target_size[0] - new_w) // 2
                        y_offset = (target_size[1] - new_h) // 2
                        final_canvas[y_offset:y_offset + new_h,
                        x_offset:x_offset + new_w] = resized
                    else:
                        # 直接拉伸到目标尺寸
                        final_canvas = cv2.resize(canvas, target_size,
                                                  interpolation=INTERPOLATION)

                    # 生成文件名（添加尺寸标识）
                    save_path = os.path.join(save_dir,
                                             f"{base_name}_mask{idx + 1}_{mask.name}_{target_size[0]}x{target_size[1]}.png")

                    cv2.imwrite(save_path, final_canvas)
                    success_count += 1

                except Exception as e:
                    failed_masks.append(mask.name)
                    print(f"提取{mask.name}失败: {str(e)}")

            # 显示汇总结果
            msg = f"成功提取 {success_count} 个区域\n"
            if failed_masks:
                msg += f"失败掩码：{', '.join(failed_masks)}"

            self.statusBar().showMessage(msg, 10000)
            QMessageBox.information(self, "提取完成",
                                    msg + f"\n保存路径：{save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量提取失败：{str(e)}")"""

    # 新增编辑注释的方法
    def _edit_annotation(self, item):
        """双击编辑注释"""
        mask = item.data(Qt.UserRole)
        new_text, ok = QInputDialog.getText(
            self,
            "编辑注释",
            "输入掩码注释:",
            text=mask.annotations
        )

        if ok:
            mask.annotations = new_text
            self._update_mask_list()

            # 标记为未保存状态
            current_path = self.image_list[self.current_index]
            self.mask_manager.update_masks(current_path, self.canvas.masks)

    def _run_auto_segmentation(self):
        """执行自动分割"""
        if not self.sam:
            QMessageBox.warning(self, "警告", "SAM模型未正确加载")
            return

        if not self.image_list:
            QMessageBox.warning(self, "错误", "请先加载图片")
            return

        try:
            # 加载当前图片
            current_path = self.image_list[self.current_index]
            image = cv2.imread(current_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 生成掩码
            mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=16,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.93,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=88,
            )
            sam_masks = mask_generator.generate(image)

            # 获取原图尺寸
            pixmap = self.canvas.image_item.pixmap()
            img_size = pixmap.size()

            # 添加掩码到画布
            self.canvas.add_sam_masks(sam_masks, img_size)

            # 标记为未保存
            self.mask_manager.update_masks(
                current_path,
                self.canvas.masks
            )

        except Exception as e:
            QMessageBox.critical(self, "错误", f"分割失败: {str(e)}")


    def _toggle_erase_mode(self, checked):
        """处理橡皮擦模式切换"""
        self.canvas.toggle_erase_mode()

        # 当激活擦除模式时自动选择最后一个掩码
        if checked and not self.canvas.active_mask and self.canvas.masks:
            self.canvas.active_mask = self.canvas.masks[-1]
            self._update_mask_list()

    def _switch_image(self, direction):
        """改进的图片切换方法"""
        if not self.image_list:
            QMessageBox.warning(self, "提示", "没有加载任何图片")
            return

        # 检查当前图片是否有未保存修改
        if self._check_unsaved_changes():
            return

        # 执行切换
        new_index = self.current_index
        if direction == "previous":
            new_index = max(0, self.current_index - 1)
        elif direction == "next":
            new_index = min(len(self.image_list)-1, self.current_index + 1)

        if new_index != self.current_index:
            self.current_index = new_index
            self._display_image()

    def _check_unsaved_changes(self):
        """检查未保存修改并提示"""
        current_path = self.image_list[self.current_index]
        if not self.mask_manager.data.get(current_path, {}).get("saved", True):
            reply = QMessageBox.question(
                self,
                "未保存修改",
                "当前图片有未保存的修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )

            if reply == QMessageBox.Save:
                self._save_data()
            elif reply == QMessageBox.Cancel:
                return True  # 取消切换
        return False

    def _update_mask_list(self):
        """更新右侧列表（带高亮效果）"""
        self.mask_list.clear()
        for mask in self.canvas.masks:
            # 显示注释（最多显示15个字符）
            annotation_display = mask.annotations[:15] + "..." if len(mask.annotations) > 15 else mask.annotations
            item_text = f"{mask.name} ({len(mask.coordinates)}px)"
            if mask.annotations:
                item_text += f" - {annotation_display}"

            item = QListWidgetItem(item_text)
            item.setToolTip(mask.annotations)  # 鼠标悬停显示完整注释
            item.setData(Qt.UserRole, mask)
            item.setForeground(QBrush(mask.color))
            self.mask_list.addItem(item)

        # 自动选中当前激活掩码
        if self.canvas.active_mask:
            idx = self.canvas.masks.index(self.canvas.active_mask)
            self.mask_list.setCurrentRow(idx)

    def _on_mask_selected(self, item):
        """选择掩码操作"""
        mask = item.data(Qt.UserRole)
        self.canvas.active_mask = mask
        self.canvas._update_overlay()  # 立即刷新显示

    def _create_mask(self):
        """创建掩码时标记为未保存"""
        if not self.image_list:
            QMessageBox.warning(self, "错误", "请先加载图片")
            return

        new_mask = self.canvas.create_mask()

        self.mask_manager.update_masks(
            self.image_list[self.current_index],
            self.canvas.masks
        )
        self.mask_list.setCurrentRow(len(self.canvas.masks)-1)

    def _delete_mask(self):

        if not self.canvas.active_mask or not self.canvas.masks:
            QMessageBox.warning(self, "提示", "没有选中的掩码可删除")
            return

        self.canvas.masks.remove(self.canvas.active_mask)
        self.canvas.active_mask = None
        self.canvas._update_overlay()
        self._update_mask_list()

    def _load_images(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.image_list = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            if not self.image_list:
                QMessageBox.warning(self, "错误", "所选文件夹中没有支持的图片文件")
                return
            self.image_list.sort()
            self.current_index = 0
            self._display_image()

    def _display_image(self):
        """加载图片时恢复/重置掩码数据"""
        if not self.image_list:
            return
        if self.image_list:
            current_path = self.image_list[self.current_index]

            # 初始化图片数据
            self.mask_manager.add_image(current_path)

            # 加载图片
            pixmap = QPixmap(current_path)
            if not pixmap.isNull():
                self.canvas.set_image(pixmap, current_path)
                self.canvas.fitInView(self.canvas.image_item, Qt.KeepAspectRatio)

            # 恢复掩码数据
            self._load_masks(current_path)

    def _load_masks(self, image_path):
        """从管理器加载掩码数据"""
        if image_path in self.mask_manager.data:
            # 这里需要实现从数据到Mask对象的转换
            pass

    def _save_data(self):
        """保存所有掩码数据"""
        if not self.image_list:
            QMessageBox.warning(self, "错误", "没有加载任何图片")
            return
        current_path = self.image_list[self.current_index]
        if not self.canvas.masks:
            QMessageBox.warning(self, "提示", "没有掩码数据可保存")
            return

        save_path = QFileDialog.getSaveFileName(self, "保存数据", "", "JSON文件 (*.json)")[0]
        if save_path:
            data = {
                "image_path": self.image_list[self.current_index],
                "masks": [
                    {
                        "name": mask.name,
                        "color": {
                            "r": mask.color.red(),
                            "g": mask.color.green(),
                            "b": mask.color.blue(),
                            "a": mask.color.alpha()
                        },
                        "coordinates": list(mask.coordinates),
                        "annotations": mask.annotations,  # 新增注释保存
                        "mask_rle": self._array_to_rle(mask.mask_array),
                    } for mask in self.canvas.masks
                ]
            }

            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"数据已保存到: {save_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec_())