import math
import numpy as np
import depthai as dai


class HostSpatialsCalc:
    # 我们需要 device 对象来获取校准数据
    def __init__(self, device):
        self.calibData = device.readCalibration()  # 读取设备的校准数据

        # 参数设置
        self.DELTA = 5  # ROI 的偏移量
        self.THRESH_LOW = 200  # 最低阈值为 20cm
        self.THRESH_HIGH = 30000  # 最高阈值为 30m

    # 设置下限阈值
    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low

    # 设置上限阈值
    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low

    # 设置 ROI 的偏移量
    def setDeltaRoi(self, delta):
        self.DELTA = delta

    # 检查输入是否为 ROI 或点。如果是点，则转换为 ROI
    def _check_input(self, roi, frame):
        if len(roi) == 4:
            return roi  # 如果输入是 ROI，则直接返回
        if len(roi) != 2:
            raise ValueError("输入必须是 ROI（4 个值）或点（2 个值）！")  # 如果输入不是 ROI 或点，则引发错误

        # 限制点的范围，以确保 ROI 不会超出帧的边界
        self.DELTA = 5  # 在点周围取 10x10 深度像素进行深度平均
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return x - self.DELTA, y - self.DELTA, x + self.DELTA, y + self.DELTA

    # 计算角度
    def _calc_angle(self, frame, offset, HFOV):
        return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    # roi 必须是整数列表
    def calc_spatials(self, depthData, roi, averaging_method=np.mean):
        depthFrame = depthData.getFrame()  # 获取深度帧

        roi = self._check_input(roi, depthFrame)  # 如果输入是点，则将其转换为 ROI
        xmin, ymin, xmax, ymax = roi

        # 计算 ROI 中的平均深度
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        # 计算主机上空间坐标所需的信息
        HFOV = np.deg2rad(self.calibData.getFov(dai.CameraBoardSocket(depthData.getInstanceNum())))

        averageDepth = averaging_method(depthROI[inRange])  # 计算 ROI 内的平均深度

        # 获取 ROI 的质心
        centroid = {
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        # 计算深度图像宽度和高度的中间位置
        midW = int(depthFrame.shape[1] / 2)
        midH = int(depthFrame.shape[0] / 2)
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        # 计算 x 和 y 方向的角度
        angle_x = self._calc_angle(depthFrame, bb_x_pos, HFOV)
        angle_y = self._calc_angle(depthFrame, bb_y_pos, HFOV)

        # 计算空间坐标
        spatials = {
            'z': averageDepth,
            'x': averageDepth * math.tan(angle_x),
            'y': -averageDepth * math.tan(angle_y)
        }
        return spatials, centroid
