#!/usr/bin/env python3

import math
import depthai as dai
import numpy as np
import cv2

from calc import HostSpatialsCalc
from utility import TextHelper

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
camRgb = pipeline.create(dai.node.ColorCamera)

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDisp = pipeline.create(dai.node.XLinkOut)
xoutDisp.setStreamName("disp")
stereo.disparity.link(xoutDisp.input)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline, dai.DeviceInfo("169.254.1.222")) as device:
    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth")
    dispQueue = device.getOutputQueue(name="disp")
    rgbQueue = device.getOutputQueue(name="rgb")

    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    y = 486
    x = 728
    step = 3
    delta = 5
    hostSpatials.setDeltaRoi(delta)

    print("Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size.")

    while True:
        depthData = depthQueue.get()
        rgbData = rgbQueue.get()
        dispData = dispQueue.get()

        # Calculate spatial coordinates from depth frame
        spatials, centroid = hostSpatials.calc_spatials(depthData, (x, y))  # centroid == x/y in our case

        # Get disparity frame for nicer depth visualization
        disp = dispData.getFrame()
        disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        rgbFrame = rgbData.getCvFrame()

        text.rectangle(rgbFrame, (x - delta, y - delta), (x + delta, y + delta))
        text.putText(rgbFrame,  "x: " +str(x)+"X: " + ("{:.5f}m".format(spatials['x'] / 1) if not math.isnan(spatials['x']) else "--"),
                     (x + 10, y + 20))
        text.putText(rgbFrame, "y: " +str(y)+"Y: " + ("{:.5f}m".format(spatials['y'] / 1) if not math.isnan(spatials['y']) else "--"),
                     (x + 10, y + 35))
        text.putText(rgbFrame, "Z: " + ("{:.5f}m".format(spatials['z'] / 1) if not math.isnan(spatials['z']) else "--"),
                     (x + 10, y + 50))

        # Blend when both frames are received
        if len(disp.shape) < 3:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(rgbFrame, 0.4, disp, 0.6, 0)
        cv2.imshow("rgb-depth", rgbFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            y -= step
        elif key == ord('a'):
            x -= step
        elif key == ord('s'):
            y += step
        elif key == ord('d'):
            x += step
        elif key == ord('r'):  # Increase Delta
            if delta < 50:
                delta += 1
                hostSpatials.setDeltaRoi(delta)
        elif key == ord('f'):  # Decrease Delta
            if 3 < delta:
                delta -= 1
                hostSpatials.setDeltaRoi(delta)