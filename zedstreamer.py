import numpy as np
import logging
import pyzed.sl as sl
import math


class ZedCamera:

    def __init__(self):
        print("Running...")
        initParams = sl.InitParameters(camera_resolution=sl.RESOLUTION.RESOLUTION_HD720,
                                       depth_mode=sl.DEPTH_MODE.DEPTH_MODE_PERFORMANCE,
                                       coordinate_units=sl.UNIT.UNIT_METER,
                                       coordinate_system=sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP,
                                       sdk_verbose=True)
        self.cam = sl.Camera()

        if not self.cam.is_opened():
            print("Opening ZED Camera")

        status = self.cam.open(initParams)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        self.runtime = sl.RuntimeParameters(sensing_mode=sl.SENSING_MODE.SENSING_MODE_STANDARD)
        self.mat = sl.Mat()
        self.depthMap = sl.Mat()
        self.pointCloud = sl.Mat()

    def getImage(self, side):

        side = side.upper()
        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:

            if (side == "LEFT"):
                self.cam.retrieve_image(self.mat, sl.VIEW.VIEW_LEFT)
                return self.mat.get_data()

            elif (side == "RIGHT"):
                self.cam.retrieve_image(self.mat, sl.VIEW.VIEW_RIGHT)
                return self.mat.get_data()
            else:
                logging.warning(
                    "No such side of camera exists. Returning NONE")
                return None
        else:
            logging.warning(
                "Failed to grab {} IMAGE. Returning NONE".format(side))
            return None

    #     def getMergedImages(self):
    #         runtime = zcam.PyRuntimeParameters()
    #         left = core.PyMat()
    #         right = core.PyMat()

    #         err = self.cam.grab(runtime)
    #         if err == tp.PyERROR_CODE.PySUCCESS:
    #             self.cam.retrieve_image(left, sl.PyVIEW.PyVIEW_LEFT)
    #             self.cam.retrieve_image(right, sl.PyVIEW.PyVIEW_RIGHT)

    #             stitched, matchesImg = sticher.stichImages(left.get_data(), right.get_data())
    #             return stitched

    #         else:
    #             logging.error("Failed to grab MERGED IMAGE. Returning NONE")
    #             return None

    # Note no side parameter. This is because depth map is centered on LEFT camera
    def getDepthMap(self):

        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_measure(self.depthMap, sl.MEASURE.MEASURE_DEPTH)
            return self.depthMap

        else:
            logging.error("Failed to grab DEPTH MAP. Returning NONE")
            return None

    # Note no side parameter. This is because point cloud is centered on LEFT camera
    def getPointCloud(self):

        err = self.cam.grab(self.runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            self.cam.retrieve_measure(
                self.pointCloud, sl.MEASURE.MEASURE_XYZRGBA)
            return self.pointCloud

        else:
            logging.error("Failed to grab POINT CLOUD. Returning NONE")
            return None

    def getDistanceAt(self, x, y):
        err, pcValue = self.pointCloud.get_value(x, y)

        distance = math.sqrt(pcValue[0] * pcValue[0] +
                             pcValue[1] * pcValue[1] +
                             pcValue[2] * pcValue[2])

        if not np.isnan(distance) and not np.isinf(distance):
            distance = round(distance)
            print("Distance to Camera at ({0}, {1}): {2} mm\n".format(
                x, y, distance))
            return distance
        else:
            print("Can't estimate distance at this position, move the camera\n")
            return -1

    def releaseCam(self):
        self.cam.close()

    def setCamSettings(self, brightness=4, contrast=4, hue=0, sat=4, gain=75, exp=100):
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS, brightness, False)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST, contrast, False)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE, hue, False)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION, sat, False)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN, gain, False)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, exp, False)

    def resetSettings(self):
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_BRIGHTNESS, -1, True)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_CONTRAST, -1, True)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_HUE, -1, True)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_SATURATION, -1, True)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_GAIN, -1, True)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_EXPOSURE, -1, True)
        self.cam.set_camera_settings(sl.CAMERA_SETTINGS.CAMERA_SETTINGS_WHITEBALANCE, -1, True)
