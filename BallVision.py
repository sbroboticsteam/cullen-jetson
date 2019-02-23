import cv2
import matplotlib.pyplot as plt
import numpy as np

# Blue contours are the largest contours within the limit set in the code
# Red contours are the contours with the highest extent ratio
# Green are the contours out of the largest who are the most full/solid
DEBUGGING = True
DEBUG_LARGEST = True
DEBUG_EXTENT = True
DEBUG_SOLIDITY = True
DEBUG_SCORE = True
DEBUG_WH = True

DEBUG_DISTANCE = False

DEBUG_BOX = False

SHOW_OUTPUTS = True


# os.system("v4l2-ctl -d /dev/video1"
#           " -c brightness={}".format(80 + randint(-5, 5)) +
#           " -c white_balance_temperature_auto=false"
#           " -c exposure_auto=1"
#           " -c exposure_absolute=20")


class BallScore:

    def __init__(self, contour, size, extent, solidity, whRatio):
        self.contour = contour
        self.size = size
        self.extent = extent
        self.solidity = solidity
        self.whRatio = whRatio

    def getScore(self):
        return self.size + self.solidity + self.extent  # 2.74 Ideal


class Vision:

    def __init__(self, hfov):

        # FIXME get intrinsics of ZED camera

        self.focalLength = 699.772

        self.ballRadReal = 68.58  # millimeters
        self.inscCircRatio = np.pi / 4  # area inscribed circle / area square

        self.hfov = hfov

        # In lab
        self.lowerBound = np.array([15, 100, 0], dtype=np.uint8)
        self.upperBound = np.array([50, 235, 140], dtype=np.uint8)

        # Counter for x axis of scatter graph of DEBUG_DISTANCE function
        self.allDistances = []
        self.xAxis = []
        self.windowsMoved = False

        self.allBoxes = []

    def processImg(self, img):
        self.source = img
        self.resolution = {"width": img.shape[1], "height": img.shape[0]}
        self.degreesPerPix = self.hfov / (np.sqrt(self.resolution["width"] ** 2 + self.resolution["height"] ** 2))

        img = cv2.GaussianBlur(img, (15, 15), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lowerb=self.lowerBound, upperb=self.upperBound)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.createKernel(7))
        mask = cv2.dilate(mask, kernel=self.createKernel(3))
        mask = cv2.dilate(mask, kernel=self.createKernel(3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.createKernel(10))

        fstream, contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        allCenters = []

        # Contour filtering + object detection
        if len(contours) > 0:
            notableContours = self.filterCircles(contours, num=10)

            if len(notableContours) > 0:
                scores = []
                for i, cs in enumerate(notableContours):
                    scores.append(cs.getScore())

                # List of [score, BallScore] lists
                sortedScores = sorted(zip(scores, notableContours), key=lambda l: l[0], reverse=True)

                for n in range(len(sortedScores)):

                    if sortedScores[n][0] > 1.75:

                        lowerA = self.inscCircRatio * 0.90
                        upperA = self.inscCircRatio * 1.10

                        if lowerA < sortedScores[n][1].extent < upperA and 0.75 <= sortedScores[n][1].whRatio <= 1.25:

                            try:
                                cMoments = cv2.moments(sortedScores[n][1].contour)
                                centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                               int((cMoments["m01"] / cMoments["m00"])))

                                cv2.circle(self.source, tuple(centerPoint), 4, (255, 255, 255), -1)

                                minRect = cv2.minAreaRect(sortedScores[n][1].contour)
                                box = cv2.boxPoints(minRect)
                                box = np.int0(box)
                                cv2.drawContours(self.source, [box], 0, (140, 110, 255), 2)

                                allCenters.append(centerPoint)
                                cv2.putText(self.source, "Target", (centerPoint[0], centerPoint[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

                                distance = self.calcRealDistance(minRect[1][0])
                                angle = self.calcAngle(centerPoint[0])

                                if SHOW_OUTPUTS:
                                    cv2.putText(self.source, "{0:.2f}".format(distance), (centerPoint[0] - 100, centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (268, 52, 67), 2, cv2.LINE_AA)
                                    cv2.putText(self.source, "{0:.2f}".format(angle), (centerPoint[0] - 100, centerPoint[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

                                if DEBUGGING:

                                    if DEBUG_BOX:
                                        cv2.circle(self.source, tuple(box[0]), 4, (255, 0, 0), -1)
                                        cv2.circle(self.source, tuple(box[1]), 4, (0, 255, 0), -1)
                                        cv2.circle(self.source, tuple(box[2]), 4, (0, 0, 255), -1)
                                        cv2.circle(self.source, tuple(box[3]), 4, (0, 0, 0), -1)

                            except ZeroDivisionError:
                                pass

                if DEBUGGING:
                    self.showDebugStatements(notableContours)

        cv2.imshow("Source", self.source)
        cv2.imshow("Mask", mask)

    def createKernel(self, size):
        return np.ones((size, size), np.uint8)

    def filterCircles(self, contours, num):
        cAreas = []
        scores = []

        for i, c in enumerate(contours):
            cAreas.append(cv2.contourArea(c))

        # sortedAreas[i] designates separate lists of [area, contour]
        sortedAreas = sorted(zip(cAreas, contours), key=lambda l: l[0], reverse=True)
        largestArea = sortedAreas[0][0]

        for i in range(len(sortedAreas)):

            if sortedAreas[i][0] > 500:
                area = sortedAreas[i][0]

                x, y, w, h = cv2.boundingRect(sortedAreas[i][1])
                rectArea = w * h
                extent = (float(area) / rectArea)

                hull = cv2.convexHull(sortedAreas[i][1])
                hullArea = cv2.contourArea(hull)
                solidity = (float(area) / hullArea)

                relativeArea = (float(area) / largestArea)

                minRect = cv2.minAreaRect(sortedAreas[i][1])
                box = cv2.boxPoints(minRect)
                box = np.int0(box)
                minWHRatio = np.sqrt(((box[0][1] - box[1][1]) ** 2 + (box[0][0] - box[1][0]) ** 2) /
                                     ((box[0][1] - box[3][1]) ** 2 + (box[0][0] - box[3][0]) ** 2))

                cScore = BallScore(sortedAreas[i][1], relativeArea, extent, solidity, minWHRatio)
                scores.append(cScore)

            if len(scores) >= num:
                break

        return scores

    def hypot(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calcRealDistance(self, pxWidth):
        return (self.ballRadReal * self.focalLength) / pxWidth

    def calcAngle(self, centerX):
        centerX -= self.resolution["width"] / 2
        return self.degreesPerPix * centerX

    def showDebugStatements(self, scores):
        # Counter for nth contour
        c = 0
        # e = 0
        # s = 0
        # sc = 0
        # i = 0

        for score in scores:
            # try:
            cMoments = cv2.moments(score.contour)
            centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                           int((cMoments["m01"] / cMoments["m00"])))
            # except ZeroDivisionError:
            #     pass

            if (DEBUG_LARGEST):
                cv2.drawContours(self.source, score.contour, -1, (255, 0, 0), 2)
                cv2.putText(self.source, "{}".format(c), tuple(centerPoint), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.size), (centerPoint[0] + 50, centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
                c += 1

            if (DEBUG_EXTENT):
                cv2.drawContours(self.source, score.contour, -1, (0, 0, 255), 2)
                cv2.putText(self.source, "{}".format("Ext: "), (centerPoint[0] - 50, centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.extent), (centerPoint[0] + 50, centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                # e += 1

            if (DEBUG_SOLIDITY):
                cv2.drawContours(self.source, score.contour, -1, (0, 255, 0), 2)
                cv2.putText(self.source, "{}".format("Sol: "), (centerPoint[0] - 50, centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.solidity), (centerPoint[0] + 50, centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                # s += 1

            if (DEBUG_SCORE):
                cv2.drawContours(self.source, score.contour, -1, (255, 255, 255), 2)
                cv2.putText(self.source, "{}".format("Sco: "), (centerPoint[0] - 50, centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.getScore()), (centerPoint[0] + 50, centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                # sc += 1

            if (DEBUG_WH):
                cv2.putText(self.source, "{}".format("WH: "), (centerPoint[0] - 50, centerPoint[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.whRatio), (centerPoint[0] + 50, centerPoint[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2, cv2.LINE_AA)

            if (DEBUG_DISTANCE):
                plt.scatter(self.xAxis, self.allDistances)
                plt.pause(float(1) / 30)

    def getSourceImg(self):
        return self.source
