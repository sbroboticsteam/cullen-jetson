import cv2
import matplotlib.pyplot as plt
import numpy as np

# Blue contours are the largest contours within the limit set in the code
# Red contours are the contours with the highest extent ratio
# Green are the contours out of the largest who are the most full/solid
DEBUGGING = False
DEBUG_LARGEST = False
DEBUG_EXTENT = False
DEBUG_SOLIDITY = False
DEBUG_SCORE = True
DEBUG_DISTANCE = False
DEBUG_RATIO = True
DEBUG_IOU = True

DEBUG_BOX = False

SHOW_OUTPUTS = True


# os.system("v4l2-ctl -d /dev/video1"
#           " -c brightness={}".format(80 + randint(-5, 5)) +
#           " -c white_balance_temperature_auto=false"
#           " -c exposure_auto=1"
#           " -c exposure_absolute=20")


class BallScore:

    def __init__(self, contour, size, extent, solidity, areaRatio, circIOU):
        self.contour = contour
        self.size = size
        self.extent = extent
        self.solidity = solidity
        self.areaRatio = areaRatio
        self.iou = circIOU

    def getScore(self):
        return self.size + self.extent + self.solidity


class Vision:

    def __init__(self, hfov):

        # FIXME get intrinsics of ZED camera

        self.ballRadReal = 68.58  # millimeters
        self.inscCircRatio = np.pi / 4  # area inscribed circle / area square

        self.hfov = hfov

        # FIXME get tennis ball's HSV
        # self.lowerBound = np.array([17, 163, 70])
        # self.upperBound = np.array([30, 219, 227])
        self.lowerBound = np.array([30, 255, 135], dtype=np.uint8)
        self.upperBound = np.array([40, 255, 185], dtype=np.uint8)

        # Counter for x axis of scatter graph of DEBUG_DISTANCE function
        self.allDistances = []
        self.xAxis = []
        self.windowsMoved = False

        self.allBoxes = []

    def processImg(self, img):
        self.source = img
        self.resolution = {"width": img.shape[1], "height": img.shape[0]}

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lowerb=self.lowerBound, upperb=self.upperBound)

        open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.createKernel(3))
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, self.createKernel(15))
        mask = cv2.dilate(close, kernel=self.createKernel(7))

        fstream, contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        allCenters = []

        # Hough Circles for IOU later
        circles = cv2.HoughCircles(mask, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=0)
        # TODO test this
        circles = [[cx, cy, r] for cx, cy, r in [i for i in circles[0, :]]]

        # Contour filtering + object detection
        if len(contours) > 0:
            notableContours = self.filterCircles(contours, circles, num=10)

            if len(notableContours) > 0:
                scores = []
                for i, cs in enumerate(notableContours):
                    scores.append(cs.getScore())

                # List of [score, BallScore] lists
                sortedScores = sorted(zip(scores, notableContours), key=lambda l: l[0], reverse=True)

                for n in range(len(sortedScores)):

                    if sortedScores[n][0] > 2.5:

                        lowerA = self.inscCircRatio * 0.95
                        upperA = self.inscCircRatio * 1.05

                        if lowerA < sortedScores[n][1].areaRatio < upperA:

                            if sortedScores[n][1].iou > 0.9:

                                try:
                                    cMoments = cv2.moments(sortedScores[n][1].contour)
                                    centerPoint = (int((cMoments["m10"] / cMoments['m00'])),
                                                   int((cMoments["m01"] / cMoments["m00"])))

                                    cv2.circle(img, tuple(centerPoint), 4, (255, 255, 255), -1)

                                    minRect = cv2.minAreaRect(sortedScores[n][1].contour)
                                    box = cv2.boxPoints(minRect)
                                    box = np.int0(box)
                                    cv2.drawContours(img, [box], 0, (140, 110, 255), 2)

                                    allCenters.append(centerPoint)
                                    cv2.putText(img, "Target", (centerPoint[0], centerPoint[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

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

    def createKernel(self, size):
        return np.ones((size, size), np.uint8)

    def filterCircles(self, contours, circles, num):
        cAreas = []
        scores = []

        for i, c in enumerate(contours):
            cAreas.append(cv2.contourArea(c))

        # sortedAreas[i] designates separate lists of [area, contour]
        sortedAreas = sorted(zip(cAreas, contours), key=lambda l: l[0], reverse=True)
        largestArea = sortedAreas[0][0]

        for i in range(len(sortedAreas)):

            if sortedAreas[i][0] > 1000:
                area = sortedAreas[i][0]

                x, y, w, h = cv2.boundingRect(sortedAreas[i][1])
                rectArea = w * h
                extent = (float(area) / rectArea)

                hull = cv2.convexHull(sortedAreas[i][1])
                hullArea = cv2.contourArea(hull)
                solidity = (float(area) / hullArea)

                relativeArea = (float(area) / largestArea)

                areaRatio = sortedAreas[i][0] / rectArea

                (x, y), radius = cv2.minEnclosingCircle(sortedAreas[i][1])

                centerDiffs = abs(circles[:, 2] - radius)
                sortedDiffs = sorted(zip(centerDiffs, circles), key=lambda l: l[0], reverse=True)
                iou = self.circleIOU(sortedDiffs[0, 0:3], [x, y, radius])

                cScore = BallScore(sortedAreas[i][1], relativeArea, extent, solidity, areaRatio, iou)
                scores.append(cScore)

            if len(scores) >= num:
                break

        return scores

    def hypot(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def circleIOU(self, c1, c2):
        area1 = np.pi * c1[2] ** 2
        area2 = np.pi * c2[2] ** 2
        union = area1 + area2

        d = self.hypot(c1[0], c1[1], c2[0], c2[1])

        if d < c1[2] + c2[2]:

            a = c1[2] ** 2
            b = c2[2] * 2

            x = (a - b + d * d) / (2 * d)
            z = x * x
            y = np.sqrt(a - z)

            if d < abs(c1[2] - c2[2]):
                return np.pi * min(a, b)

            intersection = a * np.arcsin(y / c1[2]) + b * np.arcsin(y / c2[2]) - y * (x + np.sqrt(z + b - a))

            return intersection / union

        return 0

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
                cv2.putText(self.source, "{}".format(c), tuple(centerPoint), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.size), (centerPoint[0] + 50, centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                c += 1

            if (DEBUG_EXTENT):
                cv2.drawContours(self.source, score.contour, -1, (0, 0, 255), 2)
                cv2.putText(self.source, "{}".format("Ext: "), (centerPoint[0], centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.extent), (centerPoint[0] + 50, centerPoint[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                # e += 1

            if (DEBUG_SOLIDITY):
                cv2.drawContours(self.source, score.contour, -1, (0, 255, 0), 2)
                cv2.putText(self.source, "{}".format("Sol: "), (centerPoint[0], centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.solidity), (centerPoint[0] + 50, centerPoint[1] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                # s += 1

            if (DEBUG_SCORE):
                cv2.drawContours(self.source, score.contour, -1, (255, 255, 255), 2)
                cv2.putText(self.source, "{}".format("Sco: "), (centerPoint[0], centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.getScore()), (centerPoint[0] + 50, centerPoint[1] - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                # sc += 1

            if (DEBUG_RATIO):
                cv2.putText(self.source, "{}".format("Rat: "), (centerPoint[0], centerPoint[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.areaRatio), (centerPoint[0] + 50, centerPoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2, cv2.LINE_AA)

            if (DEBUG_IOU):
                cv2.putText(self.source, "{}".format("IOU: "), (centerPoint[0], centerPoint[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(self.source, "{0:.2f}".format(score.iou), (centerPoint[0] + 50, centerPoint[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2, cv2.LINE_AA)

            if (DEBUG_DISTANCE):
                plt.scatter(self.xAxis, self.allDistances)
                plt.pause(float(1) / 30)

    def getSourceImg(self):
        return self.source
