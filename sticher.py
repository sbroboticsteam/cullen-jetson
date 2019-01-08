import cv2
import numpy as np

sifter = cv2.xfeatures2d.SIFT_create()


def alphaBlend(img1, img2):
    imgMask = np.ma.masked_not_equal(img1, 0)
    warpedMask = np.ma.masked_not_equal(img2, 0)
    mask = np.logical_and(imgMask.mask, warpedMask.mask)

    notIntImg = np.ma.array(img1, mask=mask).filled(0)
    notIntWarped = np.ma.array(img2, mask=mask).filled(0)
    intImg = img1 - notIntImg
    intWarped = img2 - notIntWarped
    blended = cv2.addWeighted(intImg, 0.5, intWarped, 0.5, 0)

    blended = blended + notIntImg + notIntWarped

    return blended


def getTransform(img1, img2):
    # compute sift descriptors
    kp1, desc1 = sifter.detectAndCompute(img1, None)
    kp2, desc2 = sifter.detectAndCompute(img2, None)

    # find all mactches
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)

    imgMatch = np.zeros(img1.shape)
    imgMatch = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, imgMatch)

    # find perspective transform matrix using RANSAC
    srcPts = np.asarray([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    dstPts = np.asarray([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    warpMat, mask = cv2.findHomography(srcPts, dstPts, method=cv2.RANSAC)

    return warpMat, imgMatch


def stichImages(left, right):
    left = cv2.copyMakeBorder(left, 0, 0, 0, right.shape[1], cv2.BORDER_CONSTANT)
    # Python complained about too many values to unpack. Excuse this
    r = left.shape[0]
    c = left.shape[1]

    warpMat, imgMatch = getTransform(left, right)
    right = cv2.warpPerspective(right, warpMat, dsize=(c,r))
    stiched = alphaBlend(left, right)

    return stiched, imgMatch
