import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D


def intersect2D(Line1, Line2):
    x1, y1, x2, y2 = Line1[0]
    a1, b1, a2, b2 = Line2[0]
    L1 = [[x1, y1], [x2, y2]]
    L2 = [[a1, b1], [a2, b2]]
    s = np.vstack([L1, L2])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z != 0:
        return np.array([x / z, y / z])
    else:
        return np.array([x, y])


def eucDistance(x1, y1, x2, y2):
    distance = np.sqrt(np.power(x1 - x2, 2) + np.power(y1 - y2, 2))
    return distance


def average2DCoord(points, Inliers):
    xSum = 0
    ySum = 0
    for index in Inliers:
        error, i, j = index
        x, y, indI, indJ = points[np.where(points[:, 3] == j)][0]
        if y < 0:
            print("y: {} has error: {}".format(y, error))
        xSum += x
        ySum += y
    result = np.array([xSum / (Inliers.size / 3), ySum / (Inliers.size / 3)])
    return result


def vanishingP(Lines):
    candidates = Lines.size / 4
    print(candidates)
    completed = []
    result = np.array([], dtype=np.float64).reshape(0, 3)
    averageError = np.array([], dtype=np.float64)
    # print(Lines.size)
    for i in range(0, int(candidates)):
        Line1 = Lines[i]
        completed.append(i)
        intersections = np.array([], dtype=np.float64).reshape(0, 4)

        for j in range(0, int(candidates)):
            if j not in completed:
                intr = intersect2D(Line1, Lines[j])
                intr = np.append(intr, [i, j])

                distanceToLine = eucDistance(Line1[0][0], Line1[0][1], intr[0], intr[1])
                if distanceToLine > 50:
                    intersections = np.vstack([intersections, intr])
                    # if intr[0] > 1000 or intr[0] < 0 or intr[1] > 560 or intr[1] < 0:
                    #     print(
                    #         "this is x: {}, and this is y: {}".format(intr[0], intr[1])
                    #     )
                    # show lines that hit
                    # img2 = img.copy()
                    # cv.line(
                    #     img2,
                    #     (int(Line1[0][0]), int(Line1[0][1])),
                    #     (int(intr[0]), int(intr[1])),
                    #     (255, 0, 0),
                    #     2,
                    # )
                    # cv.line(
                    #     img2,
                    #     (int(Lines[j][0][0]), int(Lines[j][0][1])),
                    #     (int(intr[0]), int(intr[1])),
                    #     (255, 0, 0),
                    #     2,
                    # )
                    # cv.line(
                    #     img2,
                    #     (int(Lines[j][0][0]), int(Lines[j][0][1])),
                    #     (int(Lines[j][0][2]), int(Lines[j][0][3])),
                    #     (127, 255, 0),
                    #     2,
                    # )
                    #
                    # # if i > 100:
                    # fig, ax = plt.subplots()
                    # plt.imshow(img2)
                    # plt.xlim(-200, 1000)
                    # # plt.ylim(-1000, 1000)

                    # print("hit is at == x:{} and y: {} ".format(intr[0], intr[1]))
                    # tempX1 = [Line1[0][0], intr[0]]
                    # tempY1 = [Line1[0][1], intr[1]]
                    # tempX2 = [Lines[j][0][0], intr[0]]
                    # tempY2 = [Lines[j][0][1], intr[1]]
                    # line1 = Line2D(tempX1, tempY1, color="b")
                    # line2 = Line2D(tempX2, tempY2, color="b")

                    # # line2 = Line2D(tempA2, tempB, color="b")
                    # # plt.axline((Line1[0][0], Line1[0][1]), (intr[0], intr[1]))
                    # # plt.plot((Lines[j][0][0], Lines[j][0][1]), (intr[0], intr[1]))

                    # ax.add_line(line1)
                    # ax.add_line(line2)

                    # # print("We should see #{} of points".format(counter))

                    # cv.imshow("Standard-Window", img2)
                    # plt.show()
                    # k = cv.waitKey(0)

        distance = np.array([], dtype=np.float64).reshape(0, 3)
        # print(intersections)
        for j in range(0, int(intersections.size / 4)):
            if (
                Line1[0][0] < intersections[j][0]
            ):  # is the intersection behind of in front of the line segment?
                # get euclidian distance
                d = eucDistance(
                    Line1[0][0],
                    Line1[0][1],
                    intersections[j][0],
                    intersections[j][1],
                )
                tempV = np.array([d * -1, i, intersections[j][3]])
                distance = np.vstack([distance, tempV])
            else:
                d = eucDistance(
                    Line1[0][0],
                    Line1[0][1],
                    intersections[j][0],
                    intersections[j][1],
                )
                tempV = np.array([d, i, intersections[j][3]])

                distance = np.vstack([distance, tempV])

        if distance.size > 30:  # good nr?
            distance = distance[distance[:, 0].argsort()]

            distance
            prev = 0
            prevprev = 0
            errorV = np.array([], dtype=np.float64).reshape(0, 3)
            # error calc explained
            # points         x-------x----------x----------------------x----x--x-x-x-x-x----x--x---x-----------x---x---------------x
            #             prevprev  prev       dist
            #   error 1 is:
            #                 -------
            #   error 2 is:
            #                         ----------
            # Point at dist's error value depends also on prevprev in order to percive clusters of points together as more important.
            # This diminishes the chance that two outliers which only by chance coincide close together gets mistaken as inliers.
            # The first two points are always concidered outliers
            # We squared error values because we want to emphesise large distances.
            for d in distance:  # pingpong
                dist, indI, indJ = d
                if prev == 0:
                    errorV = np.vstack([errorV, np.array([200000, indI, indJ])])
                    prev = dist
                elif prevprev == 0:
                    errorV = np.vstack([errorV, np.array([200000, indI, indJ])])
                    prevprev = prev
                    prev = dist
                else:
                    error1 = pow((prev - prevprev), 2)
                    error2 = pow((dist - prev), 2)
                    currError = np.array([error1 + error2, indI, indJ])
                    errorV = np.vstack([errorV, currError])
                    prevprev = prev
                    prev = dist
            # compare distances and group the 95th percentile of points closest to eachother
            errorV = errorV[errorV[:, 0].argsort()]

            pInliers = int((errorV.size / 3) * 0.1)
            pInliersV = np.array([], dtype=np.float64).reshape(0, 3)
            if pInliers > 1:
                for j in range(0, pInliers):
                    pInliersV = np.vstack([pInliersV, errorV[j]])

            # more outlier removal, remove points further away than the average distance from the mean coordinate point
            # Then recalculate the mean coordinate point
            averageError = np.mean(pInliersV, axis=0)
            # for v in pInliersV:
            #     error, indI, indJ = v
            #     averageError += error
            # img2 = img.copy()
            # counter = 0
            # for index in pInliersV:
            #     error, i, j = index
            #     x, y, indI, indJ = intersections[np.where(intersections[:, 3] == j)][0]
            #     img2 = cv.circle(
            #         img2,
            #         (int(x), int(y)),
            #         radius=4,
            #         color=(0, 0, 10 * counter),
            #         thickness=-1,
            #     )
            #     if x < 0 or y < 0 or x > 1000 or y > 560:
            #         print("coordinates : [{},{}]".format(x, y))
            #     counter += 1
            # print("We should see #{} of points".format(counter))
            # cv.imshow("Standard-Window", img2)
            if pInliersV.size > 0:
                vanishingPointGuess = average2DCoord(intersections, pInliersV)
                for completedLines in pInliersV:
                    error, completedI, completedJ = completedLines
                    completed.append(completedJ)

                # print(vanishingPointGuess[0])
                # print(
                #     "vanishing point # {} Coord: {}  Has an error of: {}".format(
                #         i, vanishingPointGuess, averageError[0]
                #     )
                # )
                # calculate mean coordinate point and output this as vanishing point
                # print("")
        if pInliersV.size > 0:
            tempResult = np.array(
                [(vanishingPointGuess[0], vanishingPointGuess[1], averageError[0])]
            )
        result = np.vstack([result, tempResult])
    # compare distances of all potential vanishing points
    # the largest / closest / points with least error values, becomes the first vanishing point
    # the other groups have to be orthogonal to the first if they are to be vanishing points also,
    # remove the group least likely to be a vanishing point (outliers)
    # perform a least squares method on the three vanishing points to make them completely orthogonal

    return result


def value_close(value, Line_arr):
    result = 0
    counter = 0
    for rows in Line_arr:
        counter += 1
        saved_group, saved_theta = rows[:2]
        if abs(abs(value) - abs(saved_theta)) < 4:
            result = saved_group
            break
    # print("group nr: " + str(result))
    if result == 0:
        result = np.max(lineArray, axis=0)[0] + 1
    return result


def return_three_max_val_indices(arr):
    maxval1 = 0
    maxval2 = 0
    maxval3 = 0
    for i in range(0, arr.size):
        if arr[i] > maxval1:
            result1 = i
            maxval1 = arr[i]
        elif arr[i] > maxval2:
            result2 = i
            maxval2 = arr[i]
        elif arr[i] > maxval3:
            result3 = i
            maxval3 = arr[i]
    result = np.array([result1, result2, result3])
    return result


# img = cv.imread("building.jpg")
# img = cv.imread("City_Scape.png")
interp = cv.INTER_AREA
img = cv.imread("complex.jpg")

if img is None:
    sys.exit("Could not read image.")
np.append([1, 2, 3], [4, 5, 6])
rows, cols, channels = img.shape

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurry = cv.GaussianBlur(gray, (5, 5), 1)
edges = cv.Canny(blurry, 70, 255, apertureSize=3, L2gradient=1)
lines = cv.HoughLinesP(edges, 3, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
length = np.array([], dtype=np.float64).reshape(0, 2)
counter = 0
print(len(lines))
for line in lines:
    x1, y1, x2, y2 = line[0]
    tempLength = np.array([eucDistance(x1, y1, x2, y2), counter])
    length = np.vstack([length, tempLength])
    counter += 1

# length = length[length[:, 0].argsort()]
# amount = int((length.size/2)*0.4)
# mask = np.ones(len(lines))
# for i in range(0,amount):


lineArray = np.array([])
group_index = 1
segregated_array = np.array([])
VanishingPCoord = vanishingP(lines)

for line in lines:
    x1, y1, x2, y2 = line[0]
    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
    else:
        m = 1000000
    c = y2 - m * x2
    theta = np.degrees(np.arctan(m))
    # print("theta : " + str(theta))
    if lineArray.size != 0:
        group_index = value_close(theta, lineArray)
        lineArray = np.concatenate(
            (lineArray, np.array([[group_index, theta, x1, y1, x2, y2]])), axis=0
        )
    else:
        lineArray = np.array([[group_index, theta, x1, y1, x2, y2]])

    if segregated_array.size == 0:
        segregated_array = np.array([1])
    elif segregated_array.size > group_index:
        segregated_array[int(group_index - 1)] = (
            segregated_array[int(group_index - 1)] + 1
        )
    else:
        segregated_array = np.append(segregated_array, [1])


# print(segregated_array)
axels = return_three_max_val_indices(segregated_array)
# print("axel indices: " + str(axels))
for line in lineArray:
    group, theta, x1, y1, x2, y2 = line
    # print(group)
    if int(group - 1) == axels[0]:
        cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    elif int(group - 1) == axels[1]:
        cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    elif int(group - 1) == axels[2]:
        cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
average = np.mean(VanishingPCoord, axis=0)
print("height: {},  width: {}".format(img.shape[0], img.shape[1]))
fig, ax = plt.subplots()
# plt.xlim(-500, img.shape[0] + 500)
# plt.ylim(1000, -img.shape[1] - 500)
plt.imshow(img, interpolation="nearest")
for dot in VanishingPCoord:
    x, y, err = dot
    if average[2] > err:
        print("x : {},  y: {}".format(x, y))
        if x < 0 or y < 0 or x > 1000 or y > 560:
            circle = plt.Circle((x, y), 5, color="g")
            ax.add_patch(circle)
            print("ojojojoj")
        else:
            circle = plt.Circle((x, y), 5, color="b")
            ax.add_patch(circle)

print("height: {},  width: {}".format(img.shape[0], img.shape[1]))
# print(lines)
# print(segregated_array)
cv.imshow("wind", edges)
k = cv.waitKey(0)
# cv.imshow("wind", img)
# k = cv.waitKey(0)

# todo, fixa en bra punkt
# todo2 ifall den över funkar:
# klura ut hur vi prioriterar vilka vanishing points som linjerna borde höra till. som i "Efficient Computation of Vanishing Points"
plt.show()
k = cv.waitKey(0)
