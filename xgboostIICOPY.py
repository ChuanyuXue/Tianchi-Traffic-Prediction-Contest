import numpy as np
import collections
import xgboost
import math
from datetime import  datetime, timedelta


def loadFile(filePath):
    idHash = collections.OrderedDict()
    hashFile = open(filePath)
    hashFile.readline()  # to skip the frist line
    #count = 0
    #to load the data in room
    for eachLine in hashFile:
        #if count > 10000:
        #    break
        eachTimes = eachLine.split(",")
        id = eachTimes[0]
        idHash[id] = collections.OrderedDict()

        for i in range(1, len(eachTimes)):
            tempList = eachTimes[i].split(":")
            timeID = tempList[0]
            eachDates = tempList[1].split(";")

            idHash[id][timeID] = collections.OrderedDict()
            for eachDate in eachDates:
                if eachDate != '':
                    dateValuePair = eachDate.split("-")
                    if len(dateValuePair) == 2:
                        dateID = dateValuePair[0]
                        value = dateValuePair[1]
                        idHash[id][timeID][dateID] = [float(value),float(value)]
                        #count = count + 1

    return idHash

def loadLinkInfo(filePath):
    idHash = {}
    hashFile = open(filePath)
    hashFile.readline()  # to skip the frist line

    for eachLine in hashFile:
        eachInfo = eachLine.split(";")
        linkID = eachInfo[0]
        roadLength = eachInfo[1]
        roadWidth = eachInfo[2]
        idHash[linkID] = [int(roadLength), int(roadWidth)]
    return idHash

def dataSmooth(idHash):
    for linkID in idHash.keys():
        for timeID in idHash[linkID].keys():
            for dateID in idHash[linkID][timeID].keys():
                if (str(int(dateID) - 1) not in idHash[linkID][timeID]) or (str(int(dateID) + 1) not in idHash[linkID][timeID]):
                    continue
                else:
                    idHash[linkID][timeID][dateID][1] = (idHash[linkID][timeID][str(int(dateID) - 1)][0]\
                                                    + idHash[linkID][timeID][dateID][0]\
                                                     + idHash[linkID][timeID][str(int(dateID) + 1)][0]) / 3

    for linkID in idHash.keys():
        for timeID in idHash[linkID].keys():
            if (str(int(timeID) - 2) not in idHash[linkID]) or\
                (str(int(timeID) + 2) not in idHash[linkID]) or\
                    (str(int(timeID) - 4)) not in idHash[linkID] or\
                    (str(int(timeID) + 4)) not in idHash[linkID]:
                continue
            else:
                for dateID in idHash[linkID][timeID].keys():
                    if (dateID not in idHash[linkID][str(int(timeID) - 2)]) or\
                       (dateID not in idHash[linkID][str(int(timeID) + 2)]) or \
                        (dateID not in idHash[linkID][str(int(timeID) + 4)]) or \
                        (dateID not in idHash[linkID][str(int(timeID) - 4)])    :
                        continue
                    else:
                        idHash[linkID][timeID][dateID][1] = (idHash[linkID][str(int(timeID) - 2)][dateID][1]\
                                                        + idHash[linkID][timeID][dateID][1]\
                                                         + idHash[linkID][str(int(timeID) + 2)][dateID][1] \
                                                            + idHash[linkID][str(int(timeID) + 4)][dateID][1] \
                                                             + idHash[linkID][str(int(timeID) - 4)][dateID][1])/ 5

    return idHash


def dataFix(idHash):
    count = 0
    countall = 0
    for linkID in idHash:
        for timeID in idHash[linkID]:
            valueList = []
            for dateID in idHash[linkID][timeID]:
                valueList.append(idHash[linkID][timeID][dateID][0])
            average = np.average(valueList)
            dev = np.std(valueList)
            for dateID in idHash[linkID][timeID]:
                countall = countall + 1
                if(idHash[linkID][timeID][dateID][0] < average - 3 * dev or idHash[linkID][timeID][dateID][0] > average + 3 * dev):
                    idHash[linkID][timeID][dateID][0] = average
    return idHash


def modle(timeStart, timeEnd, linkInfoHash, medianTimeHash, preStart, preEnd):
    # get histroy median feature
    linkAndTimeHash = {}
    linkAndTimeHashS = {}
    for linkID in medianTimeHash.keys():
        for timeID in medianTimeHash[linkID]:
            if preStart <= int(timeID) and int(timeID) < preEnd \
                    or 900 <= int(timeID) and int(timeID) < 960 \
                    or 1080 <= int(timeID) and int(timeID) < 1140:
                valueList = []
                valueListS = []
                for dateID in medianTimeHash[linkID][timeID]:
                    if 0 <= int(dateID) and int(dateID) < 92:
                        valueList.append(medianTimeHash[linkID][timeID][dateID][0])
                        valueListS.append(medianTimeHash[linkID][timeID][dateID][1])
                if len(valueList) != 0:
                    medianValue = np.median(valueList)
                    key = linkID + '-' + timeID
                    linkAndTimeHash[key] = medianValue
                if len(valueListS) != 0:
                    medianValueS = np.median(valueListS)
                    key = linkID + '-' + timeID
                    linkAndTimeHashS[key] = medianValueS



    # get history mean feature
    linkAndTimeHashM = {}
    linkAndTimeHashMS = {}
    for linkID in medianTimeHash.keys():
        for timeID in medianTimeHash[linkID]:
            if preStart <= int(timeID) and int(timeID) < preEnd \
                    or 900 <= int(timeID) and int(timeID) < 960 \
                    or 1080 <= int(timeID) and int(timeID) < 1140:
                valueList = []
                valueListS = []
                for dateID in medianTimeHash[linkID][timeID]:
                    if 63 <= int(dateID) and int(dateID) < 92:
                        valueList.append(medianTimeHash[linkID][timeID][dateID][0])
                        valueListS.append(medianTimeHash[linkID][timeID][dateID][1])
                if len(valueList) != 0:
                    medianValue = np.median(valueList)
                    key = linkID + '-' + timeID
                    linkAndTimeHashM[key] = medianValue
                if len(valueListS) != 0:
                    medianValueS = np.median(valueListS)
                    key = linkID + '-' + timeID
                    linkAndTimeHashMS[key] = medianValueS

    # get history max and min and average feature
    linkAndTimeHashMax = {}
    linkAndTimeHashMaxS = {}
    linkAndTimeHashMin = {}
    linkAndTimeHashMinS = {}
    linkAndTimeHashAvg = {}
    linkAndTimeHashAvgS = {}
    linkAndTimeHashStd = {}
    linkAndTimeHashStdS = {}

    for linkID in medianTimeHash.keys():
        for timeID in medianTimeHash[linkID]:
            if preStart <= int(timeID) and int(timeID) < preEnd \
                    or 900 <= int(timeID) and int(timeID) < 960 \
                    or 1080 <= int(timeID) and int(timeID) < 1140:
                valueList = []
                valueListS = []
                for dateID in medianTimeHash[linkID][timeID]:
                    if 0 <= int(dateID) and int(dateID) < 92:
                        valueList.append(medianTimeHash[linkID][timeID][dateID][0])
                        valueListS.append(medianTimeHash[linkID][timeID][dateID][1])
                if len(valueList) != 0:
                    maxValue = np.max(valueList)
                    minValue = np.min(valueList)
                    avgValue = np.average(valueList)
                    stdValue = np.std(valueList)
                    key = linkID + '-' + timeID
                    linkAndTimeHashMax[key] = maxValue
                    linkAndTimeHashMin[key] = minValue
                    linkAndTimeHashAvg[key] = avgValue
                    linkAndTimeHashStd[key] = stdValue

                if len(valueListS) != 0:
                    maxValueS = np.max(valueListS)
                    minValueS = np.min(valueListS)
                    avgValueS = np.average(valueListS)
                    stdValueS = np.std(valueListS)
                    key = linkID + '-' + timeID
                    linkAndTimeHashMaxS[key] = maxValueS
                    linkAndTimeHashMinS[key] = minValueS
                    linkAndTimeHashAvgS[key] = avgValueS
                    linkAndTimeHashStdS[key] = stdValueS

    # get history Rank feature
    linkAndTimeHashRank = {}
    linkAndTimeHashRankS = {}
    linkAndTimeHashPercent = {}
    linkAndTimeHashPercentS = {}

    RankTemp = {}
    RankTempS = {}
    for key in linkAndTimeHash:
        keyList = key.split('-')
        linkID = keyList[0]
        timeID = keyList[1]
        if linkID not in RankTemp:
            RankTemp[linkID] = collections.OrderedDict()
            RankTempS[linkID] = collections.OrderedDict()
        RankTemp[linkID][timeID] = linkAndTimeHash[key]
        RankTempS[linkID][timeID] = linkAndTimeHashS[key]

    for linkID in RankTemp.keys():
        RankTemp[linkID] = sorted(RankTemp[linkID].items(), key=lambda d: d[1])
    for linkID in RankTempS.keys():
        RankTempS[linkID] = sorted(RankTempS[linkID].items(), key=lambda d: d[1])

    for linkID in RankTemp:
        count = 0
        size = len(RankTemp[linkID])
        for timeID in RankTemp[linkID]:
            key = linkID + '-' + timeID[0]
            linkAndTimeHashRank[key] = count
            linkAndTimeHashPercent[key] = count / size
            count = count + 1

    for linkID in RankTempS:
        count = 0
        size = len(RankTemp[linkID])
        for timeID in RankTempS[linkID]:
            key = linkID + '-' + timeID[0]
            linkAndTimeHashRankS[key] = count
            linkAndTimeHashPercentS[key] = count / size
            count = count + 1


    # get the link diff history
    hisLinkDiffAvg = {}
    hisLinkDiffMed = {}
    for key in linkAndTimeHash:
        keyList = key.split("-")
        timeID = keyList[1]
        if timeID not in hisLinkDiffAvg:
            hisLinkDiffAvg[timeID] = []
        if timeID not in hisLinkDiffMed:
            hisLinkDiffMed[timeID] = []
        hisLinkDiffAvg[timeID].append(linkAndTimeHash[key])
        hisLinkDiffMed[timeID].append(linkAndTimeHash[key])

    for timeID in hisLinkDiffAvg:
        hisLinkDiffAvg[timeID] = np.average(hisLinkDiffAvg[timeID])
    for timeID in hisLinkDiffMed:
        hisLinkDiffMed[timeID] = np.median(hisLinkDiffMed[timeID])


    # get the link diff history smooth
    hisLinkDiffAvgS = {}
    hisLinkDiffMedS = {}
    for key in linkAndTimeHashS:
        keyList = key.split("-")
        timeID = keyList[1]
        if timeID not in hisLinkDiffAvgS:
            hisLinkDiffAvgS[timeID] = []
        if timeID not in hisLinkDiffMedS:
            hisLinkDiffMedS[timeID] = []
        hisLinkDiffAvgS[timeID].append(linkAndTimeHashS[key])
        hisLinkDiffMedS[timeID].append(linkAndTimeHashS[key])

    for timeID in hisLinkDiffAvgS:
        hisLinkDiffAvgS[timeID] = np.average(hisLinkDiffAvgS[timeID])
    for timeID in hisLinkDiffMedS:
        hisLinkDiffMedS[timeID] = np.median(hisLinkDiffMedS[timeID])

    # get timeI tempory feature
    linkAndDateHashTimeI = {}
    linkAndDateHashTimeIS = {}
    linkAndDateHashTimeIMax = {}
    linkAndDateHashTimeIMaxS = {}
    linkAndDateHashTimeIMin = {}
    linkAndDateHashTimeIMinS = {}
    linkAndDateHashTimeIAvg = {}
    linkAndDateHashTimeIAvgS = {}
    linkAndDateHashTimeIStd = {}
    linkAndDateHashTimeIStdS = {}

    for linkID in medianTimeHash.keys():
        dateHash = {}
        dateHashS = {}
        for timeID in medianTimeHash[linkID].keys():
            if timeEnd - 60 <= int(timeID) and int(timeID) < timeEnd:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHash:
                        dateHash[dateID] = []
                        dateHashS[dateID] = []
                    dateHash[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
                    dateHashS[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
        for dateID in dateHash.keys():
            value = np.median(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeI[key] = value
        for dateID in dateHashS.keys():
            value = np.median(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIS[key] = value
        # MAX
        for dateID in dateHash.keys():
            value = np.max(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIMax[key] = value
        for dateID in dateHashS.keys():
            value = np.max(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIMaxS[key] = value
        # MIN
        for dateID in dateHash.keys():
            value = np.min(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIMin[key] = value
        for dateID in dateHashS.keys():
            value = np.min(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIMinS[key] = value
        # AVG
        for dateID in dateHash.keys():
            value = np.average(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIAvg[key] = value
        for dateID in dateHashS.keys():
            value = np.average(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIAvgS[key] = value
        # STD
        for dateID in dateHash.keys():
            value = np.std(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIStd[key] = value
        for dateID in dateHashS.keys():
            value = np.std(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIStdS[key] = value

    # get timeII tempory feature:
    linkAndDateHashTimeII = {}
    linkAndDateHashTimeIIS = {}
    linkAndDateHashTimeIIMax = {}
    linkAndDateHashTimeIIMaxS = {}
    linkAndDateHashTimeIIMin = {}
    linkAndDateHashTimeIIMinS = {}
    linkAndDateHashTimeIIAvg = {}
    linkAndDateHashTimeIIAvgS = {}
    linkAndDateHashTimeIIStd = {}
    linkAndDateHashTimeIIStdS = {}

    for linkID in medianTimeHash.keys():
        dateHash = {}
        dateHashS = {}
        for timeID in medianTimeHash[linkID].keys():
            if 840 <= int(timeID) and int(timeID) < 900:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHash:
                        dateHash[dateID] = []
                        dateHashS[dateID] = []
                    dateHash[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
                    dateHashS[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
        for dateID in dateHash.keys():
            value = np.median(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeII[key] = value
        for dateID in dateHashS.keys():
            value = np.median(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIS[key] = value
            # MAX
            for dateID in dateHash.keys():
                value = np.max(dateHash[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIMax[key] = value
            for dateID in dateHashS.keys():
                value = np.max(dateHashS[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIMaxS[key] = value
            # MIN
            for dateID in dateHash.keys():
                value = np.min(dateHash[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIMin[key] = value
            for dateID in dateHashS.keys():
                value = np.min(dateHashS[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIMinS[key] = value
            # AVG
            for dateID in dateHash.keys():
                value = np.average(dateHash[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIAvg[key] = value
            for dateID in dateHashS.keys():
                value = np.average(dateHashS[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIAvgS[key] = value
            # STD
            for dateID in dateHash.keys():
                value = np.std(dateHash[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIStd[key] = value
            for dateID in dateHashS.keys():
                value = np.std(dateHashS[dateID])
                key = linkID + "-" + dateID
                linkAndDateHashTimeIIStdS[key] = value

    # get timeIII tempory feature:
    linkAndDateHashTimeIII = {}
    linkAndDateHashTimeIIIS = {}
    linkAndDateHashTimeIIIMax = {}
    linkAndDateHashTimeIIIMaxS = {}
    linkAndDateHashTimeIIIMin = {}
    linkAndDateHashTimeIIIMinS = {}
    linkAndDateHashTimeIIIAvg = {}
    linkAndDateHashTimeIIIAvgS = {}
    linkAndDateHashTimeIIIStd = {}
    linkAndDateHashTimeIIIStdS = {}

    for linkID in medianTimeHash.keys():
        dateHash = {}
        dateHashS = {}
        for timeID in medianTimeHash[linkID].keys():
            if 1020 <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHash:
                        dateHash[dateID] = []
                        dateHashS[dateID] = []
                    dateHash[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
                    dateHashS[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
        for dateID in dateHash.keys():
            value = np.median(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIII[key] = value
        for dateID in dateHashS.keys():
            value = np.median(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIS[key] = value
        # MAX
        for dateID in dateHash.keys():
            value = np.max(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIMax[key] = value
        for dateID in dateHashS.keys():
            value = np.max(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIMaxS[key] = value
        # MIN
        for dateID in dateHash.keys():
            value = np.min(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIMin[key] = value
        for dateID in dateHashS.keys():
            value = np.min(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIMinS[key] = value
        # AVG
        for dateID in dateHash.keys():
            value = np.average(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIAvg[key] = value
        for dateID in dateHashS.keys():
            value = np.average(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIAvgS[key] = value
            # STD
        for dateID in dateHash.keys():
            value = np.std(dateHash[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIStd[key] = value
        for dateID in dateHashS.keys():
            value = np.std(dateHashS[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashTimeIIIStdS[key] = value

    # get original tempory feature
    linkAndDateHashI = {}
    linkAndDateHashII = {}
    linkAndDateHashIII = {}
    linkAndDateHashIV = {}
    linkAndDateHashV = {}
    linkAndDateHashVI = {}

    for linkID in medianTimeHash.keys():
        dateHashI = {}
        dateHashII = {}
        dateHashIII = {}
        dateHashIV = {}
        for timeID in medianTimeHash[linkID].keys():
            if timeStart <= int(timeID) and int(timeID) < timeEnd \
                    or 780 <= int(timeID) and int(timeID) < 900 \
                    or 960 <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashI:
                        dateHashI[dateID] = []
                    dateHashI[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
            if (timeEnd - 60) <= int(timeID) and int(timeID) < timeEnd \
                    or (900 - 60) <= int(timeID) and int(timeID) < 900 \
                    or (1080 - 60) <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashII:
                        dateHashII[dateID] = []
                    dateHashII[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
            if (timeEnd - 30) <= int(timeID) and int(timeID) < timeEnd \
                    or (900 - 30) <= int(timeID) and int(timeID) < 900 \
                    or (1080 - 30) <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashIII:
                        dateHashIII[dateID] = []
                    dateHashIII[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
            if (timeEnd - 10) <= int(timeID) and int(timeID) < timeEnd \
                    or (900 - 10) <= int(timeID) and int(timeID) < 900 \
                    or (1080 - 10) <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashIV:
                        dateHashIV[dateID] = []
                    dateHashIV[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
        for dateID in dateHashI.keys():
            value = np.median(dateHashI[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashI[key] = value

        for dateID in dateHashII.keys():
            value = np.median(dateHashII[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashII[key] = value

        for dateID in dateHashIII.keys():
            value = np.median(dateHashIII[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashIII[key] = value

        for dateID in dateHashIV.keys():
            value = np.median(dateHashIV[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashIV[key] = value

        for dateID in dateHashIII.keys():
            value = np.mean(dateHashIII[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashV[key] = value

        for dateID in dateHashIV.keys():
            value = np.mean(dateHashIV[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashVI[key] = value

    # get smooth tempory feature
    linkAndDateHashIS = {}
    linkAndDateHashIIS = {}
    linkAndDateHashIIIS = {}
    linkAndDateHashIVS = {}
    linkAndDateHashVS = {}
    linkAndDateHashVIS = {}

    for linkID in medianTimeHash.keys():
        dateHashI = {}
        dateHashII = {}
        dateHashIII = {}
        dateHashIV = {}
        for timeID in medianTimeHash[linkID].keys():
            if timeStart <= int(timeID) and int(timeID) < timeEnd \
                    or 780 <= int(timeID) and int(timeID) < 900 \
                    or 960 <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashI:
                        dateHashI[dateID] = []
                    dateHashI[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
            if (timeEnd - 60) <= int(timeID) and int(timeID) < timeEnd \
                    or (900 - 60) <= int(timeID) and int(timeID) < 900 \
                    or (1080 - 60) <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashII:
                        dateHashII[dateID] = []
                    dateHashII[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
            if (timeEnd - 30) <= int(timeID) and int(timeID) < timeEnd \
                    or (900 - 30) <= int(timeID) and int(timeID) < 900 \
                    or (1080 - 30) <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashIII:
                        dateHashIII[dateID] = []
                    dateHashIII[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
            if (timeEnd - 10) <= int(timeID) and int(timeID) < timeEnd \
                    or (900 - 10) <= int(timeID) and int(timeID) < 900 \
                    or (1080 - 10) <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateHashIV:
                        dateHashIV[dateID] = []
                    dateHashIV[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
        for dateID in dateHashI.keys():
            value = np.median(dateHashI[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashIS[key] = value

        for dateID in dateHashII.keys():
            value = np.median(dateHashII[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashIIS[key] = value

        for dateID in dateHashIII.keys():
            value = np.median(dateHashIII[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashIIIS[key] = value

        for dateID in dateHashIV.keys():
            value = np.median(dateHashIV[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashIVS[key] = value

        for dateID in dateHashIII.keys():
            value = np.mean(dateHashIII[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashVS[key] = value

        for dateID in dateHashIV.keys():
            value = np.mean(dateHashIV[dateID])
            key = linkID + "-" + dateID
            linkAndDateHashVIS[key] = value

    # get template max min avg feature
    linkAndDateHashMax = {}
    linkAndDateHashMin = {}
    linkAndDateHashAvg = {}
    for linkID in medianTimeHash.keys():
        dateList = {}
        for timeID in medianTimeHash[linkID].keys():
            if timeEnd - 60 <= int(timeID) and int(timeID) < timeEnd \
                    or 840 <= int(timeID) and int(timeID) < 900 \
                    or 1020 <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateList:
                        dateList[dateID] = []
                    dateList[dateID].append(medianTimeHash[linkID][timeID][dateID][0])
                for dateID in dateList.keys():
                    max = np.max(dateList[dateID])
                    min = np.min(dateList[dateID])
                    avg = np.average(dateList[dateID])
                    key = linkID + "-" + dateID
                    linkAndDateHashMax[key] = max
                    linkAndDateHashMin[key] = min
                    linkAndDateHashAvg[key] = avg

    # get sommth template max min avg feature
    linkAndDateHashMaxS = {}
    linkAndDateHashMinS = {}
    linkAndDateHashAvgS = {}
    for linkID in medianTimeHash.keys():
        dateList = {}
        for timeID in medianTimeHash[linkID].keys():
            if timeEnd - 60 <= int(timeID) and int(timeID) < timeEnd \
                    or 840 <= int(timeID) and int(timeID) < 900 \
                    or 1020 <= int(timeID) and int(timeID) < 1080:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if dateID not in dateList:
                        dateList[dateID] = []
                    dateList[dateID].append(medianTimeHash[linkID][timeID][dateID][1])
                for dateID in dateList.keys():
                    max = np.max(dateList[dateID])
                    min = np.min(dateList[dateID])
                    avg = np.average(dateList[dateID])
                    key = linkID + "-" + dateID
                    linkAndDateHashMaxS[key] = max
                    linkAndDateHashMinS[key] = min
                    linkAndDateHashAvgS[key] = avg

    #WRONG get template avg for diff(TIME I):
    # linkAndDateHash contains just one hour data

    templateAvgTimeIW = 0
    count = 0
    for i in linkAndDateHashTimeI:
        templateAvgTimeIW = templateAvgTimeIW + linkAndDateHashTimeI[i]
        count = count + 1
    templateAvgTimeI = templateAvgTimeIW / count

    #WRONG get template avg for diff(TIME II):
    templateAvgTimeIIW = 0
    count = 0
    for i in linkAndDateHashTimeII:
        templateAvgTimeIIW = templateAvgTimeIIW + linkAndDateHashTimeII[i]
        count = count + 1
    templateAvgTimeII = templateAvgTimeIIW / count

    #WRONG get template avg for diff(TIME II):
    templateAvgTimeIIIW = 0
    count = 0
    for i in linkAndDateHashTimeIII:
        templateAvgTimeIIIW = templateAvgTimeIIIW + linkAndDateHashTimeIII[i]
        count = count + 1
    templateAvgTimeIIIW = templateAvgTimeIIIW / count


    # get template avg for diff(TIME I):
    # linkAndDateHash contains just one hour data

    templateAvgTimeI = {}
    for key in linkAndDateHashTimeI:
        keyList = key.split('-')
        linkID = keyList[0]
        if linkID not in templateAvgTimeI:
            templateAvgTimeI[linkID] = []
        templateAvgTimeI[linkID].append(linkAndDateHashTimeI[key])
    for linkID in templateAvgTimeI:
        templateAvgTimeI[linkID] = np.mean(templateAvgTimeI[linkID])

    # get template avg for diff(TIME II):
    templateAvgTimeII = {}
    for key in linkAndDateHashTimeII:
        keyList = key.split('-')
        linkID = keyList[0]
        if linkID not in templateAvgTimeII:
            templateAvgTimeII[linkID] = []
        templateAvgTimeII[linkID].append(linkAndDateHashTimeII[key])
    for linkID in templateAvgTimeII:
        templateAvgTimeII[linkID] = np.mean(templateAvgTimeII[linkID])

    # get template avg for diff(TIME III):
    templateAvgTimeIII = {}
    for key in linkAndDateHashTimeIII:
        keyList = key.split('-')
        linkID = keyList[0]
        if linkID not in templateAvgTimeIII:
            templateAvgTimeIII[linkID] = []
        templateAvgTimeIII[linkID].append(linkAndDateHashTimeIII[key])
    for linkID in templateAvgTimeIII:
        templateAvgTimeIII[linkID] = np.mean(templateAvgTimeIII[linkID])

    # get rank of each day in template time one hour(TIME I):
    linkAndDateHashTimeIRank = {}
    linkAndDateHashTimeIPercent = {}
    linkAndDateHashTimeIMaxRank = {}
    linkAndDateHashTimeIMaxPercent = {}
    linkAndDateHashTimeIMinRank = {}
    linkAndDateHashTimeIMinPercent = {}
    linkAndDateHashTimeIAvgRank = {}
    linkAndDateHashTimeIAvgPercent = {}
    linkAndDateHashTimeIRankS = {}
    linkAndDateHashTimeIPercentS = {}
    linkAndDateHashTimeIMaxRankS = {}
    linkAndDateHashTimeIMaxPercentS = {}
    linkAndDateHashTimeIMinRankS = {}
    linkAndDateHashTimeIMinPercentS = {}
    linkAndDateHashTimeIAvgRankS = {}
    linkAndDateHashTimeIAvgPercentS = {}

    tempHash = {}
    tempHashMax = {}
    tempHashMin = {}
    tempHashAvg = {}

    tempHashS = {}
    tempHashMaxS = {}
    tempHashMinS = {}
    tempHashAvgS = {}
    for key in linkAndDateHashTimeI.keys():
        keyList = key.split('-')
        linkID = keyList[0]
        dateID = keyList[1]
        if linkID not in tempHash:
            tempHash[linkID] = collections.OrderedDict()
        if linkID not in tempHashMax:
            tempHashMax[linkID] = collections.OrderedDict()
        if linkID not in tempHashMin:
            tempHashMin[linkID] = collections.OrderedDict()
        if linkID not in tempHashAvg:
            tempHashAvg[linkID] = collections.OrderedDict()
        if linkID not in tempHashS:
            tempHashS[linkID] = collections.OrderedDict()
        if linkID not in tempHashMaxS:
            tempHashMaxS[linkID] = collections.OrderedDict()
        if linkID not in tempHashMinS:
            tempHashMinS[linkID] = collections.OrderedDict()
        if linkID not in tempHashAvgS:
            tempHashAvgS[linkID] = collections.OrderedDict()

        tempHash[linkID][dateID] = linkAndDateHashTimeI[key]
        tempHashMax[linkID][dateID] = linkAndDateHashTimeIMax[key]
        tempHashMin[linkID][dateID] = linkAndDateHashTimeIMin[key]
        tempHashAvg[linkID][dateID] = linkAndDateHashTimeIAvg[key]
        tempHashS[linkID][dateID] = linkAndDateHashTimeIS[key]
        tempHashMaxS[linkID][dateID] = linkAndDateHashTimeIMaxS[key]
        tempHashMinS[linkID][dateID] = linkAndDateHashTimeIMinS[key]
        tempHashAvgS[linkID][dateID] = linkAndDateHashTimeIAvgS[key]

    for linkID in tempHash.keys():
        tempHash[linkID] = sorted(tempHash[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMax.keys():
        tempHashMax[linkID] = sorted(tempHashMax[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMin.keys():
        tempHashMin[linkID] = sorted(tempHashMin[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashAvg.keys():
        tempHashAvg[linkID] = sorted(tempHashAvg[linkID].items(), key=lambda d: d[1])
    for linkID in tempHash.keys():
        tempHashS[linkID] = sorted(tempHashS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMax.keys():
        tempHashMaxS[linkID] = sorted(tempHashMaxS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMin.keys():
        tempHashMinS[linkID] = sorted(tempHashMinS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashAvg.keys():
        tempHashAvgS[linkID] = sorted(tempHashAvgS[linkID].items(), key=lambda d: d[1])

    for linkID in tempHash:
        count = 0
        size = len(tempHash[linkID])
        for dateID in tempHash[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIRank[key] = count
            linkAndDateHashTimeIPercent[key] = count / size
            count = count + 1

    for linkID in tempHashMax:
        count = 0
        size = len(tempHashMax[linkID])
        for dateID in tempHashMax[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIMaxRank[key] = count
            linkAndDateHashTimeIMaxPercent[key] = count / size
            count = count + 1

    for linkID in tempHashMin:
        count = 0
        size = len(tempHashMin[linkID])
        for dateID in tempHashMin[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIMinRank[key] = count
            linkAndDateHashTimeIMinPercent[key] = count / size
            count = count + 1

    for linkID in tempHashAvg:
        count = 0
        size = len(tempHashAvg[linkID])
        for dateID in tempHashAvg[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIAvgRank[key] = count
            linkAndDateHashTimeIAvgPercent[key] = count / size
            count = count + 1


    #GET THE SMOOTH FEATURE
    for linkID in tempHashS:
        count = 0
        size = len(tempHashS[linkID])
        for dateID in tempHashS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIRankS[key] = count
            linkAndDateHashTimeIPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashMaxS:
        count = 0
        size = len(tempHashMaxS[linkID])
        for dateID in tempHashMaxS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIMaxRankS[key] = count
            linkAndDateHashTimeIMaxPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashMinS:
        count = 0
        size = len(tempHashMinS[linkID])
        for dateID in tempHashMinS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIMinRankS[key] = count
            linkAndDateHashTimeIMinPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashAvgS:
        count = 0
        size = len(tempHashAvgS[linkID])
        for dateID in tempHashAvgS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIAvgRankS[key] = count
            linkAndDateHashTimeIAvgPercentS[key] = count / size
            count = count + 1

    # get rank of each day in template time one hour(TIME II):
    linkAndDateHashTimeIIRank = {}
    linkAndDateHashTimeIIPercent = {}
    linkAndDateHashTimeIIMaxRank = {}
    linkAndDateHashTimeIIMaxPercent = {}
    linkAndDateHashTimeIIMinRank = {}
    linkAndDateHashTimeIIMinPercent = {}
    linkAndDateHashTimeIIAvgRank = {}
    linkAndDateHashTimeIIAvgPercent = {}
    linkAndDateHashTimeIIRankS = {}
    linkAndDateHashTimeIIPercentS = {}
    linkAndDateHashTimeIIMaxRankS = {}
    linkAndDateHashTimeIIMaxPercentS = {}
    linkAndDateHashTimeIIMinRankS = {}
    linkAndDateHashTimeIIMinPercentS = {}
    linkAndDateHashTimeIIAvgRankS = {}
    linkAndDateHashTimeIIAvgPercentS = {}

    tempHash = {}
    tempHashMax = {}
    tempHashMin = {}
    tempHashAvg = {}

    tempHashS = {}
    tempHashMaxS = {}
    tempHashMinS = {}
    tempHashAvgS = {}
    for key in linkAndDateHashTimeI.keys():
        keyList = key.split('-')
        linkID = keyList[0]
        dateID = keyList[1]
        if linkID not in tempHash:
            tempHash[linkID] = collections.OrderedDict()
        if linkID not in tempHashMax:
            tempHashMax[linkID] = collections.OrderedDict()
        if linkID not in tempHashMin:
            tempHashMin[linkID] = collections.OrderedDict()
        if linkID not in tempHashAvg:
            tempHashAvg[linkID] = collections.OrderedDict()
        if linkID not in tempHashS:
            tempHashS[linkID] = collections.OrderedDict()
        if linkID not in tempHashMaxS:
            tempHashMaxS[linkID] = collections.OrderedDict()
        if linkID not in tempHashMinS:
            tempHashMinS[linkID] = collections.OrderedDict()
        if linkID not in tempHashAvgS:
            tempHashAvgS[linkID] = collections.OrderedDict()

        tempHash[linkID][dateID] = linkAndDateHashTimeII[key]
        tempHashMax[linkID][dateID] = linkAndDateHashTimeIIMax[key]
        tempHashMin[linkID][dateID] = linkAndDateHashTimeIIMin[key]
        tempHashAvg[linkID][dateID] = linkAndDateHashTimeIIAvg[key]
        tempHashS[linkID][dateID] = linkAndDateHashTimeIIS[key]
        tempHashMaxS[linkID][dateID] = linkAndDateHashTimeIIMaxS[key]
        tempHashMinS[linkID][dateID] = linkAndDateHashTimeIIMinS[key]
        tempHashAvgS[linkID][dateID] = linkAndDateHashTimeIIAvgS[key]

    for linkID in tempHash.keys():
        tempHash[linkID] = sorted(tempHash[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMax.keys():
        tempHashMax[linkID] = sorted(tempHashMax[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMin.keys():
        tempHashMin[linkID] = sorted(tempHashMin[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashAvg.keys():
        tempHashAvg[linkID] = sorted(tempHashAvg[linkID].items(), key=lambda d: d[1])
    for linkID in tempHash.keys():
        tempHashS[linkID] = sorted(tempHashS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMax.keys():
        tempHashMaxS[linkID] = sorted(tempHashMaxS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMin.keys():
        tempHashMinS[linkID] = sorted(tempHashMinS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashAvg.keys():
        tempHashAvgS[linkID] = sorted(tempHashAvgS[linkID].items(), key=lambda d: d[1])

    for linkID in tempHash:
        count = 0
        size = len(tempHash[linkID])
        for dateID in tempHash[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIRank[key] = count
            linkAndDateHashTimeIIPercent[key] = count / size
            count = count + 1

    for linkID in tempHashMax:
        count = 0
        size = len(tempHashMax[linkID])
        for dateID in tempHashMax[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIMaxRank[key] = count
            linkAndDateHashTimeIIMaxPercent[key] = count / size
            count = count + 1

    for linkID in tempHashMin:
        count = 0
        size = len(tempHashMin[linkID])
        for dateID in tempHashMin[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIMinRank[key] = count
            linkAndDateHashTimeIIMinPercent[key] = count / size
            count = count + 1

    for linkID in tempHashAvg:
        count = 0
        size = len(tempHashAvg[linkID])
        for dateID in tempHashAvg[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIAvgRank[key] = count
            linkAndDateHashTimeIIAvgPercent[key] = count / size
            count = count + 1

    for linkID in tempHashS:
        count = 0
        size = len(tempHashS[linkID])
        for dateID in tempHashS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIRankS[key] = count
            linkAndDateHashTimeIIPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashMaxS:
        count = 0
        size = len(tempHashMaxS[linkID])
        for dateID in tempHashMaxS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIMaxRankS[key] = count
            linkAndDateHashTimeIIMaxPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashMinS:
        count = 0
        size = len(tempHashMinS[linkID])
        for dateID in tempHashMinS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIMinRankS[key] = count
            linkAndDateHashTimeIIMinPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashAvgS:
        count = 0
        size = len(tempHashAvgS[linkID])
        for dateID in tempHashAvgS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIAvgRankS[key] = count
            linkAndDateHashTimeIIAvgPercentS[key] = count / size
            count = count + 1

    # get rank of each day in template time one hour(TIME III):
    linkAndDateHashTimeIIIRank = {}
    linkAndDateHashTimeIIIPercent = {}
    linkAndDateHashTimeIIIMaxRank = {}
    linkAndDateHashTimeIIIMaxPercent = {}
    linkAndDateHashTimeIIIMinRank = {}
    linkAndDateHashTimeIIIMinPercent = {}
    linkAndDateHashTimeIIIAvgRank = {}
    linkAndDateHashTimeIIIAvgPercent = {}
    linkAndDateHashTimeIIIRankS = {}
    linkAndDateHashTimeIIIPercentS = {}
    linkAndDateHashTimeIIIMaxRankS = {}
    linkAndDateHashTimeIIIMaxPercentS = {}
    linkAndDateHashTimeIIIMinRankS = {}
    linkAndDateHashTimeIIIMinPercentS = {}
    linkAndDateHashTimeIIIAvgRankS = {}
    linkAndDateHashTimeIIIAvgPercentS = {}

    tempHash = {}
    tempHashMax = {}
    tempHashMin = {}
    tempHashAvg = {}
    tempHashS = {}
    tempHashMaxS = {}
    tempHashMinS = {}
    tempHashAvgS = {}

    for key in linkAndDateHashTimeIII.keys():
        keyList = key.split('-')
        linkID = keyList[0]
        dateID = keyList[1]
        if linkID not in tempHash:
            tempHash[linkID] = collections.OrderedDict()
        if linkID not in tempHashMax:
            tempHashMax[linkID] = collections.OrderedDict()
        if linkID not in tempHashMin:
            tempHashMin[linkID] = collections.OrderedDict()
        if linkID not in tempHashAvg:
            tempHashAvg[linkID] = collections.OrderedDict()
        if linkID not in tempHashS:
            tempHashS[linkID] = collections.OrderedDict()
        if linkID not in tempHashMaxS:
            tempHashMaxS[linkID] = collections.OrderedDict()
        if linkID not in tempHashMinS:
            tempHashMinS[linkID] = collections.OrderedDict()
        if linkID not in tempHashAvgS:
            tempHashAvgS[linkID] = collections.OrderedDict()

        tempHash[linkID][dateID] = linkAndDateHashTimeIII[key]
        tempHashMax[linkID][dateID] = linkAndDateHashTimeIIIMax[key]
        tempHashMin[linkID][dateID] = linkAndDateHashTimeIIIMin[key]
        tempHashAvg[linkID][dateID] = linkAndDateHashTimeIIIAvg[key]
        tempHashS[linkID][dateID] = linkAndDateHashTimeIIIS[key]
        tempHashMaxS[linkID][dateID] = linkAndDateHashTimeIIIMaxS[key]
        tempHashMinS[linkID][dateID] = linkAndDateHashTimeIIIMinS[key]
        tempHashAvgS[linkID][dateID] = linkAndDateHashTimeIIIAvgS[key]

    for linkID in tempHash.keys():
        tempHash[linkID] = sorted(tempHash[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMax.keys():
        tempHashMax[linkID] = sorted(tempHashMax[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMin.keys():
        tempHashMin[linkID] = sorted(tempHashMin[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashAvg.keys():
        tempHashAvg[linkID] = sorted(tempHashAvg[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashS.keys():
        tempHashS[linkID] = sorted(tempHashS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMaxS.keys():
        tempHashMaxS[linkID] = sorted(tempHashMaxS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashMinS.keys():
        tempHashMinS[linkID] = sorted(tempHashMinS[linkID].items(), key=lambda d: d[1])
    for linkID in tempHashAvgS.keys():
        tempHashAvgS[linkID] = sorted(tempHashAvgS[linkID].items(), key=lambda d: d[1])

    for linkID in tempHash:
        count = 0
        size = len(tempHash[linkID])
        for dateID in tempHash[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIRank[key] = count
            linkAndDateHashTimeIIIPercent[key] = count / size
            count = count + 1

    for linkID in tempHashMax:
        count = 0
        size = len(tempHashMax[linkID])
        for dateID in tempHashMax[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIMaxRank[key] = count
            linkAndDateHashTimeIIIMaxPercent[key] = count / size
            count = count + 1

    for linkID in tempHashMin:
        count = 0
        size = len(tempHashMin[linkID])
        for dateID in tempHashMin[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIMinRank[key] = count
            linkAndDateHashTimeIIIMinPercent[key] = count / size
            count = count + 1

    for linkID in tempHashAvg:
        count = 0
        size = len(tempHashAvg[linkID])
        for dateID in tempHashAvg[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIAvgRank[key] = count
            linkAndDateHashTimeIIIAvgPercent[key] = count / size
            count = count + 1

    for linkID in tempHashS:
        count = 0
        size = len(tempHashS[linkID])
        for dateID in tempHashS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIRankS[key] = count
            linkAndDateHashTimeIIIPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashMaxS:
        count = 0
        size = len(tempHashMaxS[linkID])
        for dateID in tempHashMaxS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIMaxRankS[key] = count
            linkAndDateHashTimeIIIMaxPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashMinS:
        count = 0
        size = len(tempHashMinS[linkID])
        for dateID in tempHashMinS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIMinRankS[key] = count
            linkAndDateHashTimeIIIMinPercentS[key] = count / size
            count = count + 1

    for linkID in tempHashAvgS:
        count = 0
        size = len(tempHashAvgS[linkID])
        for dateID in tempHashAvgS[linkID]:
            key = linkID + '-' + dateID[0]
            linkAndDateHashTimeIIIAvgRankS[key] = count
            linkAndDateHashTimeIIIAvgPercentS[key] = count / size
            count = count + 1

    # get template feature diff avg I
    tempLinkDiffAvgI = {}
    tempLinkDiffMedI = {}
    for key in linkAndDateHashTimeI:
        keyList = key.split("-")
        dateID = keyList[1]
        if dateID not in tempLinkDiffAvgI:
            tempLinkDiffAvgI[dateID] = []
        if dateID not in tempLinkDiffMedI:
            tempLinkDiffMedI[dateID] = []
        tempLinkDiffAvgI[dateID].append(linkAndDateHashTimeI[key])
        tempLinkDiffMedI[dateID].append(linkAndDateHashTimeI[key])
    for dateID in tempLinkDiffAvgI:
        tempLinkDiffAvgI[dateID] = np.median(tempLinkDiffAvgI[dateID])
    for dateID in tempLinkDiffMedI:
        tempLinkDiffMedI[dateID] = np.median(tempLinkDiffMedI[dateID])

    # get template feature diff avg II
    tempLinkDiffAvgII = {}
    tempLinkDiffMedII = {}
    for key in linkAndDateHashTimeII:
        keyList = key.split("-")
        dateID = keyList[1]
        if dateID not in tempLinkDiffAvgII:
            tempLinkDiffAvgII[dateID] = []
        if dateID not in tempLinkDiffMedII:
            tempLinkDiffMedII[dateID] = []
        tempLinkDiffAvgII[dateID].append(linkAndDateHashTimeII[key])
        tempLinkDiffMedII[dateID].append(linkAndDateHashTimeII[key])
    for dateID in tempLinkDiffAvgII:
        tempLinkDiffAvgII[dateID] = np.median(tempLinkDiffAvgII[dateID])
    for dateID in tempLinkDiffMedII:
        tempLinkDiffMedII[dateID] = np.median(tempLinkDiffMedII[dateID])

    # get template feature diff avg III
    tempLinkDiffAvgIII = {}
    tempLinkDiffMedIII = {}
    for key in linkAndDateHashTimeIII:
        keyList = key.split("-")
        dateID = keyList[1]
        if dateID not in tempLinkDiffAvgIII:
            tempLinkDiffAvgIII[dateID] = []
        if dateID not in tempLinkDiffMedIII:
            tempLinkDiffMedIII[dateID] = []
        tempLinkDiffAvgIII[dateID].append(linkAndDateHashTimeIII[key])
        tempLinkDiffMedIII[dateID].append(linkAndDateHashTimeIII[key])
    for dateID in tempLinkDiffAvgIII:
        tempLinkDiffAvgIII[dateID] = np.median(tempLinkDiffAvgIII[dateID])
    for dateID in tempLinkDiffMedIII:
        tempLinkDiffMedIII[dateID] = np.median(tempLinkDiffMedIII[dateID])

    # get the original week feature:
    linkAndWeekHash = {}
    for linkID in medianTimeHash.keys():
        for timeID in medianTimeHash[linkID].keys():
            if preStart <= int(timeID) and int(timeID) < preEnd \
                    or 900 <= int(timeID) and int(timeID) < 960 \
                    or 1080 <= int(timeID) and int(timeID) < 1140:
                weekHash = {}
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if 0 <= int(dateID) and int(dateID) < 63:
                        week = str(int(dateID) % 7)
                        if week not in weekHash:
                            weekHash[week] = []
                        weekHash[week].append(medianTimeHash[linkID][timeID][dateID][0])
                for week in weekHash.keys():
                    value = np.median(weekHash[week])
                    key = linkID + "-" + week
                    linkAndWeekHash[key] = value

    # get the smooth week feature:
    linkAndWeekHashS = {}
    for linkID in medianTimeHash.keys():
        for timeID in medianTimeHash[linkID].keys():
            if preStart <= int(timeID) and int(timeID) < preEnd \
                    or 900 <= int(timeID) and int(timeID) < 960 \
                    or 1080 <= int(timeID) and int(timeID) < 1140:
                weekHash = {}
                for dateID in medianTimeHash[linkID][timeID].keys():
                    if 0 <= int(dateID) and int(dateID) < 63:
                        week = str(int(dateID) % 7)
                        if week not in weekHash:
                            weekHash[week] = []
                        weekHash[week].append(medianTimeHash[linkID][timeID][dateID][1])
                for week in weekHash.keys():
                    value = np.median(weekHash[week])
                    key = linkID + "-" + week
                    linkAndWeekHashS[key] = value


    #get train and test:
    trainVector = []
    lableVector = []
    weightVector = []

    count = 0
    for linkID in medianTimeHash.keys():
        for timeID in medianTimeHash[linkID].keys():
            if preStart <= int(timeID) and int(timeID) < preEnd\
                    or 900 <= int(timeID) and int(timeID) < 960\
                    or 1080 <= int(timeID) and int(timeID) < 1140:
                for dateID in medianTimeHash[linkID][timeID].keys():
                    linkAndTimeKey = linkID + "-" + timeID
                    linkAndDateKey = linkID + "-" + dateID
                    linkAndWeekKey = linkID + "-" + str(int(dateID) % 7)
                    if 63 <= int(dateID) and int(dateID) < 92:
                        if linkAndTimeKey not in linkAndTimeHash:
                            break
                        if linkAndDateKey not in linkAndDateHashI:
                            break
                        if linkAndDateKey not in linkAndDateHashII:
                            break
                        if linkAndDateKey not in linkAndDateHashIII:
                            break
                        if linkAndDateKey not in linkAndDateHashIV:
                            break
                        if linkAndDateKey not in linkAndDateHashV:
                            break
                        if linkAndDateKey not in linkAndDateHashVI:
                            break
                        if linkAndTimeKey not in linkAndTimeHashS:
                            break
                        if linkAndDateKey not in linkAndDateHashIS:
                            break
                        if linkAndDateKey not in linkAndDateHashIIS:
                            break
                        if linkAndDateKey not in linkAndDateHashIIIS:
                            break
                        if linkAndDateKey not in linkAndDateHashIVS:
                            break
                        if linkAndDateKey not in linkAndDateHashVS:
                            break
                        if linkAndDateKey not in linkAndDateHashVIS:
                            break
                        if linkAndDateKey not in linkAndDateHashTimeI:
                            break
                        if linkAndDateKey not in linkAndDateHashTimeII:
                            break
                        if linkAndDateKey not in linkAndDateHashTimeIII:
                            break
                        count = count + 1
                        print(count)
                        sample = []
                        sample.append(linkID)
                        sample.append(timeID)
                        sample.append(dateID)
                        sample.append(linkAndTimeHash[linkAndTimeKey])
                        sample.append(linkAndTimeHashS[linkAndTimeKey])
                        sample.append(linkAndTimeHashM[linkAndTimeKey])
                        sample.append(linkAndTimeHashMS[linkAndTimeKey])
                        sample.append(linkAndTimeHashMax[linkAndTimeKey])
                        sample.append(linkAndTimeHashMaxS[linkAndTimeKey])
                        sample.append(linkAndTimeHashMin[linkAndTimeKey])
                        sample.append(linkAndTimeHashMinS[linkAndTimeKey])
                        sample.append(linkAndTimeHashAvg[linkAndTimeKey])
                        sample.append(linkAndTimeHashAvgS[linkAndTimeKey])
                        sample.append(linkAndTimeHashStd[linkAndTimeKey])
                        sample.append(linkAndTimeHashStdS[linkAndTimeKey])
                        sample.append(linkAndTimeHashRank[linkAndTimeKey])
                        sample.append(linkAndTimeHashRankS[linkAndTimeKey])
                        sample.append(linkAndTimeHashPercent[linkAndTimeKey])
                        sample.append(linkAndTimeHashPercentS[linkAndTimeKey])
                        sample.append(linkAndDateHashI[linkAndDateKey])
                        sample.append(linkAndDateHashII[linkAndDateKey])
                        sample.append(linkAndDateHashIII[linkAndDateKey])
                        sample.append(linkAndDateHashIV[linkAndDateKey])
                        sample.append(linkAndDateHashIS[linkAndDateKey])
                        sample.append(linkAndDateHashIIS[linkAndDateKey])
                        sample.append(linkAndDateHashIIIS[linkAndDateKey])
                        sample.append(linkAndDateHashIVS[linkAndDateKey])
                        sample.append(linkAndDateHashMax[linkAndDateKey])
                        sample.append(linkAndDateHashMin[linkAndDateKey])
                        sample.append(linkAndDateHashAvg[linkAndDateKey])
                        sample.append(linkAndDateHashMaxS[linkAndDateKey])
                        sample.append(linkAndDateHashMinS[linkAndDateKey])
                        sample.append(linkAndDateHashAvgS[linkAndDateKey])
                        sample.append(linkAndTimeHash[linkAndTimeKey] - hisLinkDiffAvg[timeID])
                        sample.append(linkAndTimeHashS[linkAndTimeKey] - hisLinkDiffAvgS[timeID])
                        sample.append(linkAndTimeHash[linkAndTimeKey] - hisLinkDiffMed[timeID])
                        sample.append(linkAndTimeHash[linkAndTimeKey] - hisLinkDiffMedS[timeID])
                        if preStart <= int(timeID) and int(timeID) < preEnd:
                            sample.append(linkAndDateHashTimeI[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMax[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMin[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvg[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - templateAvgTimeI[linkID])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - templateAvgTimeIW)
                            sample.append(linkAndDateHashTimeIStd[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIStdS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - tempLinkDiffAvgI[dateID])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - tempLinkDiffMedI[dateID])


                        if 900 <= int(timeID) and int(timeID) < 960:
                            sample.append(linkAndDateHashTimeII[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMax[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMin[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvg[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - templateAvgTimeII[linkID])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - templateAvgTimeIIW)
                            sample.append(linkAndDateHashTimeIIStd[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIStdS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - tempLinkDiffAvgII[dateID])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - tempLinkDiffMedII[dateID])

                        if 1080 <= int(timeID) and int(timeID) < 1140:
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMax[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMin[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvg[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - templateAvgTimeIII[linkID])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - templateAvgTimeIIIW)
                            sample.append(linkAndDateHashTimeIIIStd[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIStdS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - tempLinkDiffAvgIII[dateID])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - tempLinkDiffMedIII[dateID])
                        if linkAndTimeKey not in linkAndTimeHash or linkAndTimeKey not in linkAndTimeHashRank:
                            sample.append(float("nan"))
                        else:
                            sample.append(linkAndTimeHash[linkAndTimeKey]*linkAndTimeHashPercent[linkAndTimeKey])

                        if linkAndTimeKey not in linkAndTimeHashS or linkAndTimeKey not in linkAndTimeHashRank:
                            sample.append(float("nan"))
                        else:
                            sample.append(linkAndTimeHashS[linkAndTimeKey]*linkAndTimeHashPercentS[linkAndTimeKey])
                        sample.append(linkAndWeekHash[linkAndWeekKey])
                        sample.append(linkAndWeekHashS[linkAndWeekKey])
                        sample.append(linkInfoHash[linkID][0])
                        sample.append(linkInfoHash[linkID][1])
                        sample.append(linkInfoHash[linkID][0] * linkInfoHash[linkID][1])
                        #for write file
                        sample.append(medianTimeHash[linkID][timeID][dateID][0])
                        #weight = 1.0 / (91 - int(dateID) + 0.4 - int(timeID)/(60 * 24))
                        #weightVector.append(weight)
                        trainVector.append(sample)
                        lableVector.append(medianTimeHash[linkID][timeID][dateID][0])

    testVector = []
    missingVector = []
    linkIndexVector = []
    timeIndexVector = []
    dateIndexVector = []
    count = 0
    for linkID in medianTimeHash.keys():
        for timeID in medianTimeHash[linkID].keys():
            if preStart <= int(timeID) and int(timeID) < preEnd\
                    or 900 <= int(timeID) and int(timeID) < 960\
                    or 1080 <= int(timeID) and int(timeID) < 1140:
                for dateID in range(92, 122):
                    linkAndTimeKey = linkID + "-" + timeID
                    linkAndDateKey = linkID + "-" + str(dateID)
                    linkAndWeekKey = linkID + "-" + str(dateID % 7)

                    sample = []
                    sample.append(linkID)
                    sample.append(timeID)
                    sample.append(dateID)
                    sample.append(linkAndTimeHash[linkAndTimeKey])
                    sample.append(linkAndTimeHashS[linkAndTimeKey])
                    if linkAndTimeKey not in linkAndTimeHashM:
                        sample.append(float('nan'))
                        sample.append(float('nan'))
                    else:
                        sample.append(linkAndTimeHashM[linkAndTimeKey])
                        sample.append(linkAndTimeHashMS[linkAndTimeKey])
                    if linkAndTimeKey not in linkAndTimeHash:
                        for i in range(0, 12):
                            sample.append(float("nan"))
                    else:
                        sample.append(linkAndTimeHashMax[linkAndTimeKey])
                        sample.append(linkAndTimeHashMaxS[linkAndTimeKey])
                        sample.append(linkAndTimeHashMin[linkAndTimeKey])
                        sample.append(linkAndTimeHashMinS[linkAndTimeKey])
                        sample.append(linkAndTimeHashAvg[linkAndTimeKey])
                        sample.append(linkAndTimeHashAvgS[linkAndTimeKey])
                        sample.append(linkAndTimeHashStd[linkAndTimeKey])
                        sample.append(linkAndTimeHashStdS[linkAndTimeKey])
                        sample.append(linkAndTimeHashRank[linkAndTimeKey])
                        sample.append(linkAndTimeHashRankS[linkAndTimeKey])
                        sample.append(linkAndTimeHashPercent[linkAndTimeKey])
                        sample.append(linkAndTimeHashPercentS[linkAndTimeKey])
                    if linkAndDateKey not in linkAndDateHashI:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashI[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashII:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashII[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashIII:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashIII[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashIV:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashIV[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashIS:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashIS[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashIIS:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashIIS[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashIIIS:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashIIIS[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashIVS:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashIVS[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashMax:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashMax[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashMin:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashMin[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashAvg:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashAvg[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashMax:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashMaxS[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashMinS:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashMinS[linkAndDateKey])
                    if linkAndDateKey not in linkAndDateHashAvgS:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndDateHashAvgS[linkAndDateKey])
                    if linkAndTimeKey not in linkAndTimeHash:
                        for i in range(0, 4):
                            sample.append(float("nan"))
                    else:
                        sample.append(linkAndTimeHash[linkAndTimeKey] - hisLinkDiffAvg[timeID])
                        sample.append(linkAndTimeHashS[linkAndTimeKey] - hisLinkDiffAvgS[timeID])
                        sample.append(linkAndTimeHash[linkAndTimeKey] - hisLinkDiffMed[timeID])
                        sample.append(linkAndTimeHash[linkAndTimeKey] - hisLinkDiffMedS[timeID])

                    if preStart <= int(timeID) and int(timeID) < preEnd:
                        if linkAndDateKey not in linkAndDateHashTimeI:
                            for i in range(0, 30):
                                sample.append(float("nan"))
                        else:
                            sample.append(linkAndDateHashTimeI[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMax[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMin[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvg[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMaxPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIMinPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIAvgPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - templateAvgTimeI[linkID])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - templateAvgTimeIW)
                            sample.append(linkAndDateHashTimeIStd[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIStdS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - tempLinkDiffAvgI[str(dateID)])
                            sample.append(linkAndDateHashTimeI[linkAndDateKey] - tempLinkDiffMedI[str(dateID)])


                    if 900 <= int(timeID) and int(timeID) < 960:
                        if linkAndDateKey not in linkAndDateHashTimeI:
                            for i in range(0, 30):
                                sample.append(float("nan"))
                        else:
                            sample.append(linkAndDateHashTimeII[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMax[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMin[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvg[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMaxPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIMinPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIAvgPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - templateAvgTimeII[linkID])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - templateAvgTimeIIW)
                            sample.append(linkAndDateHashTimeIIStd[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIStdS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - tempLinkDiffAvgII[str(dateID)])
                            sample.append(linkAndDateHashTimeII[linkAndDateKey] - tempLinkDiffMedII[str(dateID)])

                    if 1080 <= int(timeID) and int(timeID) < 1140:
                        if linkAndDateKey not in linkAndDateHashTimeI:
                            for i in range(0, 30):
                                sample.append(float("nan"))
                        else:
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMax[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMin[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvg[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMaxPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIMinPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgRank[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgRankS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgPercent[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIAvgPercentS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - templateAvgTimeIII[linkID])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - templateAvgTimeIIIW)
                            sample.append(linkAndDateHashTimeIIIStd[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIIIStdS[linkAndDateKey])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - tempLinkDiffAvgIII[str(dateID)])
                            sample.append(linkAndDateHashTimeIII[linkAndDateKey] - tempLinkDiffMedIII[str(dateID)])
                        # -1 means this sample has a correct feature
                        missingVector.append(-1)
                    if linkAndTimeKey not in linkAndTimeHash or linkAndTimeKey not in linkAndTimeHashRank:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndTimeHash[linkAndTimeKey] * linkAndTimeHashPercent[linkAndTimeKey])

                    if linkAndTimeKey not in linkAndTimeHashS or linkAndTimeKey not in linkAndTimeHashRank:
                        sample.append(float("nan"))
                    else:
                        sample.append(linkAndTimeHashS[linkAndTimeKey] * linkAndTimeHashPercentS[linkAndTimeKey])
                    if linkAndWeekKey not in linkAndWeekHash:
                        for i in range(0, 2):
                            sample.append(float("nan"))
                    else:
                        sample.append(linkAndWeekHash[linkAndWeekKey])
                        sample.append(linkAndWeekHashS[linkAndWeekKey])
                    sample.append(linkInfoHash[linkID][0])
                    sample.append(linkInfoHash[linkID][1])
                    sample.append(linkInfoHash[linkID][0] * linkInfoHash[linkID][1])
                    # for write file
                    if str(dateID) not in medianTimeHash[linkID][timeID]:
                        sample.append(float("nan"))
                    else:
                        sample.append(medianTimeHash[linkID][timeID][str(dateID)][0])
                    testVector.append(sample)
                    linkIndexVector.append(linkID)
                    timeIndexVector.append(timeID)
                    dateIndexVector.append(str(dateID))

    # # Create a zscore algorithm
    # featureList = []
    # for i in range(0, len(trainVector[0])):
    #     featureList.append([])
    #
    # for row in trainVector:
    #     for i in range(0, len(row)):
    #         featureList[i].append(row[i])
    #
    # for row in testVector:
    #     for i in range(0, len(row)):
    #         featureList[i].append(row[i])
    #
    # avgList = []
    # devList = []
    #
    # for i in range(0, len(featureList)):
    #     print(featureList[i])
    #     avgList.append(np.average(featureList[i]))
    #     devList.append(np.std(featureList[i]))
    #
    # for row in trainVector:
    #     for i in range(0, len(row)):
    #         row[i] = (row[i] - avgList[i]) / devList[i]
    #
    # for row in testVector:
    #     for i in range(0, len(row)):
    #         row[i] = (row[i] - avgList[i]) / devList[i]




    weightVector = np.array(weightVector)
    trainFile = open("C:\\Users\\XueChuanyu\\Desktop\\final\\train914_offlinefull.csv", "w")
    trainFile.writelines("linkID,timeID,dateID,median,medianS,medianM,medianMS,hisMax,hisMaxS,hisMin,hisMinS,hisAvg,hisAvgS,hisStd,hisStdS,hisRank,hisRankS,hisPercent,hisPercentS,hisLinkDiffAvg,hisLinkDiffAvgS,hisLinkDiffMed,hisLinkDiffMedS,temporyI,temporyII,temporyIII,temporyIV,\
        temporyIS,temporyIIS,temporyIIIS,temporyIVS,tempMax,tempMin,tempAvg,tempMaxS,tempMinS,tempAvgS,selfTempMedian,selfTempMedianS,selfTempMax,selfTempMaxS,\
        selfTempMin, selfTempMins, selfTempAvg, selfTempAvgS,selfTempRank,selfTempRankS,selfTempPercent,selfTempPercentS,selfTempMaxRank,selfTempMaxRankS,selfTempMaxPercent,selfTempMaxPercentS,selfTempMinRank,selfTempMinRankS,\
        selfTempMinPercent,selfTempMinPercentS,selfTempAvgRank,selfTempAvgRankS,selfTempAvgPercent,selfTempAvgPercent,selfTempDiff,selfTempDiffW,selfTempStd,selfTempStdS,selfLinkDiffAvg,selfLinkDiffMed,distributedMedian,distributedMedianS,week,weekS,length,width,len*wid,label'\n'")
    train = np.array(trainVector)
    for eachLine in train:
        for i in eachLine:
            trainFile.writelines(str(i) + ",")
        trainFile.write("\n")
    trainFile.close()

    testFile = open("C:\\Users\\XueChuanyu\\Desktop\\final\\test914_offlinefull.csv", "w")
    testFile.writelines("linkID,timeID,dateID,median,medianS,medianM,medianMS,hisMax,hisMaxS,hisMin,hisMinS,hisAvg,hisAvgS,hisStd,hisStdS,hisRank,hisRankS,hisPercent,hisPercentS,hisLinkDiffAvg,hisLinkDiffAvgS,hisLinkDiffMed,hisLinkDiffMedS,temporyI,temporyII,temporyIII,temporyIV,\
        temporyIS,temporyIIS,temporyIIIS,temporyIVS,tempMax,tempMin,tempAvg,tempMaxS,tempMinS,tempAvgS,selfTempMedian,selfTempMedianS,selfTempMax,selfTempMaxS,\
        selfTempMin, selfTempMins, selfTempAvg, selfTempAvgS,selfTempRank,selfTempRankS,selfTempPercent,selfTempPercentS,selfTempMaxRank,selfTempMaxRankS,selfTempMaxPercent,selfTempMaxPercentS,selfTempMinRank,selfTempMinRankS,\
        selfTempMinPercent,selfTempMinPercentS,selfTempAvgRank,selfTempAvgRankS,selfTempAvgPercent,selfTempAvgPercent,selfTempDiff,selfTempDiffW,selfTempStd,selfTempStdS,selfLinkDiffAvg,selfLinkDiffMed,distributedMedian,distributedMedianS,week,weekS,length,width,len*wid,label'\n'")
    test = np.array(testVector)
    for eachLine in test:
        for i in eachLine:
            testFile.writelines(str(i) + ",")
        testFile.write("\n")
    testFile.close()

    def mapeObj(preds, dtrain):
        gaps = dtrain.get_label()
        grad = np.sign(preds - gaps) / gaps
        hess = []
        for i in range(0, len(gaps)):
            temp = 1 / abs(preds[i] - gaps[i])
            hess.append(temp)
        hess = np.array(hess)
        for i in range(0, len(gaps)):
            if gaps[i] == 0:
                grad[i] = 0
                hess[i] = 0
        return grad, hess




    param = {'max_depth': 7, 'eta': 0.2, 'silent': 1, \
             'colsample_bytree': 0.5, 'min_child_weight': 8, 'subsample': 0.7, \
             'num_parallel_tree': 5}
    num_round = 5000

    print("NOW IS MODELING")
    xgbTrain = xgboost.DMatrix(train, label=lableVector)
    xgbTest = xgboost.DMatrix(test, missing=float('nan'))
    modle = xgboost.train(param, xgbTrain, num_round, obj=mapeObj)

    preds = modle.predict(xgbTest)

    return preds, linkIndexVector, timeIndexVector, dateIndexVector, missingVector


def mapeTest(link, idHash):
    preList = link[0]
    linkIndexList = link[1]
    timeIndexList = link[2]
    dateIndexList = link[3]

    sum = 0.0
    for i in range(0, len(preList)):
        linkID = linkIndexList[i]
        timeID = timeIndexList[i]
        dateID = dateIndexList[i]
        realValue = idHash[linkID][timeID][dateID]
        sum = sum + (abs(preList[i] - realValue) / realValue)
    sum = sum / len(preList)
    return sum



def writeResult(link, outFile):
    preList = link[0]
    linkIndexList = link[1]
    timeIndexList = link[2]
    dateIndexList = link[3]
    missingVector = link[4]

    file = open(outFile, "w")
    for i in range(0, len(preList)):
        date = int(dateIndexList[i]) - 121
        hour = int(int(timeIndexList[i]) / 60)
        minite = int(int(timeIndexList[i]) % 60)
        if missingVector[i] == -1:
            value = preList[i]
        else:
            value = missingVector[i]
        timeWindowStartSTR = "2017-07-" + "%02d" % date + " " + "%02d" % hour + ":" + "%02d" % minite + ":00"
        timeWindowStart = datetime.strptime(timeWindowStartSTR, "%Y-%m-%d %H:%M:%S")
        timeWindowEnd = timeWindowStart + timedelta(minutes=2)
        timeWindowEndSTR = datetime.strftime(timeWindowEnd, "%Y-%m-%d %H:%M:%S")

        write = linkIndexList[i] \
                + "#" + "2017-07-" + "%02d" % date \
                + "#" + "[" + timeWindowStartSTR + "," + timeWindowEndSTR + ")" \
                + "#" + str(value) \
                + "\n"
        file.writelines(write)
    file.close()