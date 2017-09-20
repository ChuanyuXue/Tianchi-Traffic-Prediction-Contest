from datetime import datetime


def DataProcessing(inputFilePath, outputFilePath):

    dataFileIn = open(inputFilePath)
    dataFileIn.readline()  # Skim the frist line
    count = 0
    idHash = {}

    #load Data
    for line in dataFileIn:
        
        if line == "":
            break

        eachLine = line.split(';')
        linkID = eachLine[0]
        date = eachLine[1]
        timeWindow = eachLine[2]
        value = eachLine[3].replace("\n",'')

        timeWindow = timeWindow.replace('[', '').split(',')
        timeWindowStart = datetime.strptime(timeWindow[0], "%Y-%m-%d %H:%M:%S")
        hour = timeWindowStart.hour
        minute = timeWindowStart.minute
        time = timeWindowStart.time()

        time = int(time.hour * 60 + time.minute)
        date = int((datetime.strptime(date, "%Y-%m-%d") - datetime.strptime("2017-03-01", "%Y-%m-%d")).days)
        if date >= 0:
            if linkID not in idHash:
                idHash[linkID] = {}
            if time not in idHash[linkID]:
                idHash[linkID][time] = {}
            idHash[linkID][time][date] = value



    dataFileIn.close()

    #Sort Hash
    for linkID in idHash.keys():

        sorted(idHash[linkID].items(), key=lambda d: d[0])

        for time in idHash[linkID].keys():

            sorted(idHash[linkID][time].items(), key=lambda d: d[0])


    #Write output file
    dataFileOut = open(outputFilePath, "w")
    for linkID in idHash.keys():
        dataFileOut.writelines("\n" + linkID)
        for time in idHash[linkID].keys():
            #strtime = time.strftime("%H:%M:%S")
            dataFileOut.writelines("," + str(time) + ":")
            for date in idHash[linkID][time].keys():
                value = idHash[linkID][time][date]
                dataFileOut.writelines(str(date) + "-" + str(value) + ";")

    dataFileOut.close()

