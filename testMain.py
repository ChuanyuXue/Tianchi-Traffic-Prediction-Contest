import DataProcessing
import xgboostIICOPY
#import linkRealationAnalizing
inputFile = "C:\\Users\\XueChuanyu\\Desktop\\final\\quaterfinal_gy_cmp_training_traveltime.txt"
outputFile = "C:\\Users\\XueChuanyu\\Desktop\\final\\sortedData1.txt"

orderedFile = "C:\\Users\\XueChuanyu\\Desktop\\final\\sortedData.txt"
linkFile = "C:\\Users\\XueChuanyu\\Desktop\\final\\gy_contest_link_top(20170715).txt"
infoFile = "C:\\Users\\XueChuanyu\\Desktop\\final\\gy_contest_link_info.txt"
outFile = "C:\\Users\\XueChuanyu\\Desktop\\final\\result910.txt"

def main():
    #DataProcessing.DataProcessing(inputFile, outputFile)
    #xgboostII.check(xgboostII.loadFile(orderedFile))
    #xgboostII.median( xgboostII.loadFile(orderedFile))
    #xgboostII.dataFix(xgboostII.loadFile(orderedFile))
    lis = xgboostIICOPY.modle(360, 480, xgboostIICOPY.loadLinkInfo(infoFile),\
                          xgboostIICOPY.dataSmooth(xgboostIICOPY.dataFix(xgboostIICOPY.loadFile(orderedFile))), 480, 540)
    xgboostIICOPY.writeResult(lis, outFile)
    #print(xgboostII.mapeTest(lis))




    
    
    
if __name__ == "__main__":
    main()
