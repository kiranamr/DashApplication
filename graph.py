import numpy as np

import pandas as pd
from pandas import DataFrame


MainDataF = pd.read_excel('/home/dst/Documents/Data_Science/all_reports.xlsx', sheet_name='AnalysisResults',
                          index_col=None)
data_xls = MainDataF.copy()
data_xls = data_xls.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
q2 = data_xls.quantile(.25, axis=0)

q4 = data_xls.quantile(.75, axis=0)
print(MainDataF.shape)

dff = DataFrame()


class read:

    def reder():
        global MainDataF
        global data_xls
        b = data_xls.shape

        MainData = MainDataF.copy()

        print("75 % of dencity ", q4[1])

        print("25 % of dencity ", q2[1])
        ls = list(data_xls.columns.values)
        print(ls.__len__())
        print(q2[1])

        mainDataFrame = data_xls.copy()
        OutOfRange = data_xls.copy()
        BelowRange = data_xls.copy()

        for index in range(ls.__len__()):
            arr = np.array([])
            l = ls[index]

            mainDataFrame.loc[mainDataFrame[l] < q2[index], l] = 0
            mainDataFrame.loc[mainDataFrame[l] > q4[index], l] = 0
            OutOfRange.loc[OutOfRange[l] < q4[index], l] = 0
            BelowRange.loc[BelowRange[l] > q2[index], l] = 0

        print('in renge', mainDataFrame)
        print('out of range', OutOfRange)
        print('below range', BelowRange)

        InRengeArray = np.count_nonzero(mainDataFrame, axis=1)
        OutOfRangeArray = np.count_nonzero(OutOfRange, axis=1)
        BelowRangeArray = np.count_nonzero(BelowRange, axis=1)

        print('Inrenge', InRengeArray)
        print('Outof', OutOfRangeArray)
        print('Below', BelowRangeArray)

        mainDataFrame['IQR_mean'] = mainDataFrame.mean(axis=1)
        OutOfRange['IQR_mean'] = OutOfRange.mean(axis=1)
        BelowRange['IQR_mean'] = BelowRange.mean(axis=1)

        RangeDataFrame = DataFrame()
        RangeDataFrame['In_Range'] = pd.Series(InRengeArray)
        RangeDataFrame['Outof_Range'] = pd.Series(OutOfRangeArray)
        RangeDataFrame['Below_Range'] = pd.Series(BelowRangeArray)

        RangeDataFrame['Range'] = RangeDataFrame.idxmax(axis=1)

        arr1 = np.array([])
        for row in range(b[0]):
            if RangeDataFrame.iloc[row][3] == 'In_Range':
                arr1 = np.append(arr1, mainDataFrame.iloc[row][ls.__len__()])
            elif RangeDataFrame.iloc[row][3] == 'Below_Range':
                arr1 = np.append(arr1, BelowRange.iloc[row][ls.__len__()])
            else:
                arr1 = np.append(arr1, OutOfRange.iloc[row][ls.__len__()])
        print(RangeDataFrame)
        MainData['Row_Main'] = MainData.mean(axis=1)
        df = DataFrame()
        df['SerialNames'] = pd.Series(MainData['SampleNumber'])

        df['Row_Main'] = pd.Series(MainData['Row_Main'])
        df['IQR_main'] = pd.Series(arr1)
        df['Range'] = pd.Series(RangeDataFrame['Range'])

        print(df)

        return df

    def AccessRow(index):
        global MainDataF
        global dff

        data = MainDataF.copy()
        b = data.shape
        data = data.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        q2 = data.quantile(.25, axis=0)

        q4 = data.quantile(.75, axis=0)

        row = data.iloc[index]

        ar = np.array([])
        for l in range(b[1]):
            ar = np.append(ar, data.iloc[index][l])
        print(ar)

        dataFrame = pd.DataFrame()
        dataFrame['Header'] = pd.Series(list(MainDataF.columns.values))
        dataFrame['Row_Data'] = pd.Series(ar)
        bellow = np.where(np.array(row) < np.array(q2))
        outter = np.where(np.array(row) > np.array(q4))
        inRange = np.where(np.logical_and(np.array(row) > np.array(q2), np.array(row) < np.array(q4)))
        arr = np.array([])
        for clm in range(inRange.__len__()):
            dataFrame.loc[inRange[clm], 'Range'] = 'In_Range'
        for clm in range(bellow.__len__()):
            dataFrame.loc[bellow[clm], 'Range'] = 'Below_Range'
        for clm in range(outter.__len__()):
            dataFrame.loc[outter[clm], 'Range'] = 'Outof_Range'
        # dataFrame.index.name='index'
        m = np.array([])
        # print('printing row',row)
        for r in range(0, 180):
            m = np.append(m, r)
        dataFrame['Index'] = pd.Series(m)
        dff = dataFrame.copy()
        return dff

