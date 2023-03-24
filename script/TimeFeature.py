from datetime import datetime, timedelta
from chinese_calendar import is_workday, is_holiday
import numpy as np


def build_real_time(data):
    time_feat = []
    for i in range(data.shape[0]):
        _t = []
        for j in range(data.shape[1]):
            time = np.datetime_as_string(data[i][j], unit="m")
            tmp = time.strip().split(":")
            hour, minute = int(tmp[0][-2:]), int(tmp[1])
            _tmp = tmp[0].split("-")
            year, month, day = int(_tmp[0]), int(_tmp[1]), int(_tmp[2][:2])
            _t.append([hour, minute, month, day])
        time_feat.append(_t)

    time_feat = np.array(time_feat)
    
    return time_feat


def time_id(time, time2id, period, dataset):
    # print(time)
    time = np.datetime_as_string(time, unit="m")
    tmp = time.strip().split(":")
    hour, minute = int(tmp[0][-2:]), int(tmp[1])
    # tmp = time.strip().split()
    # hour, minute = int(tmp[1].split(":")[0]), int(tmp[1].split(":")[1])

    _tmp = tmp[0].split("-")
    year, month, day = int(_tmp[0]), int(_tmp[1]), int(_tmp[2][:2])
    
    time = datetime(year, month, day)
    if is_workday(time):
        return time2id["0:"+str(hour)+":"+str(minute)]
    else:
        if period:
            return time2id["1:"+str(hour)+":"+str(minute)]
        else:
            if dataset in ["taxi", "bike"]:
                return time2id["1:"+str(hour)+":"+str(minute)] - 48
            else:
                return time2id["1:"+str(hour)+":"+str(minute)] - 73

def build_time_embedding(data, dataset, time2id, period):
    """
    Args:
        data: numpy, (1188, 4) e.g., 2019-01-01T05:30:00.000000000'
        dataset: "HZMetro", "SHMetro"
    return:
        _data: the data transformer by embedding
        one file is saved for time2id, id2time.
    """
    # 5:30-11:30 interval 15 mins, 5:30-6:15 (4), 5:30-23:15 (18*4=72), 5:30-11:30 (73)

    time_feat = np.zeros((data.shape))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            time_feat[i][j] = time_id(data[i][j], time2id, period, dataset)
    return time_feat

if __name__ == "__main__":
    build_time_embedding()