"""
July 12, 2017 - 30th
For CyberHealthGIS REU, research how to automatically label road conditions through cycling
Data wrangling of JSON to CSV

Note:
There are pandas warnings - ignore
When running entirely, after graph, it may crash. Restartd and 'n' to graph and then 'y' to 
extracting features and should run fine
"""
import pandas
import commands
import datetime as dt
import os
import csv
import numpy as np
import time
from dateutil.parser import parse
from time import mktime, strptime
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy import stats
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from matplotlib.offsetbox import AnchoredText

#-----------------------------------------------------------------------------
#                          Fixing up and splitting data...
#-----------------------------------------------------------------------------
#create new .csv, one for phone data and one for pebble data
def splitDevices(userFileName):    
  csv = pandas.read_csv(userFileName + ".csv", sep=',', header=0, skipinitialspace=False)
  csvPhone = csv[csv['PhoneDevice'] == 'motorola-XT1096']
  csvPebble = csv[csv['PebbleDevice'] == 'PebbleTimeSteel']
  csvPhone.to_csv(os.getcwd() + "/byDevice/" + userFileName + "Phone.csv", index=False, sep=',')
  csvPebble.to_csv(os.getcwd() + "/byDevice/" + userFileName + "Pebble.csv", index=False, sep=',')

#split data by user (activity = x)
def SplitUsers(fileName, itemList):
  for i in range(len(itemList)):
    csvPhone = pandas.read_csv(os.getcwd() + "/byDevice/" +fileName + "Phone.csv" , sep=',', header=0, skipinitialspace=False)
    csvPebble = pandas.read_csv(os.getcwd() + "/byDevice/" +fileName + "Pebble.csv" , sep=',', header=0, skipinitialspace=False)
    phone = csvPhone[csvPhone['Activity'] == itemList[i]] #1
    pebble = csvPebble[csvPebble['Activity'] == itemList[i]] #2
    if not phone.empty: 
      newFileItemPhone = "".join(itemList[i].title().split(" "))
      phone.to_csv(os.getcwd() + "/byDeviceUser/" + fileName + "Phone" + newFileItemPhone +".csv", index=False, sep=',')
    if not pebble.empty: 
      newFileItemPebble = "".join(itemList[i].title().split(" "))
      pebble.to_csv(os.getcwd() + "/byDeviceUser/" + fileName + "Pebble" + newFileItemPebble +".csv", index=False, sep=',')

#convert datetime to epoch
def convertEpoch(value):
  d = str(parse(value))
  if(len(d) == 26):
    tupleTime = dt.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f')
  else:
    tupleTime = dt.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
  msecond = tupleTime.microsecond
  cur_time = (float(mktime(tupleTime.timetuple())) + (msecond * 0.000001))*1000
  return cur_time

#update data, removing extra columns, updating
def updateSpecificData(fileName, itemList):
     for i in range(len(itemList)):
          newFileItem = "".join(itemList[i].title().split(" "))
          phone = pandas.read_csv(os.getcwd() + "/byDeviceUser/" + fileName + "Phone" + newFileItem + ".csv", parse_dates=True)
          pebble = pandas.read_csv(os.getcwd() + "/byDeviceUser/" + fileName + "Pebble" + newFileItem + ".csv", parse_dates=True)
          phoneKeep = ['Time','Activity','PhoneDevice','android_sensor_accelerometer0','android_sensor_accelerometer1','android_sensor_accelerometer2','Lat','Lon']
          pebbleKeep = ['PebbleAccT','Activity','PebbleDevice','PebbleAccX','PebbleAccY','PebbleAccZ','Lat','Lon']
          updatePhone = phone[phoneKeep].rename(columns={'Lon':'Long'}) 
          updatePebble = pebble[pebbleKeep].rename(columns={'Lon':'Long'}).rename(columns={'PebbleAccT':'Time'})
          updatePhone['Time'] = updatePhone['Time'].apply(convertEpoch) 
          updatePebble['Time'] = updatePebble['Time'].apply(convertEpoch)
          updatePhone = (updatePhone.sort_values('Time')).drop_duplicates('Time')
          updatePebble = (updatePebble.sort_values('Time')).drop_duplicates('Time')
          updatePhone.to_csv(os.getcwd() + "/byDeviceUser/" + fileName + "Phone" + newFileItem + ".csv", index=False)
          updatePebble.to_csv(os.getcwd() + "/byDeviceUser/" + fileName + "Pebble" + newFileItem + ".csv", index=False)

#rename trim files in trim folder
def renameTrim():
  for subdir, dirs, files in os.walk(os.getcwd() + "/trim/"):
    for csv_file in files[1:]:
      if csv_file.endswith("csv"):
        featureFile = csv_file.split(".csv")[0]
        oldFileName = csv_file
        newFileName = "TRIM" + featureFile + ".csv"
        print oldFileName
        print newFileName
        os.rename(os.getcwd() + "/trim/"+oldFileName,os.getcwd() + "/trim/"+newFileName)

#graph data for phone and pebble data
def graph():
  for subdir, dirs, files in os.walk(os.getcwd() + "/trim"):
    for cur_file in files:
      if cur_file.endswith(".csv"):
        cols = []
        if (cur_file.find("Phone") != -1):
          cols = ["Time", "android_sensor_accelerometer0", "android_sensor_accelerometer1", "android_sensor_accelerometer2", "Activity"]
        else:
          cols = ["Time", "PebbleAccX", "PebbleAccY", "PebbleAccZ", "Activity"]
        features = np.genfromtxt(subdir + "/" + cur_file, delimiter=',', dtype = None, names=True, usecols=cols, unpack=True)
        features_set = pandas.DataFrame(data = features, columns = cols);
        current_palette = sns.color_palette()

        lgd_x = mpatches.Patch(color="#3290BE", label='X')
        lgd_y = mpatches.Patch(color="#FFBE68", label='Y')
        lgd_z = mpatches.Patch(color="#2ecc71", label='Z')

        sns.set(style="whitegrid", color_codes=True)

        fig = plt.figure(figsize=(12, 9))
        fig.canvas.set_window_title(cur_file[:-4])
        ax = fig.add_subplot(1,1,1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Acceleration (mG)", fontsize=16)

        ax.legend(handles = [lgd_x, lgd_y, lgd_z])
        if cur_file.find("Phone") != -1:
          plt.plot(features["Time"], features["android_sensor_accelerometer0"], c = "#3290BE", lw = 2)
          plt.plot(features["Time"], features["android_sensor_accelerometer1"], c = "#FFBE68", lw = 2)
          plt.plot(features["Time"], features["android_sensor_accelerometer2"], c = "#2ecc71", lw = 2)
        else:
          plt.plot(features["Time"], features["PebbleAccX"], c = "#3290BE", lw = 2)
          plt.plot(features["Time"], features["PebbleAccY"], c = "#FFBE68", lw = 2)
          plt.plot(features["Time"], features["PebbleAccZ"], c = "#2ecc71", lw = 2)

        plt.savefig(os.getcwd() + "/graphs/GRAPH" + cur_file[:-4] + '.png', bbox_inches = 'tight')
        plt.close()

#----------------------------------------------------------------------
#                         Create/for features...
#----------------------------------------------------------------------
WINDOW_SIZE = 1 #unit = seconds
OVERLAP = 0.5
header = [ 'Variance X','Variance Y', 'Variance Z',
                 'Avg Height X' , 'Avg Height Y', 'Avg Height Z',
                 'Stdev Height X', 'Stdev Height Y', 'Stdev Height Z',
                 'Energy X', 'Energy Y', 'Energy Z',
                 'Entropy X', 'Entropy Y', 'Entropy Z',
                 'Average X', 'Average Y', 'Average Z',
                 'Average XY', 'Average XZ', 'Average YZ',
                 'Standard Deviation X', 'Standard Deviation Y', 'Standard Deviation Z',
                 'Correlation XY', 'Correlation XZ', 'Correlation YZ',
                 'RMS X', 'RMS Y', 'RMS Z',
                 'Axis Order XY', 'Axis Order XZ', 'Axis Order YZ',
                 'Num Peaks X', 'Num Peaks Y', 'Num Peaks Z',
                 'Average Peaks X', 'Average Peaks Y', 'Average Peaks Z',
                 'Standard Deviation Peaks X', 'Standard Deviation Peaks Y', 'Standard Deviation Peaks Z',
                 'Num Valleys X', 'Num Valleys Y', 'Num Valleys Z',
                 'Average Valleys X', 'Average Valleys Y', 'Average Valleys Z',
                 'Standard Deviation Valleys X', 'Standard Deviation Valleys Y', 'Standard Deviation Valleys Z',
                 'Zero Crossings X', 'Zero Crossings Y', 'Zero Crossings Z',
                 'Num Points','RoadType']

def median(sorted_x):
        sorted_x.sort()
        if len(sorted_x) % 2 == 0:
            median = (sorted_x[len(sorted_x)/2] + sorted_x[len(sorted_x)/2 - 1])/2
        else:
            median = sorted_x[len(sorted_x)/2]
        return median

def sliceData():
    '''iterates through files in csv_path, creating sliding windows and calling extract_features on each window'''
    for subdir, dirs, files in os.walk(os.getcwd() + "/trim"):
        for csv_file in files:
            if csv_file.endswith("csv"):
                featureFile = csv_file.split(".csv")[0]
                print "Extracting features from " + csv_file
                f = csv.reader(open(subdir + "/" + csv_file), delimiter=",")
                raw_data_header = f.next() #Raw Data file header
                # Initialize windows of data
                time = []
                x = []
                y = []
                z = []
                cur_window_time = 0
                cur_overlap_time = 0
                initial_second = True

                for row in f:
                    cur_time = row[0]
                    cur_x = row[3]
                    cur_y = row[4]
                    cur_z = row[5]
                    time.append(float(cur_time)/1000) # Converts raw data time from ms to s
                    x.append(float(cur_x))
                    y.append(float(cur_y))
                    z.append(float(cur_z))
                    cur_window_time = time[len(time)-1] - time[0]

                    # Since lists are initially empty, handle initial second separately
                    if initial_second:
                        if cur_window_time <= OVERLAP:
                            cur_overlap_time = time[len(time)-1]

                        if cur_window_time >= WINDOW_SIZE:
                            initial_second = False
                            extractFeatures(featureFile,time[:-1], x[:-1], y[:-1], z[:-1])

                    else:
                        if time[0] - cur_overlap_time >= OVERLAP:
                            if cur_window_time >= WINDOW_SIZE:
                                extractFeatures(featureFile,time[:-1], x[:-1], y[:-1], z[:-1])
                                cur_overlap_time = time[0] # Overlap time for next window
                                time.pop(0)
                                x.pop(0)
                                y.pop(0)
                                z.pop(0)

                        # Remove first index until Overlap time is reached
                        else:
                            time.pop(0)
                            x.pop(0)
                            y.pop(0)
                            z.pop(0)

def extractFeatures(featureFile, time, x, y, z):
    ''' If window size is within 10% of expected number, extracts features and writes to output'''
    if (featureFile.find("Pebble") != -1):
        SAMPLING_RATE = 25  
        if not (len(time) >= (0.9 * WINDOW_SIZE * SAMPLING_RATE) and len(time) <= (1.1 * WINDOW_SIZE * SAMPLING_RATE)):
            return

    heights_x = side_height(x, time)
    heights_y = side_height(y, time)
    heights_z = side_height(z, time)

    x_peaks = peaks(x)
    y_peaks = peaks(y)
    z_peaks = peaks(z)

    x_valleys = valleys(x)
    y_valleys = valleys(y)
    z_valleys = valleys(z)

    stdev_peaks_x = 0.0
    stdev_peaks_y = 0.0
    stdev_peaks_z = 0.0

    sorted_x = list(x)
    sorted_y = list(y)
    sorted_z = list(z)

    if not x_peaks:
          avg_peaks_x = median(sorted_x)
    else:
        avg_peaks_x = average(x_peaks)
        stdev_peaks_x = stdev(x_peaks)

    if not y_peaks:
        avg_peaks_y = median(sorted_y)
    else:
        avg_peaks_y = average(y_peaks)
        stdev_peaks_y = stdev(y_peaks)

    if not z_peaks:
        avg_peaks_z = median(sorted_z)
    else:
        avg_peaks_z = average(z_peaks)
        stdev_peaks_z = stdev(z_peaks)

    stdev_valleys_x = 0.0
    stdev_valleys_y = 0.0
    stdev_valleys_z = 0.0

    if not x_valleys:
        avg_valleys_x = median(x)
    else:
        avg_valleys_x = average(x_valleys)
        stdev_valleys_x = stdev(x_valleys)

    if not y_valleys:
        avg_valleys_y = median(y)
    else:
        avg_valleys_y = average(y_valleys)
        stdev_valleys_y = stdev(y_valleys)

    if not z_valleys:
        avg_valleys_z = median(z);
    else:
        avg_valleys_z = average(z_valleys)
        stdev_valleys_z = stdev(z_valleys)

    avg_height_x = 0.0
    avg_height_y = 0.0
    avg_height_z = 0.0
    stdev_heights_x = 0.0
    stdev_heights_y = 0.0
    stdev_heights_z = 0.0

    if heights_x:
        stdev_heights_x = stdev(heights_x)
        avg_height_x = average(heights_x)

    if heights_y:
        stdev_heights_y = stdev(heights_y)
        avg_height_y = average(heights_y)

    if heights_z:
        stdev_heights_z = stdev(heights_z)
        avg_height_z = average(heights_z)

    cur_features =[ var(x), var(y), var(z), avg_height_x, avg_height_y, avg_height_z,stdev_heights_x, stdev_heights_y, stdev_heights_z,
        energy(x), energy(y), energy(z),entropy(x), entropy(y), entropy(z),average(x), average(y), average(z),avg_diff(x, y), avg_diff(x, z), avg_diff(y, z),
        stdev(x), stdev(y), stdev(z),sig_corr(x,y), sig_corr(x,z), sig_corr(y,z),rms(x), rms(y), rms(z),axis_order(x,y), axis_order(x,z), axis_order(y,z),
        len(x_peaks), len(y_peaks), len(z_peaks),avg_peaks_x, avg_peaks_y, avg_peaks_z,stdev_peaks_x, stdev_peaks_y, stdev_peaks_z,
        len(x_valleys), len(y_valleys), len(z_valleys),avg_valleys_x, avg_valleys_y, avg_valleys_z,stdev_valleys_x, stdev_valleys_y, stdev_valleys_z,
        z_crossings(x), z_crossings(y), z_crossings(z),len(time), label(featureFile)]

    w = open(os.getcwd() + "/features/" + featureFile + "Features.csv", 'a+')
    writer = csv.writer(w)
    if os.stat(os.getcwd() + "/features/" + featureFile + "Features.csv").st_size == 0:
        writer.writerow(header)
    writer.writerow(cur_features)
    w.close()

#functions for calculating various features
def var(x): return np.var(x)

def label(featureFile): 
    if (featureFile.find("Phone") != -1):
        label = featureFile[:featureFile.find("Phone")]
    else:
        label = featureFile[:featureFile.find("Pebble")]
    label = label[4:]
    return label

#Find Average Jerk
def avg_jerk(x, time):
        jerk = 0.0
        for index in range(1,len(x)):
            jerk += (x[index] - x[index-1])/(time[index]-time[index-1])
        if len(x) == 1:
            return jerk
        else:
            return round(jerk/(len(x)-1))

#Find Average Distance Between Each Value
def avg_diff(x,  y):
        diff = 0.0
        for index in range(1,len(x)):
            diff += x[index] - y[index]
        return diff/len(x)

#Finds number of times axis order changes
def axis_order (x, y):
    changes = 0
    xgreatery = None
    for cnt in range(len(x)):
        if (cnt == 0):
            if x[cnt] > y[cnt]:
                xgreatery = True
            elif x[cnt] < y[cnt]:
                xgreatery = None
        else:
            if x[cnt] > y[cnt]:
                if not xgreatery:
                    changes+=1
            elif x[cnt] < y[cnt]:
                if xgreatery:
                    changes+=1
    return changes

#Find Energy
def energy(x):
        energy = 0
        for k in range(len(x)):
            ak = 0.0
            bk = 0.0
            for i in range(len(x)):
                angle = 2*math.pi*i*k/len(x)
                ak += x[i]*math.cos(angle)
                bk += -x[i]*math.sin(angle)
            energy += (math.pow(ak, 2)+math.pow(bk, 2))/len(x)
        return energy

#Find Entropy
def entropy(x):
    spectralentropy = 0.0
    for j in range(len(x)):
        ak = 0.0
        bk = 0.0
        aj = 0.0
        bj = 0.0
        mag_j = 0.0
        mag_k = 0.0
        cj = 0.0
        for i in range(len(x)):
            angle = 2*math.pi*i*j/len(x)
            ak = x[i]*math.cos(angle) #Real
            bk = -x[i]*math.sin(angle) #Imaginary
            aj+=ak
            bj+=bk
            mag_k += math.sqrt(math.pow(ak, 2)+math.pow(bk, 2))
        mag_j = math.sqrt(math.pow(aj, 2)+math.pow(bj, 2))
        if mag_k != 0:# and cj != 0:
            cj = mag_j/mag_k
            if (cj != 0):
              spectralentropy += cj*math.log(cj)/math.log(2)
              return -spectralentropy
            else:
              return 0
        else:
            return 0

#calculates side_height
def side_height(x, time):
        heights = []
        q1_check = None #true greater than, false less than
        q3_check = None
        moved_to_middle = None
        cur_q1_points = []
        cur_q3_points = []
        peaks_valleys = []
        sorted_x = list(x)
        cur_median = median(sorted_x)
        q1 = min(x) + abs((cur_median - min(x))/2)
        q3 = cur_median + abs((max(x) - cur_median)/2)
        cur_x = 0.0
        for i in range(len(x)):
            cur_x = x[i]
            if i == 0:
                if cur_x > q3:
                    cur_q3_points.append(cur_x)
                    q1_check = True
                    q3_check = True
                elif cur_x > q1:
                    q1_check = True
                else:
                    cur_q1_points.append(cur_x)
            else:
                if cur_x > q3:
                    q3_check = True
                    q1_check = True
                    if moved_to_middle:
                        if cur_q1_points:
                            peaks_valleys.append(min(cur_q1_points)) #add valley
                        del cur_q1_points[:]
                        moved_to_middle = None
                    cur_q3_points.append(cur_x)
                elif cur_x > q1:
                    if (q3_check and q1_check) or (not q3_check and not q1_check):
                        moved_to_middle = True
                    q1_check = True
                    q3_check = None
                else:
                    if moved_to_middle:
                        if cur_q3_points:
                            peaks_valleys.append(max(cur_q3_points)) #add peak
                        del cur_q3_points[:]
                        moved_to_middle = None
                    cur_q1_points.append(cur_x)
                    q1_check = None
                    q3_check = None
        for i in range(len(peaks_valleys)-1):
            heights.append(abs(peaks_valleys[i+1] - peaks_valleys[i]))
        return heights

#calculates the distance from the peak/valley to the mean
def dist_to_mean (x):
        avg = 0.0
        increasing = None
        decreasing = None
        dist = []
        avg = average(x)
        for i in range(len(x)):
            if x[i] > x[i-1]:
                increasing = True
                if decreasing:
                    dist.append(avg-x[i-1])
                    decreasing = None
            elif x[i] < x[i-1]:
                decreasing = None
                if increasing:
                    dist.append(x[i-1]-avg)
                    increasing = None
        return dist

#calculates average
def average(x):
        avg = 0.0
        for cnt in x:
            avg += cnt
        return avg/len(x)

#Find Standard Deviation
def stdev(x):
        avg = average(x)
        std = 0.0
        for cur_x in x:
            std +=math.pow((cur_x - avg),2)
        return math.sqrt(std/len(x))

#Find Signal Correlation
def sig_corr(x, y):
        correlation = 0.0
        for cnt in range(len(x)):
            correlation += x[cnt] * y[cnt]
        return correlation/len(x)

#Find Root Mean Square
def rms(x):
    avg = 0.0
    for cnt in x:
        avg += math.pow(cnt,2)
    return math.sqrt(avg/len(x))

def peaks (x):
        peaks = []
        q1_check = None #true greater than, false less than
        q3_check = None
        moved_to_middle = None
        cur_q3_points = []
        sorted_x = list(x)
        cur_median = median(sorted_x)
        q1 = min(x) + abs((cur_median - min(x))/2)
        q3 = cur_median + abs((max(x) - cur_median)/2)
        cur_x = 0.0
        for i,cur_x in enumerate(x):
            if i == 0:
                if cur_x > q3:
                    cur_q3_points.append(cur_x)
                    q1_check = True
                    q3_check = True
                elif cur_x > q1:
                    q1_check = True
            else:
                if cur_x > q3:
                    q3_check = True
                    q1_check = True
                    if moved_to_middle:
                        moved_to_middle = None
                    cur_q3_points.append(cur_x)
                elif cur_x > q1:
                    if (q3_check and q1_check) or (not q3_check and not q1_check):
                        moved_to_middle = True
                    q1_check = True
                    q3_check = None
                else:
                    if moved_to_middle:
                        if cur_q3_points:
                            peaks.append(max(cur_q3_points)) #add peak
                        del cur_q3_points[:]
                        moved_to_middle = None
                    q1_check = None
                    q3_check = None
        return peaks

def valleys (x):
        valleys = []
        q1_check = None #true greater than, false less than
        q3_check = None
        moved_to_middle = None
        cur_q1_points = []
        sorted_x = list(x)
        cur_median = median(sorted_x)
        q1 = min(x) + abs((cur_median - min(x))/2)
        q3 = cur_median + abs((max(x) - cur_median)/2)
        cur_x = 0.0
        for i, cur_x in enumerate(x):
            if i == 0:
                if cur_x > q3:
                    q1_check = True
                    q3_check = True
                elif cur_x > q1:
                    q1_check = True
                else:
                    cur_q1_points.append(cur_x)
            else:
                if cur_x > q3:
                    q3_check = True
                    q1_check = True
                    if moved_to_middle:
                        if cur_q1_points:
                            valleys.append(min(cur_q1_points)) #add valley
                        del cur_q1_points[:]
                        moved_to_middle = None
                elif cur_x > q1:
                    if (q3_check and q1_check) or (not q3_check and not q1_check):
                        moved_to_middle = True
                    q1_check = True
                    q3_check = None
                else:
                    if moved_to_middle:
                        moved_to_middle = None
                    cur_q1_points.append(cur_x)
                    q1_check = None
                    q3_check = None

        return valleys

#Find Zero Crossings
def z_crossings(x):
        cur_sign = 0
        prev_sign = 0
        sign = 0
        cnt = 0
        crossings = 0

        while prev_sign == 0 and cnt < len(x)-1:
            prev_sign = math.copysign(1,x[cnt])
            cnt+=1
        if prev_sign == 0:
            return crossings
        while cnt < len(x):
            cur_sign = math.copysign(1,x[cnt])
            while cur_sign == 0 and cnt < len(x)-1:
                cnt+=1
                cur_sign = math.copysign(1,x[cnt])
            if cur_sign == 0: #the last value was zero, so no more crossings will occur
                break
            sign = cur_sign - prev_sign
            if sign == 2: #1-(-1)
                crossings+=1
                break
            elif sign == 0: #1-(+1), -1-(-1)
                break
            elif sign == -2: #-1-(+1)
                crossings+=1
                break
            prev_sign = cur_sign
            cnt+=1
        return crossings

def main():
    userFile = raw_input("Filename (excluding .json extension): ")
    convert = raw_input("Convert to .csv? (y/n) ")

    if (convert == "y" or convert == "yes"):
        print "Converting json to csv..."
        commands.getstatusoutput('json2csv -i ' + userFile +
                                '.json -f Time,PebbleAccT,Activity,PhoneDevice,PebbleDevice,PebbleAccX,PebbleAccY,PebbleAccZ,android_sensor_accelerometer0,android_sensor_accelerometer1,android_sensor_accelerometer2,Lat,Lon  -o ' +
                                userFile + '.csv') #linux command to convert JSON to CSV
    split = raw_input("Split? (y/n) ") 
    if (split == "y" or split == "yes"):
      numItems = int(raw_input("Num of items: "))
      itemList = []
      for i in range(numItems):
        name = raw_input("Item name " + str(i+1) + " (case-sens) : ")
        itemList.append(name)
      splitByDevice = raw_input("Split by device? (y/n) ")
      splitByDeviceUser = raw_input("Split by device & user? (y/n) ")
      print "splitting..."
      if (splitByDevice == "y" or splitByDevice == "yes"):
        splitDevices(userFile)
      if (splitByDeviceUser == "y" or splitByDeviceUser == "yes"):
        SplitUsers(userFile, itemList)
        updateSpecificData(userFile, itemList)
    rename = raw_input("Rename trim files in trim folder? (y/n): ")
    if (rename == 'y' or rename == "yes"):
      renameTrim()
    createPlots = raw_input("Plot trimmed data? (y/n) ")
    if (createPlots == 'y' or createPlots == "yes"):
      graph()
    extractFeat = raw_input("Calcluate features? (y/n) ")
    if (extractFeat == "y" or extractFeat == "yes"):
      sliceData()
    print "DONE :)"
     
main()

"""
NOTES:
Phone:
Activity, Time, Phone Device, PhoneOS, Lat, Lon, GPS_Accuracy, GPS_Timestamp, GPS_Bearing,
GPS_Speed, GPS_Altitude, GPS_Provider, android_sensor_accelerometerA,android_sensor_accelerometerT,
android_sensor_accelerometer0, android_sensor_accelerometer1, android_sensor_accelerometer2

Pebble:
PebbleAccT, PebbleAccX, PebbleAccY, PebbleAccZ, PebbleOS, PebbleDevice, Heading, Activity,
Time, Lat, Lon, GPS_Accuracy

"""
