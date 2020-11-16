# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:46:22 2017

@author: qicdz
"""
import pandas as pd
import matplotlib.pyplot as plt
import time
import calendar
import os
import xlwt

'''
统计每个停车场当月的停车数据
'''
count_parking_num = []

'''
得到一个二维数组，分别存储2016年mon月份每一辆车的进、出时间戳，并去掉停车超过一天的数据
'''
def get_time_list(name, mon):
    df = pd.read_excel('data5\\'+name)
    in_time =df.ix[8:, 1]
    out_time = df.ix[8:, 2]
    parking_time = []
    for i in range(8, len(in_time)+8):
        parking_time.append([in_time[i],out_time[i]])
    #由于数据集间的差别，这里需要多次使用try except来读取数据，防止出现ValueError    
    stamp1 = "%Y-%m-%d %H:%M:%S"
    stamp2 = "%Y/%m/%d %H:%M:%S"#两种时间格式
    time_list = []
    for i in parking_time:
        try:
            t1 = time.strptime(str(i[0])[:-7], stamp1)#在正常数据情况下是无法取得的
        except ValueError:
            try:
                t1 = time.strptime(str(i[0]), stamp1)
            except ValueError:
                t1 = time.strptime(str(i[0]), stamp2)                
        if t1[0] == 2016 and t1[1] ==mon :#确认时间值符合再进行求解
            in_localtime = float(time.mktime(t1))#包括所有数据
            try:
                t2 = time.strptime(str(i[1])[:-7], stamp1)
            except ValueError:
                try:
                    t2 = time.strptime(str(i[1]), stamp1)
                except ValueError:
                    t2 = time.strptime(str(i[1]), stamp2)                       
            out_localtime = float(time.mktime(t2))#包括所有数据
            time_list.append([in_localtime, out_localtime])
    print(name, len(time_list))
    count_parking_num.append([name, len(time_list)])
    return time_list

'''
将出、入场数据分别提出生成两个二维数组，入场的第二维数据为1，出场的第二维数据为-1，
最后将两个二维数组合并，按时间升序排列
'''
def get_car(name, mon):
    car_in = []
    car_out = []
    time_list = get_time_list(name, mon)
    if time_list == None:
        return None
    for i in time_list:
        car_in.append([i[0], 1])
        car_out.append([i[1], -1])
    car = []
    for i in range(len(car_in)):
        car.append(car_in[i])
        car.append(car_out[i]) 
    car.sort()#排序 升序操作
    return car

'''
将日期转换成时间戳
'''
def to_time_stamp(realtime):
    stamp = "%Y-%m-%d %H:%M:%S"
    time_stamp = time.mktime(time.strptime(realtime, stamp))
    return time_stamp

'''
每15分（900秒）做一个时间节点，统计当前时间节点下的停车数量
'''
def get_car_list(name, mon):#返回值为计数值 还要转换为空闲率
    car_num = 0
    car_list = []
    car = get_car(name, mon)
    if car == None:
        return None
    if 0 < mon and mon < 10:#这里分开的目的只是为了展示的时候输出效果对称 比如07 12 这种输出效果
        start = '2016-0'+str(mon)+'-01 00:00:00'
        month_range = calendar.monthrange(2016, mon)
        end = '2016-0'+str(mon)+'-'+str(month_range[1])+' 23:59:59'
    elif mon < 13:
        start = '2016-'+str(mon)+'-01 00:00:00'
        month_range = calendar.monthrange(2016, mon)
        end = '2016-'+str(mon)+'-'+str(month_range[1])+' 23:59:59'
    time1 = int(to_time_stamp(start))
    time2 = int(to_time_stamp(end))
    for i in range(time1, time2, 900):
        for j in car:
            if j[0] <= i:
                car_num += j[1]
                car.remove(j)
        car_list.append(car_num)
    return car_list

'''
根据停车数据进行绘图，横坐标为每15分钟一个的时间节点，纵坐标为停车场的空车率
'''
def get_figure(name, mon, style):
    car_list = get_car_list(name, mon)
    if car_list == None:
        print(name+'有误，程序结束')
        return 
    unoccupied = []
    for i in car_list:#计算空车率程序
        try:
            percend = (max(car_list)-i) / max(car_list)
            unoccupied.append(percend)
        except ZeroDivisionError:#数据丢失的时候均为1
            unoccupied.append(1)  
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('parking')
    for i in range(len(unoccupied)):
        sheet.write(i,0,unoccupied[i])
    wbk.save(name[:-5]+'.xls')
    #后面为绘图程序
    plt.plot(unoccupied, style)
    plt.xticks([])
    plt.yticks([])
    fig = plt.gcf()
    fig.set_size_inches(12,5)
    fig.savefig(str(mon)+'月'+name[:-5]+'.png', dpi=100)
    plt.show()
    
def main(mon, style):
    file_list = os.listdir('data5')
    for i in range(len(file_list)):
        get_figure(file_list[i], mon, style)  

if __name__=='__main__':
    main(8, 'k')