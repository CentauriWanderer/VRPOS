#!/usr/bin/env python
# coding: utf-8

# 加入直线随机分布，密集区域降低行车速度，网格半透明，修改为手动设置放大区域，箭头提到最上层，计算行车和行走总距离，dt颜色一致，加入legend，加入距离统计直方图，加入初始平均距离计算，计算DBSCAN重划分的数量，加入k-nng求ε，密集区直接单一st改为黄色，外部读入meta，计时，加入对平均距离和时间距离结果的记录，平均距离改为1-nng

# In[168]:


get_ipython().run_line_magic('reset', '-sf')


# In[169]:


"""
  _____       _   _                 
 |  __ \     | | | |                
 | |__) |   _| |_| |__   ___  _ __  
 |  ___/ | | | __| '_ \ / _ \| '_ \ 
 | |   | |_| | |_| | | | (_) | | | |
 |_|    \__, |\__|_| |_|\___/|_| |_|
         __/ |                      
        |___/                                                           
"""

################# 全局参数 ##################

"""
TMD 行车旅程最大时间（秒）
TMW 停车最大时间（秒）
CM 旅程最大容量（个）
VW 行走速度（m/s）
VDH 非城区行车速度（m/s）
VDL 城区行车速度（m/s）
"""

TMD = 14400
TMW = 3600
CM = 1000
VW = 1.25
VDH = 10
VDL = 7

"""
GRID_FLAG 网格模式开关
GRID_DIST 网眼大小
MIN_DIST  非网格模式的最小点距
"""

GRID_FLAG = True # 是否为网格模式
GRID_DIST = 20 # 网眼正方形的边长
MIN_DIST = 20 # 非网格模式下有最小点间距

"""
DBSCAN的参数
"""
MINPTS = 2
EPSILON = 150

"""
划定矩形研究区域
"""

# 南北
bound_n = 5000
bound_s = 0

# 左右
bound_l = 0
bound_r = 5000


# In[170]:


"""
  _____       _   _                 
 |  __ \     | | | |                
 | |__) |   _| |_| |__   ___  _ __  
 |  ___/ | | | __| '_ \ / _ \| '_ \ 
 | |   | |_| | |_| | | | (_) | | | |
 |_|    \__, |\__|_| |_|\___/|_| |_|
         __/ |                      
        |___/                                                           
"""
pass
################# 点生成 ##################


# In[171]:


"""
定义仓库点d
"""

loc_d = ((bound_n + bound_s) / 2, (bound_l + bound_r) / 2) # 仓库位置在中心

# 仓库集合，暂不考虑多个仓库的情况，否则牵涉另一类问题
set_d = set([0]) # 注意set()初始化函数只接受可枚举对象

# 点index的计数器，每次向POINTS内添加点+1
COUNT = 0

"""
初始点集，只有仓库
点信息字典POINTS
格式 {id：{类型：str，经纬：(float, float)，所属聚类：str}}
"""

POINTS = {0: {"type": "d", "coord": loc_d, "cluster": -1}} 


# In[172]:


"""
用k-nng计算最小pairwise距离
"""

from sklearn.neighbors import NearestNeighbors

def meandist(listp):

    neigh = NearestNeighbors(n_neighbors = 1)
    neigh.fit(listp)
    array_dist, array_idx = neigh.kneighbors()

    return float(sum(array_dist)/len(array_dist))


# In[173]:


"""
一维随机分布函数
"""

def randomline(start, end, num, POINTS):
    
    list_knownp = []
    
    for k in POINTS.keys():
        
        list_knownp.append(POINTS[k]["coord"])
    
    delta_x = end[0] - start[0]
    grad = (end[1] - start[1])/(end[0] - start[0])
    
    import random
    
    count = 0
    list_points = []
    
    if GRID_FLAG:

        while count < num:
            a = random.uniform(0,1)
            x = start[0] + a * delta_x
            y = start[1] + grad * a * delta_x
            x = round(x / GRID_DIST) * GRID_DIST # 用四舍五入自动归并
            y = round(y / GRID_DIST) * GRID_DIST
            if (x, y) in list_points or (x, y) in list_knownp: # 点不能重叠
                continue # 重新开始循环
            count += 1
            list_points.append((x, y))      
            
    else:
        
        while count < num:
            a = random.uniform(0,1)
            x = start[0] + a * delta_x
            y = start[1] + grad * a * delta_x
            flag = False
            for (x0, y0) in list_points: # 逐个验证是否满足最小距离
                if ((x - x0)**2 + (y - y0)**2)**0.5 < MIN_DIST:
                    flag = True
                    break
            if flag:
                continue
            for (x0, y0) in list_knownp: # 逐个验证是否满足最小距离
                if ((x - x0)**2 + (y - y0)**2)**0.5 < MIN_DIST:
                    flag = True
                    break
            if flag:
                continue
            count += 1
            list_points.append((x, y))
            
    print(meandist(list_points))
            
    return list_points, meandist(list_points)


# In[174]:


"""
二维高斯分布函数
"""

def gauss2d(center, radius, num, radius_min, POINTS): # radius_min可以生成环形分布
    
    list_knownp = []
    
    for k in POINTS.keys():
        
        list_knownp.append(POINTS[k]["coord"])

    import random
    
    sigma = radius / 3 # 概率百分比
    list_points = [] # 生成的点的列表
    count = 0 # 生成的点的计数
    
    if GRID_FLAG: # 有网格，点自动移动到最近的网格，点不能重叠

        while count < num:
            x = random.gauss(0, sigma)
            y = random.gauss(0, sigma)
            dist = (x**2 + y**2)**0.5
            if dist <= radius and dist > radius_min: # 如果点在半径之内
                x += center[0]
                y += center[1]
                x = round(x / GRID_DIST) * GRID_DIST # 用四舍五入自动归并
                y = round(y / GRID_DIST) * GRID_DIST
                if (x, y) in list_points or (x, y) in list_knownp: # 点不能重叠
                    continue # 重新开始循环
                count += 1
                list_points.append((x, y))
    
    else: # 无网格，但是点之间有最小距离MINDIST

        while count < num:
            x = random.gauss(0, sigma)
            y = random.gauss(0, sigma)
            dist = (x**2 + y**2)**0.5
            if dist <= radius: # 如果点在半径之内
                x += center[0]
                y += center[1]
                # 最小点间距离
                #if len(list_points) != 0: # 如果已有点
                flag = False
                for (x0, y0) in list_points: # 逐个验证是否满足最小距离
                    if ((x - x0)**2 + (y - y0)**2)**0.5 < MIN_DIST:
                        flag = True
                        break
                if flag:
                    continue
                for (x0, y0) in list_knownp: # 逐个验证是否满足最小距离
                    if ((x - x0)**2 + (y - y0)**2)**0.5 < MIN_DIST:
                        flag = True
                        break
                if flag:
                    continue
                count += 1
                list_points.append((x, y))
                
    print(meandist(list_points))
    
    return list_points, meandist(list_points)


# In[175]:


"""
建立聚类内平均距离列表
"""

list_md = []


# In[176]:


"""
生成背景客户点并计算平均距离
"""

nbc = 20

x_cen = (bound_n + bound_s) / 2
y_cen = (bound_r + bound_l) / 2

import random

# 随机点生成
bcpoints, md = gauss2d((x_cen, y_cen), 2500, nbc, 0, POINTS)
#bcpoints = []

# 背景点不记录聚类内平均距离

set_bc = set()

for p in bcpoints:
    COUNT += 1
    POINTS[COUNT] = {"type": "bc", "coord": p, "cluster": -1} # POINTS第一次更新
    set_bc.add(COUNT)
  
"""
type字段标记点的来源，d为仓库起始点，bc为背景客户，clc0为0号群落的客户。

cluster字段标记点实际位于的聚类，-1不属于任何聚类，聚类id为自然数。

类型为bc的背景客户或重新根据地理位置划分到聚类，此时type为bc，cluster字段标记为自然数；
也有可能无法划分到任一聚类，type为bc，cluster仍为-1。
类型为clcN的群落集中客户其聚类id要看DBSCAN的结果，lcN同理
d点不属于任何群落，聚类id为-1

"""
pass


# In[177]:


# 开始生成聚类点

set_clc = set() # 聚类客户点集


# """
# 直接指定聚类meta
# 注意更新index！++
# """
# 
# dict_clinfo = {0: {'center': (2000, 3000),
#                    'radius': 250,
#                    'scale': 20
#                    },
#                1: {'center': (3000, 2600),
#                    'radius': 250,
#                    'scale': 20
#                    },
#                2: {'center': (2500, 2750),
#                    'radius': 500,
#                    'scale': 30
#                    },
#                }
# 
# 
# import pickle
# 
# with open("clusters.txt", "wb") as f:
#     
#     pickle.dump(dict_clinfo, f)

# In[178]:


import pickle

with open("clusters.txt", "rb") as f:
    
    dict_clinfo = pickle.load(f)


# In[179]:


"""
加入clusters并计算平均距离
"""

for i in dict_clinfo: # keys
    
    # 随机点生成
    center = dict_clinfo[i]["center"]
    radius = dict_clinfo[i]["radius"]
    scale  = dict_clinfo[i]["scale"]
    clpoints, md = gauss2d(center, radius, scale, 0, POINTS)
    
    # 记录聚类内平均距离
    list_md.append((md, len(clpoints)))
    
    for p in clpoints:
        
        COUNT += 1
        x = p[0]
        y = p[1]
        loc_clc = (x, y)
        typ = "clc" + str(i)
        POINTS[COUNT] = {"type": typ, "coord": loc_clc, "cluster": 0} 
        set_clc.add(COUNT)


# """
# 直接指定线分布meta
# """
# 
# dict_lineinfo = {0: {'start': (2000, 2500),
#                      'end': (2500, 3000),
#                      'scale': 15
#                      },
#                  1: {'start': (2000, 3000),
#                      'end': (3000, 3000),
#                      'scale': 15
#                      },
#                  2: {'start': (2750, 2000),
#                      'end': (2751, 3000),
#                      'scale': 15
#                      },
#                  3: {'start': (2500, 3000),
#                      'end': (3000, 2500),
#                      'scale': 15
#                      },
#                  4: {'start': (2500, 2750),
#                      'end': (3250, 2750),
#                      'scale': 15
#                      }, 
#                 }
# 
# import pickle
# 
# with open("lines.txt", "wb") as f:
#     
#     pickle.dump(dict_lineinfo, f)

# In[180]:


import pickle

with open("lines.txt", "rb") as f:
    
    dict_lineinfo = pickle.load(f)


# In[181]:


"""
加入直线分布并计算平均距离
"""

for i in dict_lineinfo: # keys

    # 随机点生成
    start = dict_lineinfo[i]["start"]
    end = dict_lineinfo[i]["end"]
    scale  = dict_lineinfo[i]["scale"]
    linepoints, md = randomline(start, end, scale, POINTS)
    
    # 记录聚类内平均距离
    list_md.append((md, len(linepoints)))
    
    for p in linepoints:
        
        COUNT += 1
        x = p[0]
        y = p[1]
        loc_lc = (x, y)
        typ = "lc" + str(i)
        POINTS[COUNT] = {"type": typ, "coord": loc_lc, "cluster": 0} 
        set_clc.add(COUNT)


# In[182]:


"""
记录聚类内平均距离
"""

mdcount = 0
numerator = 0

for e in list_md:
    mdcount += e[1]
    numerator += e[0] * e[1]

temp = numerator / mdcount

print(temp)

import pickle

with open("md.txt", "rb") as f:
    
    dict_md = pickle.load(f)
    
count = len(dict_md) # 存储数据中已有长度

dict_md[count] = temp # 自动加一

with open("md.txt", "wb") as f:
    
    pickle.dump(dict_md, f)


# In[183]:


"""
集合合并
"""

set_ac = set_bc.union(set_clc) # 全部客户点集bc+clc

set_ap = set_d.union(set_ac) # 全部点集d+ac


# In[184]:


"""
保存本次生成的点
"""

import pickle

with open("points.txt", "wb") as f: # 带自动创建功能
    
    pickle.dump(POINTS, f)


# """
# 从外部读取点
# 前面的集合和POINTS被覆盖了
# """
# 
# import pickle
# 
# with open("points.txt", "rb") as f:
#     
#     POINTS = pickle.load(f)
#     
# set_d = set()
# set_bc = set()
# set_clc = set()
# 
# for p in POINTS:
#     
#     if POINTS[p]["type"] == "d":
#         
#         set_d.add(p)
#         
#     elif POINTS[p]["type"] == "bc":
#         
#         set_bc.add(p)
#         
#     elif POINTS[p]["type"][0:3] == "clc":
#         
#         set_clc.add(p)

# In[185]:


"""
点生成结果画图
"""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

for p in set_clc:

    ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "green", marker  = '.', s = 30)  
    
for p in set_bc:
    
    ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "red", marker  = '.', s = 30)

for p in set_d:
    
    ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "black", marker  = '^', s = 50)

ax.set_xlim(bound_l, bound_r)
ax.set_ylim(bound_s, bound_n)

plt.show()


# """
# 局部放大描绘是否有网格的区别
# """
# 
# temp1 = 2000
# temp2 = 2500
# temp3 = 2400
# temp4 = 2900
# 
# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# 
# for p in set_clc:
#     
#     if POINTS[p]["coord"][0] > temp1 and POINTS[p]["coord"][0] < temp2 and POINTS[p]["coord"][1] > temp3 and POINTS[p]["coord"][1] < temp4:
# 
#         ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "black", marker  = '.', s = 30)  
#     
# for p in set_bc:
#     
#     if POINTS[p]["coord"][0] > temp1 and POINTS[p]["coord"][0] < temp2 and POINTS[p]["coord"][1] > temp3 and POINTS[p]["coord"][1] < temp4:
#     
#         ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "black", marker  = '.', s = 30)
# 
# for p in set_d:
#     
#     if POINTS[p]["coord"][0] > temp1 and POINTS[p]["coord"][0] < temp2 and POINTS[p]["coord"][1] > temp3 and POINTS[p]["coord"][1] < temp4:
#     
#         ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "black", marker  = '.', s = 30)
#         
# from math import ceil, floor
# 
# (x_min, x_max) = ax.get_xlim()
# (y_min, y_max) = ax.get_ylim()
# x_min = ceil(x_min / GRID_DIST) * GRID_DIST
# x_max = floor(x_max / GRID_DIST) * GRID_DIST
# y_min = ceil(y_min / GRID_DIST) * GRID_DIST
# y_max = floor(y_max / GRID_DIST) * GRID_DIST
# x_minortk = [i for i in range(x_min, x_max + 1, GRID_DIST)]
# y_minortk = [i for i in range(y_min, y_max + 1, GRID_DIST)]
# ax.set_xticks(x_minortk)
# ax.set_yticks(y_minortk)
# 
# # grid由tick决定！
# ax.grid(alpha = 0.5)
# ax.set_axisbelow(True) # 网格在绘图下面
# 
# # 不显示tick标签
# frame1 = plt.gca()
# frame1.axes.xaxis.set_ticklabels([])
# frame1.axes.yaxis.set_ticklabels([])
# 
# ax.set_xlim(temp1, temp2)
# ax.set_ylim(temp3, temp4)
# 
#         
# plt.show()

# In[186]:


"""
  _____       _   _                 
 |  __ \     | | | |                
 | |__) |   _| |_| |__   ___  _ __  
 |  ___/ | | | __| '_ \ / _ \| '_ \ 
 | |   | |_| | |_| | | | (_) | | | |
 |_|    \__, |\__|_| |_|\___/|_| |_|
         __/ |                      
        |___/                                                           
"""
pass
################# 聚类算法 ##################


# 一部分进入cen，一部分进入sc
# 
# 现有集合：
# 
# POINTS 字典，全局常量，记录点的序号经纬来源
# set_ap 全部点集
# 
# set_d 仓库
# set_ac 所有客户 = set_bc + set_clc，注意不包含d
# set_bc 背景客户，包含在聚类范围内和不在聚类范围内的
# set_clc 聚类客户
# set_cen 聚类范围内，密集地带的客户，遗传算法的研究对象
# set_sc 不在聚类范围的分散客户
# 
# 对于总集ac，c和clc互为补集，cen和sc互为补集
# 集合bc和clc下面很少用到了
# 
# 拓展：
# 可能有部分cen客户点无法停车，城区也会提供额外的停车场
# set_cen增删 -> set_pk所有潜在的停车点 -> 遗传算法挑出set_st实际停车点 -> 对set_cen基于到set_st的距离进行分配
# 由于st本身也在set_cen中，所以会被自然归入对应的子集，子集无名，用字典列表存储

# In[187]:


"""
生成距离字典
允许(i,j) = (j,i) 以及 (i,i) = 0 允许存在，便于编程

提取坐标：
x = POINTS[0]["coord"][0]
y = POINTS[0]["coord"][1]

"""

#"""

DIST = {(i, j): ((POINTS[i]["coord"][0] - POINTS[j]["coord"][0])**2 + 
                 (POINTS[i]["coord"][1] - POINTS[j]["coord"][1])**2)**0.5
                  for i in set_ap for j in set_ap}

#"""

"""
DIST = {(i, j): 0 for i in set_ap for j in set_ap if i != j}
DIST.update({(i,i): 0 for i in set_ap})
"""
pass


# """
# 距离分布
# """
# 
# import matplotlib.pyplot as plt
# 
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# 
# ax.hist(DIST.values(), 200)
# 
# plt.show()

# In[188]:


"""
k-NNG
"""

from sklearn.neighbors import NearestNeighbors

list_samples = []

for p in POINTS:   
    list_samples.append(POINTS[p]["coord"])
    
neigh = NearestNeighbors(n_neighbors = MINPTS)
neigh.fit(list_samples)
array_dist, array_idx = neigh.kneighbors()

from numpy import mean
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

#ax.plot(mean(array_dist, axis = 1))
ax.hist(mean(array_dist, axis = 1), 200)

plt.xlabel("distance (m)")
plt.ylabel("count of distance")

plt.show()


# In[189]:


"""
DBSCAN
"""

matrix_d = []

for i in range(len(set_ap)): # 注意此时矩阵下标与点的id严格对应！后面输出列表clustering.labels_
    l = []
    for j in range(len(set_ap)):
        dist = ((POINTS[i]["coord"][0] - POINTS[j]["coord"][0])**2  + (POINTS[i]["coord"][1] - POINTS[j]["coord"][1])**2)**0.5
        l.append(dist)
    matrix_d.append(l)

import numpy as np

X = np.array(matrix_d)

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps = EPSILON, min_samples = MINPTS, metric = 'precomputed').fit(X) # 注意eps设为radius/1.5，此为经验值

print(clustering.labels_)


# In[190]:


"""
根据DBSCAN的结果划分sc和cen，并且在POINTS字典中修改cluster字段
注意这里不关心到底有几个聚类，DBSCAN结果的标号直接可用
计算从背景点划入cen的数量和原始聚类点未被划入cen的数量
"""

set_cen = set()
set_sc = set()

count_bc2cen = 0
count_cl2sc = 0

for i in range(len(set_ap)):
    
    # 排除d类不用处理，d不加入sc（if==-1）
    if i in set_d: 
        continue
        
    # 读取label并且修改POINTS里面的，很巧-1=-1
    label_old = POINTS[i]["cluster"]
    label = clustering.labels_[i]
    POINTS[i]["cluster"] = label
    
    if label_old == -1 and label != -1:
        count_bc2cen += 1
    elif label_old == 0 and label == -1:
        count_cl2sc += 1
    
    # 加入集合
    if label == -1:
        set_sc.add(i)
    else:
        set_cen.add(i)
        
print(count_bc2cen)
print(count_cl2sc)


# In[191]:


"""
这里要找出到底有几个聚类，便于自动分配颜色
"""

set_ul = set(clustering.labels_) # 用set提取标签
set_ul.discard(-1)
nul = len(set_ul) # 一共有几个聚类？


# In[192]:


"""
计算DBSCAN得出的聚类中点和点之间的平均距离
聚类划分字典
"""

dict_clustering = dict()

for l in set_ul:
    set_cl = set()
    for ip in POINTS:
        if POINTS[ip]["cluster"] == l:
            set_cl.add(ip)
    dict_clustering[l] = set_cl
    count = 0
    sumd = 0
    for p1 in set_cl:
        for p2 in set_cl:
            if p1 != p2:
                sumd += DIST[(p1, p2)]
                count += 1
    #print(sumd/count)

#print(len(dict_clustering))
#print(dict_clustering)


# In[193]:


"""
DBSCAN结果画图
"""

# 字典翻转
dict_alloc = dict()
for k, v in dict_clustering.items():
    dict_alloc.update({p: k for p in v })

import matplotlib.pyplot as plt

cm = plt.cm.get_cmap('hsv')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
for p in set_sc:
    
    ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "black", marker  = '.', s = 30)
    

for p in set_cen:

    #ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "green", marker  = '.', s = 30)
    
    ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = cm(dict_alloc[p]/nul), marker  = '.', s = 30) # 注意cm是函数，不是对象
    

for p in set_d:
    
    ax.scatter(POINTS[p]["coord"][0], POINTS[p]["coord"][1], color = "black", marker  = '^', s = 50)

ax.set_xlim(bound_l, bound_r)
ax.set_ylim(bound_s, bound_n)

plt.show()


# In[194]:


print("All points:")
print(set_ap)
print("Depot:")
print(set_d)
print("Background customers:")
print(set_bc)
print("Customers in cluster:")
print(set_clc)
print("Customers in city center:")
print(set_cen)
print("Sparse customers:")
print(set_sc)


# In[195]:


"""
  _____       _   _                 
 |  __ \     | | | |                
 | |__) |   _| |_| |__   ___  _ __  
 |  ___/ | | | __| '_ \ / _ \| '_ \ 
 | |   | |_| | |_| | | | (_) | | | |
 |_|    \__, |\__|_| |_|\___/|_| |_|
         __/ |                      
        |___/                                                           
"""
pass
################# 公共函数 ##################


# In[196]:


"""
设置参数，最长时间，最大容量，速度
公共变量
"""

def setparam(mode):
    
    global Tmax
    global Cmax
    global vl
    global vh
    global vw
    
    if mode == "walk":
        Tmax = TMW
        Cmax = CM
        vw = VW
        
    elif mode == "drive":
        Tmax = TMD
        Cmax = CM
        vl = VDL
        vh = VDH
        
    elif mode == "drive only":
        Tmax = TMD
        Cmax = CM
        vl = VDL
        vh = VDH
        
    else:
        assert(False)


# In[197]:


def tourtime(tour, mode, set_st, dict_wtime, set_cen):
    
    setparam(mode)
    
    if mode == "walk":
        """
        可能的模式：
        (st,)！
        (st, cen, st)
        (st, cen, cen, cen, st)
        """
        
        t_total = 190 # st点本身耗时，160 + ... + 30
        
        if len(tour) == 1: # 如果是单客户!       
            return t_total
        
        else:     
            for p in tour[1: -1]: # 掐头去尾切片  
                t_total += 130 # 投递时间+分配的取货时间，每多一个客户，取货时间增加十秒
                
            for i in range(0, len(tour) - 1):
                t_total += DIST[tour[i: i + 2]] / vw
                
            return t_total
        
    elif mode == "drive":
        """
        可能的模式：
        (d, sc, d)
        (d, st, d)
        (d, sc, st, sc, d)
        """
        t_total = 0 # 准备时间
        
        for p in tour[1: -1]:     
            if p in set_st: # 开车旅程经过聚类停车点，后接步行送货
                t_total += (30 + dict_wtime[p]) # 停车+步行旅程总时间，190比220少的30秒在这！
                #print(str(p)+" in st")
                #print("WTIME: " + str(dict_wtime[p]))
            elif p in set_sc: #  开车旅程经过离散客户停车点
                t_total += 220 # 停车取货步行投递步行
                #print(str(p)+" in sc")
            else:
                assert(False, "Set wrong")
                
        
        # 城区车速较低        
        for i in range(0, len(tour) - 1):
            p1 = tour[i]
            p2 = tour[i + 1]
            if p1 in set_cen and p2 in set_cen:
                v = vl
            else:
                v = vh
            t_total += DIST[(p1, p2)] / v
            
        return t_total
    
    elif mode == "drive only":
        
        t_total = 0

        for p in tour[1: -1]:     
            t_total += 220
            
        # 城区车速较低      
        for i in range(0, len(tour) - 1):
            p1 = tour[i]
            p2 = tour[i + 1]
            if p1 in set_cen and p2 in set_cen:
                v = vl
            else:
                v = vh
            t_total += DIST[(p1, p2)]/ v
            
        return t_total
            
    else:
        assert(False)


# In[198]:


def tourdist(tour):
    
    dist = 0
    
    for i in range(0, len(tour) - 1):

        dist += DIST[tour[i: i + 2]]

    return dist


# In[199]:


def cws(set_p, id_d, mode, set_st, dict_wtime, set_cen):

    # set_p 必须以集合形式输入！因为后面的adjvs
    
    setparam(mode)

    # 生成SPL

    set_tm = set() # 临时集合，防重复

    SPL = [] # 用列表因为需要排序

    for i in set_p:
        for j in set_p:
            if i != j and not (j, i) in set_tm: # 防止重复
                set_tm.add((i, j))
                sav = DIST[(id_d, i)] + DIST[(id_d, j)] - DIST[(i, j)]
                SPL.append((sav, (i, j)))

    # 排序SPL

    from operator import itemgetter

    SPL.sort(key = itemgetter(0), reverse = True) # 注意sorted(obj)函数返回临时对象，obj.sort()方法直接在对象上操作
    
    """
    SPL格式:
    [
    (saving, (i, j))
    (saving, (i, j))
    (saving, (i, j))
    ...
    ]
    """
    
    # 生成初始旅程
    
    tours = set([(id_d, i, id_d) for i in set_p]) # ((tour1), (tour2),...)，先生成[()]，再转为{()}
    
    for t in tours:
        if tourtime(t, mode, set_st, dict_wtime, set_cen) > Tmax:
            
            print("Tour " + str(t) + " infeasible!")
            print(Tmax)
            print(set_st)
            print(mode)
            print(dict_wtime)
            
            return None # 初始旅程就有不可行的，直接退出。CWS只会让每个tour的时间越来越长
    
    # 准备完毕，以下是算法主体

    idx_sp = 0

    adjvs = set(set_p) # 如果新路径“长度”大于4则从此集合中去掉非邻接点，注意是深拷贝！

    while idx_sp != len(SPL):

        p1 = SPL[idx_sp][1][0] # SPL: [(saving,(i,j)),...]
        p2 = SPL[idx_sp][1][1]
        """
        应用savings pair的条件是
        1 两点不在同一环路中
        2 两点都是起点的邻接点
        """
        # 检查邻接
        if not p1 in adjvs or not p2 in adjvs:
            idx_sp = idx_sp + 1
            continue # 退出1
        # 检查所属环路
        for t1 in tours:
            if p1 in t1:
                break
        for t2 in tours:
            if p2 in t2:
                break
        if t1 == t2: # 如果在同一环路中
            idx_sp = idx_sp + 1
            continue # 退出2
        # 翻转
        if t1[1] == p1:
            ts = t1[::-1] # 巧用切片翻转
        else:
            ts = t1
        if t2[-2] == p2:
            te = t2[::-1]
        else:
            te = t2
        """
        t1 p1所在旅程
        t2 p2所在旅程
        ts t1处理后形成的前半段
        te t2处理后形成的后半段
        tn ts+te去尾掐头合并形成的新旅程
        """
        # 连接
        tn = ts[:-1] + te[1:] # 去尾掐头
        # 检查时间和容量
        if len(tn) > Cmax:
            idx_sp = idx_sp + 1
            continue # 退出3
        if tourtime(tn, mode, set_st, dict_wtime, set_cen) > Tmax:
            idx_sp = idx_sp + 1
            continue # 退出4
        # 注意上文有四个退出点
        # 如果条件都无误则修改集合tours
        tours.add(tn) # 增加新的
        tours.remove(t1) # 删除旧的
        tours.remove(t2)
        # 删减邻接点集合
        if len(tn) > 4:
            for p in tn[2:-2]:
                adjvs.discard(p) # 集合重复去除元素，因此用discard()而不是remove()方法
        idx_sp = idx_sp + 1 
      
    return tours

"""
注意，即使savings为0或者负，以上代码依然会合并，没有检查。即使距离全为0，代码依然尝试合并
"""
pass


# In[200]:


"""
  _____       _   _                 
 |  __ \     | | | |                
 | |__) |   _| |_| |__   ___  _ __  
 |  ___/ | | | __| '_ \ / _ \| '_ \ 
 | |   | |_| | |_| | | | (_) | | | |
 |_|    \__, |\__|_| |_|\___/|_| |_|
         __/ |                      
        |___/                                                           
"""
pass
################# 老算法 ##################


# In[201]:


"""
行车路线规划
"""

import time

t_s = time.time()

# 运行
set_tours = cws(set_ac, 0, "drive only", set(), dict(), set_cen)

t_e = time.time()

runtime1 = t_e - t_s

print(runtime1)


# In[202]:


len(set_tours)


# In[203]:


"""
结果呈现
"""

import matplotlib.pyplot as plt

cm = plt.cm.get_cmap('hsv') # colormap接受0-1值，hsv色彩效果较好

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 画路径

nt = len(set_tours)
ccount = 1

for t in set_tours: # 遍历TOURS
    x = []
    y = []
    for p in t: # 遍历一个旅程t中的所有点
        x.append(POINTS[p]["coord"][0])
        y.append(POINTS[p]["coord"][1])
    ax.plot(x, y, color = cm(ccount/nt), linewidth = 1, zorder = 1) # zorder规定了层叠顺序，数字越大越在上
    ccount += 1
    
# 画点
    
for p in set_ap:
    x = POINTS[p]["coord"][0]
    y = POINTS[p]["coord"][1]
    if p in set_d:
        c = "black"
        mk = "^"
        alpha = 1
        sz = 100
    elif p in set_sc:
        c = "blue"
        mk = "o"
        alpha = 1
        sz = 20
    elif p in set_cen:
        c = "orange"
        mk = "o"
        alpha = 0.5
        sz = 20
    ax.scatter(x, y, color = c, zorder = 2, marker = mk, alpha = alpha, s = sz)
    #ax.text(x - 1.5, y + 3, p, fontsize = 10)

"""
ax.set_xlim(bound_l, bound_r)
ax.set_ylim(bound_s, bound_n)
"""
    
# 用于legend的虚拟点线
# 参见https://stackoverflow.com/questions/47391702/matplotlib-making-a-colored-markers-legend-from-scratch
# 参见https://www.matplotlib.org.cn/gallery/text_labels_and_annotations/custom_legends.html

from matplotlib.lines import Line2D
dummies = []
dummies.append(Line2D([], [], linestyle='None', color = "black", marker = "^", alpha = 1, label = "Depot"))
dummies.append(Line2D([], [], linestyle='None', color = "blue", marker = "o", alpha = 1, label = "Sparse customers"))
dummies.append(Line2D([], [], linestyle='None', color = "orange", marker = "o", alpha = 0.5, label = "Customers in clusters"))
dummies.append(Line2D([0], [0], color = cm(0), label = "(and other colors) Driving tours"))
ax.legend(handles = dummies)
    
plt.show()

print(len(set_tours))
print(set_tours)


# In[204]:


"""
计算行车距离
"""

D_d1 = 0

for t in set_tours:
    
    D_d1 += tourdist(t)

D_d1


# In[205]:


"""
计算总时间
"""

T_totmin1 = 0
D_tot1 = 0
for t in set_tours:
    T_totmin1 += tourtime(t, "drive only", set(), dict(), set_cen)
print(T_totmin1)


# In[206]:


"""
  _____       _   _                 
 |  __ \     | | | |                
 | |__) |   _| |_| |__   ___  _ __  
 |  ___/ | | | __| '_ \ / _ \| '_ \ 
 | |   | |_| | |_| | | | (_) | | | |
 |_|    \__, |\__|_| |_|\___/|_| |_|
         __/ |                      
        |___/                                                           
"""
pass
################# 新算法 ##################


# In[207]:


list_cen = list(set_cen)
list_cen.sort()

"""
健康度函数
"""

def fitness_func(solution, solution_idx):
    
    # 根据当前解设定st
    set_st = set()
    for i in range(len(solution)):
        if solution[i] == 1:
            set_st.add(list_cen[i])

    dict_palloc = dict() # 每个st点分配的cen点
    dict_wtime = dict() # 每个st点的工作时间
    dict_stwt = dict() # 每个st点的旅程集合    

    if len(set_st) != 0:

        for p in set_st:
            dict_palloc[p] = []

        # 对每个st分配cen
        for pc in set_cen: # 遍历cen
            d_min = -1 # 最小距离初始值
            pst_best = -1
            for pst in set_st: # 遍历st
                if d_min == -1 or DIST[(pc, pst)] < d_min: # 如果是初始值或找到更小的距离（即更好的st）
                    d_min = DIST[(pc, pst)]
                    pst_best = pst  
            """
            l = dict_palloc[pst_best]
            l.append(pc)
            dict_palloc[pst_best] = l
            用不着这样，直接append！
            """         
            dict_palloc[pst_best].append(pc)

        # 注意这里不会出现st对应的集合(dict_palloc的value)为空的情况，因为st点本身也是cen点，距离为0，单点只能是如{50;[50]}这样的

        # 对每个st进行路线规划       
        for pst in set_st:
            set_cwswalk = set(dict_palloc[pst]) # 这里必须转成set
            set_cwswalk.remove(pst) # 删除起点st本身
            if len(set_cwswalk) == 0: # 单st即停即送的情况留给tourtime()函数处理！
                dict_stwt[pst] = {(pst,),} 
            else:
                re = cws(set_cwswalk, pst, "walk", set_st, dict_wtime, set_cen) # 对pst和分配的客户点进行cws
                if re == None: # 如果旅程不可行，直接把fitness value降到极低
                    return 1 / 99999999
                dict_stwt[pst] = re

            # 计算st的步行工作时间
            wtime = 0
            for t in dict_stwt[pst]:
                wtime += tourtime(t, "walk", set_st, dict_wtime, set_cen)
            dict_wtime[pst] = wtime

        set_cwsdrive = set_st.union(set_sc) 

        re = cws(set_cwsdrive, 0, "drive", set_st, dict_wtime, set_cen)
        if re == None:
            return 1 / 99999999
        set_dt = re

        T_total = 0

        for dt in set_dt:
            T_total += tourtime(dt, "drive", set_st, dict_wtime, set_cen)

    else:

        set_dt = cws(set_ac, 0, "drive only", set_st, dict_wtime, set_cen) # 注意这里不能写成"drive"，因为在该模式下，tourtime函数只对set_sc里面的点加时间

        T_total = 0

        for dt in set_dt:
            T_total += tourtime(dt, "drive only", set_st, dict_wtime, set_cen)
            
    return 1 / T_total


# In[208]:


"""
重复上文以复现结果
"""

def get_sdetail(solution):
    
    set_st = set()
    for i in range(len(solution)):
        if solution[i] == 1:
            set_st.add(list_cen[i])

    dict_palloc = dict() # 每个st点分配的cen点
    dict_wtime = dict() # 每个st点的工作时间
    dict_stwt = dict() # 每个st点的旅程集合    

    if len(set_st) != 0:

        for p in set_st:
            dict_palloc[p] = []

        # 对每个st分配cen
        for pc in set_cen: # 遍历cen
            d_min = -1 # 最小距离初始值
            pst_best = -1
            for pst in set_st: # 遍历st
                if d_min == -1 or DIST[(pc, pst)] < d_min: # 如果是初始值或找到更小的距离（即更好的st）
                    d_min = DIST[(pc, pst)]
                    pst_best = pst  
            """
            l = dict_palloc[pst_best]
            l.append(pc)
            dict_palloc[pst_best] = l
            用不着这样，直接append！
            """         
            dict_palloc[pst_best].append(pc)

        # 注意这里不会出现st对应的集合(dict_palloc的value)为空的情况，因为st点本身也是cen点，距离为0，单点只能是如{50;[50]}这样的

        # 对每个st进行路线规划       
        for pst in set_st:
            set_cwswalk = set(dict_palloc[pst]) # 这里必须转成set
            set_cwswalk.remove(pst) # 删除起点st本身
            if len(set_cwswalk) == 0: # 单st即停即送的情况留给tourtime()函数处理！
                dict_stwt[pst] = {(pst,),} 
            else:
                dict_stwt[pst] = cws(set_cwswalk, pst, "walk", set_st, dict_wtime, set_cen) # 对pst和分配的客户点进行cws

            # 计算st的步行工作时间
            wtime = 0
            for t in dict_stwt[pst]:
                wtime += tourtime(t, "walk", set_st, dict_wtime, set_cen)
            dict_wtime[pst] = wtime

        set_cwsdrive = set_st.union(set_sc) 

        set_dt = cws(set_cwsdrive, 0, "drive", set_st, dict_wtime, set_cen)

        T_total = 0

        for dt in set_dt:
            T_total += tourtime(dt, "drive", set_st, dict_wtime, set_cen)

    else:

        set_dt = cws(set_ac, 0, "drive only", set_st, dict_wtime, set_cen) # 注意这里不能写成"drive"，因为在该模式下，tourtime函数只对set_sc里面的点加时间

        T_total = 0

        for dt in set_dt:
            T_total += tourtime(dt, "drive only", set_st, dict_wtime, set_cen)
            
    return (set_st, dict_palloc, dict_stwt, dict_wtime, set_dt, T_total)


# In[209]:


"""
随机生成代替GA
"""

import random

solution = []

for i in range(len(set_cen)):
    
    ran = random.uniform(0,1)
    
    if ran < 0.64:
        
        solution.append(1)
    
    else:
        
        solution.append(0)
        
Best_Solution = get_sdetail(solution)


# In[210]:


"""
提取活跃连接（行车路径）
字典{(连接):路径id}
"""     
dict_ae2 = dict()
li = list(Best_Solution[-2])

for it in range(len(li)):
    for iseg in range(len(li[it]) - 1):
        dict_ae2[(li[it][iseg], li[it][iseg + 1])] = it


# In[211]:


"""
画复合路径
"""

import matplotlib.pyplot as plt

cm = plt.cm.get_cmap('hsv')

fig, ax = plt.subplots(1, 1, figsize=(9, 9))

# 画行车路径

cm = plt.cm.get_cmap('hsv')

nt = len(Best_Solution[-2])

for p1 in set_ap:
    for p2 in set_ap:
        if p1 != p2:
            if (p1, p2) in dict_ae2.keys():
                x = [POINTS[p1]["coord"][0], POINTS[p2]["coord"][0]]
                y = [POINTS[p1]["coord"][1], POINTS[p2]["coord"][1]]
                ax.plot(x, y, color = cm(dict_ae2[(p1, p2)]/nt), linewidth = 1, zorder = 1)
            if (p2, p1) in dict_ae2.keys():
                x = [POINTS[p1]["coord"][0], POINTS[p2]["coord"][0]]
                y = [POINTS[p1]["coord"][1], POINTS[p2]["coord"][1]]
                ax.plot(x, y, color = cm(dict_ae2[(p2, p1)]/nt), linewidth = 1, zorder = 1)
    
# 画行走路径
    
for t in Best_Solution[2].values():
    for tt in t:
        x = []
        y = []
        for p in tt:
            x.append(POINTS[p]["coord"][0])
            y.append(POINTS[p]["coord"][1])
        ax.plot(x, y, c = "black", linewidth = 1, zorder = 3, linestyle= ':')
        ccount += 1
    
# 画点
    
for p in set_ap:
    x = POINTS[p]["coord"][0]
    y = POINTS[p]["coord"][1]
    if p in set_d:
        c = "black"
        mk = "^"
        alpha = 1
        sz = 100
    elif p in set_sc:
        c = "blue"
        mk = "o"
        alpha = 1
        sz = 20
    elif p in Best_Solution[0] and len(Best_Solution[1][p]) > 1: # 如果p在st里并且有长于1的子路径
        c = "green"
        mk = "o"
        alpha = 0.5
        sz = 20
    elif p in Best_Solution[0] and len(Best_Solution[1][p]) == 1:
        c = "yellow"
        mk = "o"
        alpha = 1
        sz = 20
    elif p in set_cen:
        c = "orange"
        mk = "o"
        alpha = 0.5
        sz = 20
    ax.scatter(x, y, color = c, zorder = 2, marker = mk, alpha = alpha, s = sz)
    #ax.text(x - 1.5, y + 3, p, fontsize = 10)
 


"""
ax.set_xlim(bound_l, bound_r)
ax.set_ylim(bound_s, bound_n)
"""

# 用于legend的虚拟点线

from matplotlib.lines import Line2D
dummies = []
dummies.append(Line2D([], [], linestyle='None', color = "black", marker = "^", alpha = 1, label = "Depot"))
dummies.append(Line2D([], [], linestyle='None', color = "blue", marker = "o", alpha = 1, label = "Sparse customers"))
dummies.append(Line2D([], [], linestyle='None', color = "green", marker = "o", alpha = 0.5, label = "Customers where the vehicle parks for walking subtours"))
dummies.append(Line2D([], [], linestyle='None', color = "orange", marker = "o", alpha = 0.5, label = "Customers reached by walking"))
dummies.append(Line2D([], [], linestyle='None', color = "yellow", marker = "o", alpha = 0.5, label = "Customers reached by driving"))
dummies.append(Line2D([0], [0], color = cm(0), label = "(and other colors) Driving tours"))
dummies.append(Line2D([0], [0], color = "black", linestyle= ':', label = "Walking subtours"))
ax.legend(handles = dummies)

plt.show()

print(Best_Solution[-2])
print(len(Best_Solution[-2]))


# """
# 根据原有的cluster信息dict_clinfo，先找出cluster内点，即set_cl
# """
#     
# x0 = dict_clinfo[icl]["center"][0]
# y0 = dict_clinfo[icl]["center"][1]
# 
# set_cl = set()
# 
# for p in set_ap:
# 
#     x = POINTS[p]["coord"][0]
#     y = POINTS[p]["coord"][1]
# 
#     dist = ((x - x0)**2 + (y - y0)**2)**0.5
# 
#     if dist < dict_clinfo[icl]["radius"]:
# 
#         set_cl.add(p)

# In[212]:


"""
划定放大的矩形区域，找出set_cl
重要！
"""
    
leftbd = 1500
rightbd = 3500
southbd = 1500
northbd = 3500

set_cl = set()

for p in set_ap:

    x = POINTS[p]["coord"][0]
    y = POINTS[p]["coord"][1]

    if x > leftbd and x < rightbd and y > southbd and y < northbd:

        set_cl.add(p)


# In[213]:


"""
局部放大

zorder 层数越靠上越大
底
第一层：行车路经
第二层：除了仓库的点
第三层：箭头
第四层：行走路径
第五层：仓库
顶
"""

fig, ax = plt.subplots(1, 1, figsize=(9, 9))

# 画点
for p in set_cl:

    x = POINTS[p]["coord"][0]
    y = POINTS[p]["coord"][1]

    if p in set_d:
        c = "black"
        mk = "^"
        alpha = 1
        sz = 100
        zo = 5
    elif p in set_sc:
        c = "blue"
        mk = "o"
        alpha = 1
        sz = 20
        zo = 2
    elif p in Best_Solution[0] and len(Best_Solution[1][p]) > 1:
        c = "green"
        mk = "o"
        alpha = 1
        sz = 20
        zo = 2
    elif p in Best_Solution[0] and len(Best_Solution[1][p]) == 1:
        c = "yellow"
        mk = "o"
        alpha = 1
        sz = 20
        zo = 2
    elif p in set_cen:
        c = "orange"
        mk = "o"
        alpha = 1
        sz = 20
        zo = 2

    ax.scatter(x, y, color = c, zorder = zo, marker = mk, alpha = alpha, s = sz)
    #ax.text(x - 1.5, y + 3, p, fontsize = 10)


# 提取st的行走路线并且画图
for p in set_cl:

    if p in Best_Solution[0]:

        x = []
        y = []
        for t in Best_Solution[2][p]: # 遍历一个旅程t中的所有点
            for pt in t:
                x.append(POINTS[pt]["coord"][0])
                y.append(POINTS[pt]["coord"][1])
        ax.plot(x, y, c = "black", linewidth = 1, zorder = 4, linestyle= ':') # zorder规定了层叠顺序，数字越大越在上
        ccount += 1

# 聚类内行车路线

cm = plt.cm.get_cmap('hsv')

nt = len(Best_Solution[-2])

for p1 in set_cl:
    for p2 in set_cl:
        if p1 != p2:
            if (p1, p2) in dict_ae2.keys():
                x = [POINTS[p1]["coord"][0], POINTS[p2]["coord"][0]]
                y = [POINTS[p1]["coord"][1], POINTS[p2]["coord"][1]]
                ax.plot(x, y, color = cm(dict_ae2[(p1, p2)]/nt), linewidth = 1, zorder = 1)
            if (p2, p1) in dict_ae2.keys():
                x = [POINTS[p1]["coord"][0], POINTS[p2]["coord"][0]]
                y = [POINTS[p1]["coord"][1], POINTS[p2]["coord"][1]]
                ax.plot(x, y, color = cm(dict_ae2[(p2, p1)]/nt), linewidth = 1, zorder = 1)
                

# 指向聚类外的行车路线用箭头表示
for (p1, p2) in dict_ae2.keys():
    
    if p1 in set_cl and not p2 in set_cl:
        x1 = POINTS[p1]["coord"][0]
        x2 = POINTS[p2]["coord"][0]
        y1 = POINTS[p1]["coord"][1]
        y2 = POINTS[p2]["coord"][1]
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        ax.arrow(x1, y1, (x2 - x1) / length * 50,
                (y2 - y1) / length * 50,
                 color = cm(dict_ae2[(p1, p2)]/nt),
                 linewidth = 2,
                 head_width = 10,
                 length_includes_head = True,
                 zorder = 3
                 )
        
    if p2 in set_cl and not p1 in set_cl:
        x1 = POINTS[p1]["coord"][0]
        x2 = POINTS[p2]["coord"][0]
        y1 = POINTS[p1]["coord"][1]
        y2 = POINTS[p2]["coord"][1]
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        ax.arrow(x2 - (x2 - x1) / length * 50,
                 y2 - (y2 - y1) / length * 50,
                 (x2 - x1) / length * 50, (y2 - y1) / length * 50,
                 color = cm(dict_ae2[(p1, p2)]/nt),
                 linewidth = 2,
                 head_width = 10,
                 length_includes_head = True,
                 zorder = 3
                 )
        
plt.axis('scaled') # xy轴缩放相同

if GRID_FLAG:

    # 通过minorticks画网格，间距为GIRD_DIST
    from math import ceil, floor

    (x_min, x_max) = ax.get_xlim()
    (y_min, y_max) = ax.get_ylim()
    x_min = ceil(x_min / GRID_DIST) * GRID_DIST
    x_max = floor(x_max / GRID_DIST) * GRID_DIST
    y_min = ceil(y_min / GRID_DIST) * GRID_DIST
    y_max = floor(y_max / GRID_DIST) * GRID_DIST
    x_minortk = [i for i in range(x_min, x_max + 1, GRID_DIST)]
    y_minortk = [i for i in range(y_min, y_max + 1, GRID_DIST)]
    ax.set_xticks(x_minortk)
    ax.set_yticks(y_minortk)
    
    # grid由tick决定！
    ax.grid(alpha = 0.5)
    ax.set_axisbelow(True) # 网格在绘图下面
    
    # 不显示tick标签
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    
    #ax.set_xticklabels(ax.get_xticks(), rotation = 90) # x轴标转动90度

    
# 用于legend的虚拟点线

from matplotlib.lines import Line2D
dummies = []
dummies.append(Line2D([], [], linestyle='None', color = "black", marker = "^", alpha = 1, label = "Depot"))
dummies.append(Line2D([], [], linestyle='None', color = "blue", marker = "o", alpha = 1, label = "Sparse customers"))
dummies.append(Line2D([], [], linestyle='None', color = "green", marker = "o", alpha = 1, label = "Customers where the vehicle parks for walking subtours"))
dummies.append(Line2D([], [], linestyle='None', color = "orange", marker = "o", alpha = 1, label = "Customers reached by walking"))
dummies.append(Line2D([], [], linestyle='None', color = "yellow", marker = "o", alpha = 1, label = "Customers reached by driving"))
dummies.append(Line2D([0], [0], color = cm(0), label = "(and other colors) Driving tours"))
dummies.append(Line2D([0], [0], color = "black", linestyle= ':', label = "Walking subtours"))
ax.legend(handles = dummies)

# 画图

plt.show()


# In[214]:


"""
最佳结果输出
"""

list_label = ["parking points",
              "point allocation to parking points",
              "walking subtour by each parking point",
              "working time by each parking point",
              "driving tours",
              "total duration"]

for i in range(len(Best_Solution)):
    print(i)
    print(list_label[i])
    print(Best_Solution[i])
    
flag = []

for t in Best_Solution[-2]:
    for p in t:
        if p in Best_Solution[1]:
            flag.append(1)
        else:
            flag.append(0)

#print(flag)


# In[215]:


"""
计算行车行走距离
"""

D_d2 = 0

for t in Best_Solution[4]:
    
    D_d2 += tourdist(t)

D_w2 = 0

for v in Best_Solution[2].values():
        
    for t in v:

        if len(t) > 1:

            D_w2 += tourdist(t)


# In[216]:


"""
结果记录
"""

T_totmin2 = Best_Solution[-1]
print(T_totmin1)
print(T_totmin2)
print(D_d1)
print(D_d2)
print(str(int((D_d1-D_d2)/D_d1*100))+"%")
print(D_w2)


import pickle

with open("results.txt", "rb") as f:
    
    dict_results = pickle.load(f)
    
count = len(dict_results) # 存储数据中已有长度

runtime2 = 10

dict_results[count] = (T_totmin1, T_totmin2, D_d1, D_d2, D_w2, runtime1, runtime2)

with open("results.txt", "wb") as f:
    
    pickle.dump(dict_results, f)


# import pickle
# 
# with open("st.txt", "rb") as f:
#     
#     dict_stops = pickle.load(f)
#     
# count = len(dict_stops) # 存储数据中已有长度
# 
# set_stad = set() # 当前实验的停车点坐标
# 
# for p in Best_Solution[0]:
#     set_stad.add(POINTS[p]["coord"])
# 
# dict_stops[count] = set_stad
# 
# with open("st.txt", "wb") as f:
#     
#     pickle.dump(dict_stops, f)

# In[217]:


import pickle

with open("st.txt", "rb") as f:
    
    dict_stops = pickle.load(f)
    
count = len(dict_stops) # 存储数据中已有长度

set_stad = set() # 当前实验的停车点坐标

for p in Best_Solution[0]:
    
    if len(Best_Solution[1][p])> 1:
        
        set_stad.add(POINTS[p]["coord"])

dict_stops[count] = set_stad

with open("st.txt", "wb") as f:
    
    pickle.dump(dict_stops, f)


# In[218]:


import winsound

winsound.PlaySound("*", winsound.SND_ALIAS)

