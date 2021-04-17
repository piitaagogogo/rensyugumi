# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:01:39 2019

@author: ryo
"""

# In[0]:準備
import pulp
import numpy as np
import pandas as pd
import time
import csv

start = time.time()
# In[1]:入力

N = "C:\\Users\\tai\\Desktop\\renshugumi\\219n.csv"
#その日のarr_N.csvファイル
A = "C:\\Users\\tai\\Desktop\\renshugumi\\219A.csv"
#その日のAttend.csvファイル
K = 2 #被り人数の上限数
W = 4 #同時にする練習数


# In[0]:また，準備
arr_N = np.genfromtxt(N, delimiter = ",", skip_header = 3 , dtype = None)#Nを行列に
df_N = pd.read_csv(N,encoding="shift-jis")#名前、練習名の取得用


name_dic = {}#メンバー名辞書
for i in range(len(df_N .iloc[0])-3):
    name_dic[i+1] = df_N .iloc[0][i+3]

train_dic = {}#練習名辞書
for i in range(len(df_N )-2):
    train_dic[i+1] = df_N .iloc[i+2][1]

#j = やる練習 指定
train_num_list = []
for i in range(len(arr_N)):
    if int(arr_N[i][2]) == 1:
        train_num_list.append(i+1)


#練習に必要なメンバー行列（arr_N)
col = []
for i in range(len(arr_N[0])-3):
    col.append(i+3)
arr_N = np.genfromtxt(N, delimiter = ",", skip_header = 3 , dtype = None , usecols = col)
for i in range(len(arr_N)):
    for j in range(len(arr_N[0])):
        if arr_N[i,j] == -1:
            arr_N[i,j] = 0


      
         
#メンバー出席簿(arr_A)        
arr_A = np.genfromtxt(A, delimiter = ",", skip_header = 3 , dtype = None)
col = []
for i in range(len(arr_A[0])-2):
    col.append(i+2)
arr_A = np.genfromtxt(A, delimiter = ",", skip_header = 3 , dtype = None , usecols = col)
for i in range(len(arr_A)):
    for j in range(len(arr_A[0])):
        if arr_A[i,j] == -1:
            arr_A[i,j] = 0
            
#i = member 指定
member_num_list = []
for i in range(len(arr_N[0])):
    member_num_list.append(i+1)
#ｔ = コマ 指定
koma_list = []
for i in range(len(arr_A)):
    koma_list.append(i+1)

# In[3]:最適化問題
m = pulp.LpProblem("bestpractice", pulp.LpMaximize)#問題定義
#変数定義
x = pulp.LpVariable.dicts('X', (member_num_list, train_num_list, koma_list), 0, 1, pulp.LpInteger)#tコマにiさんがj練習をするかどうか
y = pulp.LpVariable.dicts('X', (train_num_list, koma_list), 0, 1, pulp.LpInteger)#tコマにj練習をするかどうか
#目的関数定義
m += pulp.lpSum(x[i][j][t] for i in member_num_list for j in train_num_list for t in koma_list ), "TotalPoint"
#制約
for t in koma_list:#練習はまとめて4つまで　t期にする練習全部足したらw以下に
    m += pulp.lpSum(y[j][t] for j in train_num_list) <= W              
for j in train_num_list:#同じ練習は1回まで　jの練習に対して全期分足したら1以下に    
    m += pulp.lpSum(y[j][t] for t in koma_list) <= 1            
for i in member_num_list:#やる練習にしか参加できない　参加しないのはあり    
    for j in train_num_list:
        for t in koma_list:
            m += x[i][j][t] <= y[j][t]
for i in member_num_list:#必要な練習しかしない　必要な練習でもやらないのはあり
    for j in train_num_list:
        for t in koma_list:
            m += x[i][j][t] <= arr_N[j-1][i-1]
for i in member_num_list:#いる人しか参加しない　いる人で参加しないのはあり
    for j in train_num_list:
        for t in koma_list:
            m += x[i][j][t] <= arr_A[t-1][i-1]
for i in member_num_list:#tコマで1人ができる練習は1つまで
    for t in koma_list:
        m += pulp.lpSum(x[i][j][t] for j in train_num_list) <= 1
for t in koma_list:#各期の参加人数はいる人でやる練習に参加可の人の合計よりK人少ない人数以上必要
    m += pulp.lpSum(x[i][j][t] for i in member_num_list for j in train_num_list) >= pulp.lpSum(arr_N[j-1][i-1]*arr_A[t-1][i-1]*y[j][t] for i in member_num_list for j in train_num_list) - K
#解く
m.solve()
#Optimalかどうか
print (pulp.LpStatus[m.solve()])

# In[4]:結果
# #全期通して練習できた合計人数
if pulp.value(m.objective)==None:
    print("errorです。EXCELを確認しましょう。")

elif pulp.value(m.objective) >= 0:
    print("総練習人数は"+str(pulp.value(m.objective))+"人") 
#jの練習をt期にするというのを出力
    print("========================================================================")
    print("【仮メニュー結果】")
    t1 = 0
    j1 = 0
    for t in koma_list:
        tt = 0
        for j in train_num_list:
            if pulp.value(y[j][t]) == 1:
                if t1 != t:
                    t1 = t
                if j1 != j:
                    j1 = j
                    tt += 1
                    if tt == 1:
                        print(t)
                    print(train_dic[j])                 
#いつだれがかぶっているか出力
    print("========================================================================")
    print("【被りキャスト】")
    for t in koma_list:
        tt = 0
        for j in train_num_list:
            for i in member_num_list:
                if pulp.value(x[i][j][t]) != pulp.value(arr_N[j-1][i-1]*arr_A[t-1][i-1]*y[j][t]):
                    tt += 1
                    if tt == 1:
                        print(t)
                    print(name_dic[i])
#そのコマに練習できていない人　if arr_A == 1  arr_N!=xijt の人？…かぶってる人？
#単純にいるけどそのコマにやることない人を出しておく
    print("========================================================================")
    print("【練習あぶれキャスト】")
    for t in koma_list:
        tt = 0
        for i in member_num_list:
            if arr_A[t-1][i-1] == 1:
                #print(pulp.value(pulp.lpSum(x[i][j][t] for j in train_num_list)))
                if pulp.value(pulp.lpSum(x[i][j][t] for j in train_num_list)) == 0:
                    tt += 1
                    if tt == 1:
                        print(t)
                    print(name_dic[i])

    #計算の実行時間を秒で記述
    print("========================================================================")
    elapsed_time = time.time() - start
    print ("計算時間:{0}".format(elapsed_time) + "[秒]")

else:
    print("練習は入り切りませんでした。")