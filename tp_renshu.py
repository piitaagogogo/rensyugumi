# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:01:39 2019

@author: ryo
"""
import numpy as np
import pulp
import pandas as pd
import time

start = time.time()

Needfile = "C:\\Users\\tai\\Desktop\\renshugumi\\219n.csv"
Attendfile = "C:\\Users\\tai\\Desktop\\renshugumi\\219A.csv"
K = 10   #被り人数の上限数
W = 10   #同時にする練習数

df_N = pd.read_csv(Needfile, encoding = 'shift-jis', header=None).dropna(how="all").dropna(how="all",axis=1)

training = len(df_N.index)-3
member = len(df_N.columns)-3

# 全メンバーのリストを獲得。
# りょーさんコードでpeopleに対応。
member_num_list = list(range(1,member+1))
# Need由来の２。メンバー名と番号の対応辞書。namedic。
name_dic = {}
for i in range(member):
    name_dic[i+1] = df_N.iloc[1][i+3]
# Need由来の３。トレーニング名と番号の辞書。traindic
train_dic = {}
for i in range(training):
    train_dic[i+1] = df_N.iloc[i+3][1]
# Need由来の４。参加者データをnd配列にしたもの。履修者名簿。N_np。
arr_N =  df_N.fillna(0).drop(df_N.index[[0,1,2]]).drop(df_N.columns[[0,1,2]],axis=1).values.astype(np.int)
# Need由来の5。時間を割きたい練習のみの番号リスト。最もトリッキーなリスト。training。
train_num_list = []
for i in range(training):
    if int(arr_N[i][2]) == 1:
        train_num_list.append(i+1)
# Attendファイルの参照
df_A = pd.read_csv(Attendfile, encoding = 'shift-jis', header=None).dropna(how="all").dropna(how="all",axis=1)
# 練習時間のコマ番号をリストとして獲得。times。
koma_list = list(range(1,len(df_A.index)-2))
# Attend由来。出席データを配列化したもの。出席者名簿。A_np。
arr_A = df_A.fillna(0).drop(df_A.index[[0,1,2]]).drop(df_A.columns[[0,1]],axis=1).values.astype(np.int)
#コマ名リスト。timezone。
timezone_list = df_A.iloc[3:,1].values
# ここまでデータ整理


# ここからPulpによる最適化
m = pulp.LpProblem("bestpractice", pulp.LpMaximize)
x = pulp.LpVariable.dicts('X', (member_num_list, train_num_list, koma_list), 0, 1, pulp.LpInteger)#tコマにiさんがj練習をするかどうか
y = pulp.LpVariable.dicts('Y', (train_num_list, koma_list), 0, 1, pulp.LpInteger)#tコマにj練習をするかどうか
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
for i in member_num_list:   #いる人しか参加しない　いる人で参加しないのはあり
    for j in train_num_list:
        for t in koma_list:
            m += x[i][j][t] <= arr_A[t-1][i-1]
for i in member_num_list:   #tコマで1人ができる練習は1つまで
    for t in koma_list:
        m += pulp.lpSum(x[i][j][t] for j in train_num_list) <= 1
for t in koma_list:#各期の参加人数はいる人でやる練習に参加可の人の合計よりK人少ない人数以上必要
    m += pulp.lpSum(x[i][j][t] for i in member_num_list for j in train_num_list) >= pulp.lpSum(arr_N[j-1][i-1]*arr_A[t-1][i-1]*y[j][t] for i in member_num_list for j in train_num_list) - K
m += pulp.lpSum(y[j][t] for j in train_num_list for t in koma_list) == len(train_num_list)#入れた練習は全て採用する

m.solve()

print(pulp.LpStatus[m.solve()])
print("練習数は"+str(len(train_num_list)))


# **結果**

# In[ ]:


print("=============================")
if pulp.LpStatus[m.solve()] != "Infeasible":
    print ("練習は入りきった！")
    print("総練習人数は"+str(round(pulp.value(m.objective)))+"人") 
#結果
    print("=============================")
    print("【メニュースケジュール】")
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
                        print(timezone_list[t-1] )
                    print(train_dic[j]) 
    print("=============================")
    print("【被り】")
    for t in koma_list:
        tt = 0
        for j in train_num_list:
            for i in member_num_list:
                if pulp.value(x[i][j][t]) != pulp.value(arr_N[j-1][i-1]*arr_A[t-1][i-1]*y[j][t]):
                    tt += 1
                    if tt == 1:
                        print(timezone_list[t-1] )
                    print(name_dic[i])
    print("=============================")
    print("【やることない】")
    for t in koma_list:
        tt = 0
        for i in member_num_list:
            if arr_A[t-1][i-1] == 1:
                #print(pulp.value(pulp.lpSum(x[i][j][t] for j in train_num_list)))
                if pulp.value(pulp.lpSum(x[i][j][t] for j in train_num_list)) == 0:
                    tt += 1
                    if tt == 1:
                        print(timezone_list[t-1] )
                    print(name_dic[i])
    
else:
    print("練習は入り切らなかった．被り人数許容上限="+str(K)+",同時練習許容上限="+str(W))
    print("K(被り人数許容上限)やW(同時練習許容上限)を大きくするか，練習するメニュー減らしてみてね")
