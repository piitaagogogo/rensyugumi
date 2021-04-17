# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:01:39 2019

@author: ryo
"""

from pulp import *
import pandas as pd
import time
import sys
sys.setrecursionlimit(2500)     # recursionの最大値が1000だと足りないみたいなので引き上げ
# 編集用。データフレームの表示フォーマットについて。
# pd.set_option('display.max_rows',10)
# pd.set_option('display.max_columns',None)

start = time.time()

Needfile = "C:\\Users\\tai\\Desktop\\renshugumi\\219n.csv"
Attendfile = "C:\\Users\\tai\\Desktop\\renshugumi\\219A.csv"
K = 2 #被り人数の上限数
W = 4 #同時にする練習数

df_N = pd.read_csv(Needfile, encoding = 'shift-jis', header=None).dropna(how="all").dropna(how="all",axis=1)

training = len(df_N.index)-3
member = len(df_N.columns)-3
name_dic = {}
train_dic = {}

# 全練習の数までの連番リストと、全メンバーのリストを獲得。
# それぞれ、りょーさんコードでtraining、peopleに対応。
train_num_list = list(range(1,training+1))
member_num_list = list(range(1,member+1))
# Need由来の２。メンバー名と番号の対応辞書。namedic。
for i in range(member):
    name_dic[i+1] = df_N.iloc[1][i+3]
# Need由来の３。トレイニング名と番号の辞書。traindic
for i in range(training):
    train_dic[i+1] = df_N.iloc[i+3][1]
# Need由来の４。参加者データをnd配列にしたもの。N_np。
arr_N =  df_N.fillna(0).drop(df_N.index[[0,1,2]]).drop(df_N.columns[[0,1,2]],axis=1).values
# Attendファイルの参照
df_A = pd.read_csv(Attendfile, encoding = 'shift-jis', header=None).dropna(how="all").dropna(how="all",axis=1)
# 練習時間のコマ番号をリストとして獲得。times。
koma_list = list(range(len(df_A.columns)-3))
# Attend由来。出席データを配列化したもの。A_np。
arr_A = df_A.fillna(0).drop(df_A.index[[0,1,2]]).drop(df_A.columns[[0,1]],axis=1).values
#コマ名リスト。timezone
timezone_list = df_A.iloc[3:,1].values
# ここまでデータ整理

# ここからPulpによる最適化
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
print("練習数は"+str(len(training)))
# In[4]:結果
# #全期通して練習できた合計人数
if pulp.value(m.objective) >= 0:
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



