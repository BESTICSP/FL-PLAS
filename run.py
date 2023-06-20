import os
import time
import re
import string
from matplotlib import pyplot as plt
# time.sleep(3600)
print("start")
defense_method = ['none','layering','fltrust','flame','rsa','krum','ndc']
maker = ['.', 'D', '+', 'v', 's', 'p', '*']
malicious_ratio=[0,0.2,0.4,0.6,0.8,0.9]
lr=[0.0067,0.0025,0.000015]
dataset=['mnist','cifar10','cifar100']
model=['lenet','mobilenet','resnet18']
cut=[3,80,59]
num_class=[10,10,100]
backdoor_type=['trigger','semantic','edge-case']
for i in range(3):
    for j in range(6):
        for k in range(7):
            os.system('python parameterBoard.py --load_premodel True --lr '+str(lr[i])+' --num_nets 100 --part_nets_per_round 30 --fl_round 200 --malicious_ratio '+str(malicious_ratio[j])+' --dataname '+str(dataset[i])+' --model '+str(model[i])+' --device cuda:0  --num_class '+str(num_class[i])+' --backdoor_type trigger --defense_method '+defense_method[k]+' --cut '+str(cut[i])+' --test False')

for i in range(1,3):
    for j in range(6):
        for k in range(7):
            os.system('python parameterBoard.py --load_premodel True --lr 0.0025 --num_nets 100 --part_nets_per_round 30 --fl_round 200 --malicious_ratio '+str(malicious_ratio[j])+' --dataname cifar10 --model mobilenet -- --device cuda:0  --num_class 10 --backdoor_type '+backdoor_type[i]+' --defense_method '+defense_method[k]+' --cut 80 --test False')

f=open("./ma.txt",encoding='utf-8')
ma=[[],[],[],[],[],[],[]]

s0=f.readline()
i=0
while s0:
    x1=re.findall(r"(?:^| )([+-]?\d{1,2}(?:\.\d+)?)(?= |$)",s0)
    ma[i/6].append(float(x1[0]))
    s0=f.readline()
    i=i+1
  
savet=['mnist.png','cifar10.png','cifar100.png','seman.png','edge.png']
for j in range(35):
    plt.plot(malicious_ratio,ma[j],maker=maker[j%7])
    if j%7==6:
        plt.savefig('./ma'+savet[j/7]+'.png')
        plt.clf()

f=open("./ba.txt",encoding='utf-8')
ba=[[],[],[],[],[],[],[]]

s0=f.readline()
i=0
while s0:
    x1=re.findall(r"(?:^| )([+-]?\d{1,2}(?:\.\d+)?)(?= |$)",s0)
    ba[i/6].append(float(x1[0]))
    s0=f.readline()
    i=i+1
for j in range(35):
    plt.plot(malicious_ratio,ba[j],maker=maker[j%7])
    if j%7==6:
        plt.savefig('./ba'+savet[j/7]+'.png')
        plt.clf()

print("over")
