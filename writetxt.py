import os
import sys

path ='/dataT1/Free/qinan/intestine'

#myList=os.listdir(path_img)
#print('myList type = ',type(myList))
#new_myList = myList #sorted(myList)
#print('new_myList  = ', new_myList)
#1/0
#with open("train.txt",'a',encoding='utf-8') as filetext:
i = 0

f = open('/dataT1/Free/qinan/intestine/data_path_list.txt','r')
train = open('/dataT1/Free/qinan/intestine/train_path_list.txt', 'w')
test = open('/dataT1/Free/qinan/intestine/test_path_list.txt', 'w')
while True:
    i += 1 
    lines = f.readline().strip()
    if i <= 104:
        train.write(lines+'\n')
    else:
        test.write(lines+'\n')
    if not lines:
        break

f.close()