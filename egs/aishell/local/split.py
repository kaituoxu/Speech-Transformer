#coding:utf-8
import sys
tmp=[]
for line in sys.stdin:
    for word in line:
        if word != ' ':
            tmp.append(word)

split_sen=" ".join(tmp)
print(split_sen.strip())
