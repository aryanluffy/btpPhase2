from cProfile import label


dataFile=open('official-preprocessed.m2','r+')

lines=[]

for line in dataFile.readlines():
    # print(line)
    # print("XXXXXXXXXXXXX")
    lines.append(line)

dataFile.close()

labels={}

linesWithZeroLength=0
correct=0
wrong=0

for i in range(0,len(lines)-1):
    if len(lines[i])==0:
        linesWithZeroLength+=1
        continue
    if lines[i]=='\n':
        continue
    if lines[i][0]=='A':
        continue
    if lines[i][0]=='S':
        if lines[i+1]=='\n':
            labels[lines[i][2:]]=1
            correct+=1
        elif lines[i+1][0]=='A':
            labels[lines[i][2:]]=0
            wrong+=1
if lines[-1][0]=='S':
    labels[lines[-1]]=1
print(linesWithZeroLength)
print(len(labels))
print(correct)
print(wrong)
import json

with open("dict_to_json_textfile.txt", 'w') as fout:
    json_dumps_str = json.dumps(labels, indent=4)
    print(json_dumps_str, file=fout)