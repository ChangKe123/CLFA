pos=[]
neg=[]
neu=[]
with open('aggregate_label.txt','r') as f:
	for line in f.readlines():
		arr=line.strip().split('\t')
		if(arr[1]=='0'):
			pos.append((arr[0],0))
		elif arr[1]=='1':
			neg.append((arr[0],1))
		else:
			neu.append((arr[0],2))
		

train=pos[0:int(0.8*len(pos))]+neg[0:int(0.8*len(neg))]+neu[0:int(0.8*len(neu))]
val=pos[int(0.8*len(pos)):int(0.9*len(pos))]+neg[int(0.8*len(neg)):int(0.9*len(neg))]+neu[int(0.8*len(neu)):int(0.9*len(neu))]
test=pos[int(0.9*len(pos)):int(1*len(pos))]+neg[int(0.9*len(neg)):int(1*len(neg))]+neu[int(0.9*len(neu)):int(1*len(neu))]

with open('train.txt','w') as w:
	for i in train:
		w.write(i[0]+'\t'+str(i[1])+'\n')
	w.close()
with open('valid.txt','w') as w:
	for i in val:
		w.write(i[0]+'\t'+str(i[1])+'\n')
	w.close()
with open('test.txt','w') as w:
	for i in test:
		w.write(i[0]+'\t'+str(i[1])+'\n')
	w.close()
