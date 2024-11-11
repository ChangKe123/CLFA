import sys

flgs=['1']
args=sys.argv
f1=open(args[1],'r').read().strip().split('\n')
tr=open('true.txt','r').read().strip().split('\n')
f2=open(args[2],'r').read().strip().split('\n')
total=0
count=[0,0,0,0]
for p1,p2,t in zip(f1,f2,tr):
	if t in flgs:
		total+=1
		if(p2==t and p1==t):
			count[0]+=1
		elif p2==t and p1!=t:
			count[1]+=1
		elif p2!=t and p1==t:
			count[2]+=1
		else:
			count[3]+=1
ratio=['{:.4f}'.format(i/total) for i in count]
print(ratio)
print(total)
