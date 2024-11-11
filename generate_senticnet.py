wt=open('senticNet/senticnet_words.txt', 'w')
with open('senticnet-5.0/senticnet5.txt', 'r') as f:
	c=0
	dic=[]
	for line in f.readlines():
		c+=1
		if c==1:
			continue
		arr=line.strip().split('\t')
		dic.append((arr[0], arr[-1]))
	f.close()

	for d in dic:
		wt.write(d[0]+'\t'+str(d[1])+'\n')
	wt.close()
