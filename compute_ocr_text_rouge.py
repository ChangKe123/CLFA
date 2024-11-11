from rouge import Rouge

rg=Rouge(metrics=['rouge-l'])

with open('datasets/sarcasm/train.txt' ,'r') as f:
	count=0
	nc=0
	cf=[0]*21
	for line in f.readlines():
		a=eval(line)
		count+=1
		if len(a[2].strip())==0:
			nc+=1
			continue
		score=rg.get_scores(a[1], a[2])['rouge-l']['f']
		cf[int(score/0.05)]+=1
		#print(score['rouge-l']['f'])
	#print(nc/count)
	for i, c in enumerate(cf):
		print('{:.2f} {:.2f} {:.4f}'.format(i/20,(i+1)/20, c/count))
	#print(cf)
	f.close()



