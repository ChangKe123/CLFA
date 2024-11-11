fl='test.txt.ocr'
with open(fl, 'r') as f:
	for line in f.readlines():
		d=eval(line)
		print(d[-1])
	f.close()
