# encoding: utf8
"""split dataset for link prediction mission"""
import random
pers = [0.5]
name = 'Twitter'
dataset = name+'/multiplex.edges'
f = open(dataset,'rb')
edges = [i for i in f]

for p in pers:
	random.shuffle(edges)
	selected = edges[: int(len(edges)*p)]
	remain = edges[int(len(edges)*p):]

	fw1 = open(name+'/train'+str(p)+'.txt','wb')
	fw2 = open(name+'/test'+str(p)+'.txt','wb')

	for i in selected:
		fw1.write(i)
	for i in remain:
		fw2.write(i)