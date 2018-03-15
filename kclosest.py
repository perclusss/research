#!/usr/bin/python
import math
def kclosest(place,k,locations):
	'''
	>>> kclosest((4,1),1,[('ngv',4,0),('town hall',4,4),('myhotel',2,2),('parliament',8,5.5),('fed square',4,2)])
	['ngv']
	>>> kclosest((4,1),2,[('ngv',4,0),('town hall',4,4),('myhotel',2,2),('parliament',8,5.5),('fed square',4,2)])
	['ngv', 'fed square']
	'''
	x,y=place
	distance = 9999
	myList=[]
	result=[]
	indexList=[]
	for location in locations:
		dist = math.sqrt(math.pow((location[1] - x),2) + math.pow((location[2] - y),2))
		myList.append(dist)
	indexList = sorted(range(len(myList)),key=lambda x:myList[x])	
	
	for index in indexList[:k]:
		result.append(locations[index][0])
	return result

		

if __name__ == "__main__":
	import doctest
	doctest.testmod()
