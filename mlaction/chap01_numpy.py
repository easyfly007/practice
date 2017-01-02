import numpy
a=numpy.random.rand(4,5)
print(a)
b=numpy.mat(a)
print(b)
c=b.I
print(c)
d=numpy.eye(5)
print(d)
e=numpy.mat(d)
print(e)


size = a.shape
# get the shape of the numpy array or numpy mat, return tuple
# print(type(size))
# print(size)


print('###')
# tile: is to repeat some array by n
a=[[1,2,3],[4,5]]
b=numpy.tile(a, (1,3))
print(a)
print(b)
# what's the difference between numpy.tile and numpy.repeat?
# tile: 1,2,3,1,2,3
# repeat: 1,1,2,2,3,3
a=[1,2,3]
mm=d.repeat(5)
# print(mm)

# sum function
a=numpy.sum([1,2,3.2], dtype = numpy.int32)
print(a)