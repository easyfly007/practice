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
