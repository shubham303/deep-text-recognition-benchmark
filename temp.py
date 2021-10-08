class A:
	def a(self, i):
		i.j=10
	
class B :
	j =10
	
s=A()
b=B()
b.j=100
print(b.j)
s.a(b)
print(b.j)