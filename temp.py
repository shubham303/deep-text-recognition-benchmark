class A:
	def a(self, i):
		i.j=10
	
class B :
	j =10
	
a=A()
print(isinstance(a, B))