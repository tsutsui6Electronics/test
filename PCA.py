import numpy as np
import matplotlib.pyplot as plt


"""
x=np.array([[23.3,19.8,14.7,19.7,16.9,14.4],[5.24,-5.23,-7.95,11.7,-2.44,-2.14]])
"""

x=np.array([[20,43,57,94,56,67,42,87,99,14,27,74,100],[
	16,23,90,55,75,34,78,75,90,99,44,76,100],[
		63,45,25,63,66,42,66,24,62,45,23,67,84],[
			52,44,63,45,88,53,63,22,42,39,85,32,89]])

#x=np.array([[10,2,5],[2,1,6]])

matrix=np.cov(x)
#k=(x[0]-np.mean(x[0]))/np.sqrt(matrix[0,0])
#g=(x[1]-np.mean(x[1]))/np.sqrt(matrix[1,1])
#sx=np.array([k,g])


matrix=np.cov(x)
b=np.linalg.eig(matrix)
#print("分散共分散行列 {}".format(b))


#第1主成分得点
u=[]
for i in range(x.shape[0]):
	k=0
	for j in range(x.shape[0]):
		k+=np.abs(b[1][j,i])*(x[j]-np.mean(x[j]))	
	u.append(k)
		
u=np.array(u)

eig_sort=np.sort(b[0])[::-1]
print(eig_sort)
rate=eig_sort/np.sum(b[0])
print(rate)
eig_sort_index=np.argsort(b[0])[::-1]

"""
#第2主成分
u2=np.abs(b[1][0,0])*(x[0]-np.mean(x[0]))+np.abs(b[1][1,0])*(x[1]-np.mean(x[1]))
"""

sen=[str(i+1) for i in range(x.shape[1])]


N=100

#x_ax:0-
x_ax=3
y_ax=0
 

zero_point=np.linspace(-N,N,1000)

t=N/zero_point.shape[0]

plt.plot(zero_point,np.zeros_like(zero_point),'--',c='blue')
plt.plot(np.zeros_like(zero_point),zero_point,'--',c='blue')

plt.scatter(u[eig_sort_index[x_ax]],u[eig_sort_index[y_ax]])


for i,j in enumerate(sen):
	plt.text(u[eig_sort_index[x_ax],i]+t,u[eig_sort_index[y_ax],i]-t,j)
plt.xlabel("{}-component".format(x_ax+1))
plt.ylabel("{}-component".format(y_ax+1))
plt.grid()
plt.xlim(-N,N)
plt.ylim(-N,N)
plt.show()


"""
#寄与率の計算

if b[0][0]>b[0][1]:
	print(b[0][0]/np.sum(b[0]))
else:
	print(b[0][1]/np.sum(b[0]))
"""


"""
#因子負荷量の計算
#不偏分散の利用に注意!!ddof=a:len(x)-a:initialize=0
print((np.sqrt(b[0][1])*np.abs(b[1][0,1]))/np.std(x[0],ddof=1))
"""
#不偏分散
#print(np.sum((x[0]-np.mean(x[0]))**2)/(len(x[0])-1))
#co=np.corrcoef(x)
#p=np.linalg.eig(co)
#print(p)
#print(b)
#z1=b[1][0,0]*x[0]+b[1][1,1]*x[1]
#print(z1)
#print(p[1][1,1])
#z1=b[1][0]*x[0]
