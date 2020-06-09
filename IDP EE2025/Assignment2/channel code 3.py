#channel code 3
#similar to channel code 3 just changing g1 matrix and no of bits


import numpy as np
import matplotlib.pyplot as plt
MSS=np.load('mss.npy') 			

b = np.reshape(MSS, (1,np.product(MSS.shape)))[0] 


# CHANNEL CODING

G1=[[1,0,0,0,0,1,1,1,1,0,1,0],[0,1,0,0,1,0,1,1,0,1,1,0],[0,0,1,0,1,1,1,0,1,1,1,1],[0,0,0,1,0,0,0,1,1,1,1,1]]



x1=np.zeros(12)
c = np.zeros(360000)
for i in range(0,30000):
	for j in range(12*i,12*(i+1)):
		q=[]
		for k in range (0,4):
			q.append(b[(4*i)+k])
		p=np.asarray(q)
		x1=np.matmul(np.transpose(G1),np.transpose(p))
		c[j]=x1[j%12]%2


					

	
	

T=(10)**(-6)
x=np.zeros(360000)   				
T_s=2*(10**(-8))					
f_c=2*(10**(6))						
f_s=50*(10**(6))					

 

for i in range (0,360000):
	if (c[i]==0):
		x[i]=1
	else:
		x[i]=(-1)


s_t=[]
for j in range (0,180000):		
	for n in range(50*(j),50*(j+1)):
			s_t.append(x[2*j]*(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))+(x[(2*j)+1]*(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))))



l=0
for k in range(0,9000000):
	l=l+((s_t[k])**2)
print(l) 
m=(l/18000000)*(T)*3
print('The energy per information bit is :',m)




k=int(input('enter the value of E_b/N_o:'))
v=m/2
k_1=float(k/10)
sigma=np.sqrt((v)*(f_s)/((10)**(k_1)))


w=np.random.normal(0,sigma,9000000) 


r=s_t+w


s1=np.zeros(9000000)
s2=np.zeros(9000000)
s3=np.zeros(9000000)
s4=np.zeros(9000000)
u1=np.zeros(180000)
u2=np.zeros(180000)
u3=np.zeros(180000)
u4=np.zeros(180000)

for e in range(0,180000):
	for n in range(50*(e),50*(e+1)):
		s1[n]=((np.cos(2*(np.pi)*(f_c)*(n*(T_s))))+(np.sin(2*(np.pi)*(f_c)*(n*(T_s)))))
		s2[n]=(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))-(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))
		s3[n]=-(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))+(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))
		s4[n]=-(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))-(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))
		u1[e]=u1[e]+(r[n]-s1[n])**2
		u2[e]=u2[e]+(r[n]-s2[n])**2
		u3[e]=u3[e]+(r[n]-s3[n])**2
		u4[e]=u4[e]+(r[n]-s4[n])**2

y_1=[]
for o in range(0,180000):
	y=[u1[o],u2[o],u3[o],u4[o]]
	y_1.append(min(y))


# Codes u1,u2,u3,u4 to 1,2,3,4 and stores the values in y_2
y_2=[]
for h in range(0,180000):
	if (y_1[h]== u1[h]):
		y_2.append(1)
	elif (y_1[h]==u2[h]):
		y_2.append(2)
	elif (y_1[h]==u3[h]):
		y_2.append(3)
	elif (y_1[h]==u4[h]):
		y_2.append(4)
c1=np.zeros(360000)

for p in range(0,180000):
	if (y_2[p]==1):
		c1[2*p]=0
		c1[2*(p)+1]=0
	elif (y_2[p]==2):
		c1[2*p]=0
		c1[2*(p)+1]=1
	elif (y_2[p]==3):
		c1[2*p]=1
		c1[2*(p)+1]=0
	elif (y_2[p]==4):
		c1[2*p]=1
		c1[2*(p)+1]=1


M1=[0,0,0,0]
M2=[0,0,0,1]
M3=[0,0,1,0]
M4=[0,0,1,1]
M5=[0,1,0,0]
M6=[0,1,0,1]
M7=[0,1,1,0]
M8=[0,1,1,1]
M9=[1,0,0,0]
M10=[1,0,0,1]
M11=[1,0,1,0]
M12=[1,0,1,1]
M13=[1,1,0,0]
M14=[1,1,0,1]
M15=[1,1,1,0]
M16=[1,1,1,1]
M=[M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16]
l1=np.matmul(np.transpose(G1),np.transpose(M1))
l2=np.matmul(np.transpose(G1),np.transpose(M2))
l3=np.matmul(np.transpose(G1),np.transpose(M3))
l4=np.matmul(np.transpose(G1),np.transpose(M4))
l5=np.matmul(np.transpose(G1),np.transpose(M5))
l6=np.matmul(np.transpose(G1),np.transpose(M6))
l7=np.matmul(np.transpose(G1),np.transpose(M7))
l8=np.matmul(np.transpose(G1),np.transpose(M8))
l9=np.matmul(np.transpose(G1),np.transpose(M9))
l10=np.matmul(np.transpose(G1),np.transpose(M10))
l11=np.matmul(np.transpose(G1),np.transpose(M11))
l12=np.matmul(np.transpose(G1),np.transpose(M12))
l13=np.matmul(np.transpose(G1),np.transpose(M13))
l14=np.matmul(np.transpose(G1),np.transpose(M14))
l15=np.matmul(np.transpose(G1),np.transpose(M15))
l16=np.matmul(np.transpose(G1),np.transpose(M16))

p1=[]
p2=[]
p3=[]
p4=[]
p5=[]
p6=[]
p7=[]
p8=[]
p9=[]
p10=[]
p11=[]
p12=[]
p13=[]
p14=[]
p15=[]
p16=[]
for i in range(0,12):
	p1.append(l1[i]%2)
	p2.append(l2[i]%2)
	p3.append(l3[i]%2)
	p4.append(l4[i]%2)
	p5.append(l5[i]%2)
	p6.append(l6[i]%2)
	p7.append(l7[i]%2)
	p8.append(l8[i]%2)
	p9.append(l9[i]%2)
	p10.append(l10[i]%2)
	p11.append(l11[i]%2)
	p12.append(l12[i]%2)
	p13.append(l13[i]%2)
	p14.append(l14[i]%2)
	p15.append(l15[i]%2)
	p16.append(l16[i]%2)




def hamm(a,b):
	rr=[0,0,0,0,0,0,0,0,0,0,0,0]
	
	for i in range(0,12):
		if(a[i]==b[i]):
			rr[i]=0
		else:
			rr[i]=1
	return rr.count(1)
	
p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]


g=np.zeros(30000)
for j in range(0,30000):
	c3=[]
	ham=[]
	for i in range (0,16):
		
		for k  in range (12*j,12*(j+1)):
			
			c3.append(c1[k])
		
	        
	
		ham.append(hamm(c3,p[i]))
	g[j]=ham.index(min(ham))
	
print(ham)
new=g

ox=[]
for i in range(0,30000):	
	if (new[i]==0):
		ox.append(M[0])
	elif (new[i]==1):
		ox.append(M[1])
	elif (new[i]==2):
		ox.append(M[2])
	elif (new[i]==3):
		ox.append(M[3])
	elif (new[i]==4):
		ox.append(M[4])
	elif (new[i]==5):
		ox.append(M[5])
	elif (new[i]==6):
		ox.append(M[6])
	elif (new[i]==7):
		ox.append(M[7])
	elif (new[i]==8):
		ox.append(M[8])
	elif (new[i]==9):
		ox.append(M[9])
	elif (new[i]==10):
		ox.append(M[10])
	elif (new[i]==11):
		ox.append(M[11])
	elif (new[i]==12):
		ox.append(M[12])
	elif (new[i]==13):
		ox.append(M[13])
	elif (new[i]==14):
		ox.append(M[14])
	elif (new[i]==15):
		ox.append(M[15])
	
		
ox1=np.concatenate(ox)


d = ox1.reshape(400,300) 


z_1=np.zeros(120000)
for q in range(0,120000):
	if (ox1[q]==b[q]):
		z_1[q]=0
	else :
		z_1[q]=1
k_1=np.count_nonzero(z_1)
k=0
for i in range(0,120000):
	k=k+z_1[i]
print('The number of error bits is:',k_1)
print('The bit error rate is:',k/120000)
		
plt.imshow(d,'gray')
plt.show()

