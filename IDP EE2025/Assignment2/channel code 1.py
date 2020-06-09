#channel code 1
#This code prints MSS, Energy per information bit, no.of error bits, Bit error rate and Final Image of MSS for  different values of (E_b/N_o) in dB 
#In this we took variance of AWGN channel as N_o/2
#In this code We considered E_b as (T/2)*(n/k)

import numpy as np
import matplotlib.pyplot as plt
MSS=np.load('mss.npy') 			

b = np.reshape(MSS, (1,np.product(MSS.shape)))[0] 


# CHANNEL CODING(Channel Code 1)

G1=[[1,1,1,1,0,0,0,0],[1,1,0,0,1,1,0,0],[1,0,1,0,1,0,1,0],[0,1,1,0,1,0,0,1]] #Initialising of G1 Matrix 


x1=np.zeros(8) # Gives G1(Transpose)*m(k)
c = np.zeros(240000) 
# This for loop calculates G1(Tranpose)*M(k) mod 2 and gives a array of 240000 elements

for i in range(0,30000):
	for j in range(8*i,8*(i+1)):
		q=[]
		for k in range (0,4):
			q.append(b[(4*i)+k])
		p=np.asarray(q)
		x1=np.matmul(np.transpose(G1),np.transpose(p))
		c[j]=x1[j%8]%2
					

T=(10)**(-6)				 
x=np.zeros(240000)   		# Creates a row matrix with 240000 elements as zeroes
T_s=2*(10**(-8))			# Defining Time period of signal
f_c=2*(10**(6))				# Defining Frequency of the signal
f_s=50*(10**(6))			# Defining Sampling frequency

# This for loop encodes bits into constellations 

for i in range (0,240000):
	if (c[i]==0):
		x[i]=1
	else:
		x[i]=(-1)



# Modulation
# Discrete time model

s_t=[]				# Creates a waveform to transmit
for j in range (0,120000):		
	for n in range(50*(j),50*(j+1)):
			s_t.append(x[2*j]*(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))+(x[(2*j)+1]*(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))))





# Energy Calculation per informtion bit
l=0
for k in range(0,6000000):
	l=l+((s_t[k])**2)
print(l) 
m=(l/12000000)*(T)*2

print('The energy per information bit is :',m)




# Calculating power spectral Density of Gaussian Random process from Obtained E_b
k=int(input('enter the value of E_b/N_o:'))
v=m/2
k_1=float(k/10)
sigma=np.sqrt((v)*(f_s)/((10)**(k_1)))




# Creates Discrete AWGN channel with mean:0 and variance:N_o/2
w=np.random.normal(0,sigma,6000000) 



# Recieved wave form Through AWGN channel
r=s_t+w



# Demodulation by minimum distance decoding
# Creating four signals s1,s2,s3,s4 as constellation in 4-QAM modulation scheme
s1=np.zeros(6000000)
s2=np.zeros(6000000)
s3=np.zeros(6000000)
s4=np.zeros(6000000)



# u1,u2,u3,u4 are distances of r from constellations s1,s2,s3,s4 respectively
u1=np.zeros(120000)
u2=np.zeros(120000)
u3=np.zeros(120000)
u4=np.zeros(120000)



# This for loop calculate distances u1,u2,u3,u4
# Calcualting distances of 50 samples from both s_i(i=1,2,3,4) and r and storing it in u1,u2,u3,u4 respectively
for e in range(0,120000):
	for n in range(50*(e),50*(e+1)):
		s1[n]=((np.cos(2*(np.pi)*(f_c)*(n*(T_s))))+(np.sin(2*(np.pi)*(f_c)*(n*(T_s)))))
		s2[n]=(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))-(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))
		s3[n]=-(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))+(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))
		s4[n]=-(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))-(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))
		u1[e]=u1[e]+(r[n]-s1[n])**2
		u2[e]=u2[e]+(r[n]-s2[n])**2
		u3[e]=u3[e]+(r[n]-s3[n])**2
		u4[e]=u4[e]+(r[n]-s4[n])**2
		
		
		
# Taking the minimum values from [u1[i],u2[i],u3[i],u4[i] ] for i from 0 to 5499
y_1=[]
for o in range(0,120000):
	y=[u1[o],u2[o],u3[o],u4[o]]
	y_1.append(min(y))



# Codes u1,u2,u3,u4 to 1,2,3,4 and stores the values in y_2
y_2=[]
for h in range(0,120000):
	if (y_1[h]== u1[h]):
		y_2.append(1)
	elif (y_1[h]==u2[h]):
		y_2.append(2)
	elif (y_1[h]==u3[h]):
		y_2.append(3)
	elif (y_1[h]==u4[h]):
		y_2.append(4)
c1=np.zeros(240000)




# assigning bits to corresponding constellation points and stores it in array c1
for p in range(0,120000):
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



#Channel Decoding

#creates 16 possible M matrices
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

#listing the matrices
M=[M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16]

#16 possible matrix multiplication with 16 possible M's
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



#dividing l's with mod 2
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
for i in range(0,8):
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



#Defining a function which calculates the hamming distance between two 8x1 matrices
def hamm(a,b):
	rr=[0,0,0,0,0,0,0,0]
	
	for i in range(0,8):
		if(a[i]==b[i]):
			rr[i]=0
		else:
			rr[i]=1
	return rr.count(1)


#listing finally obtained p's	
p=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16]



#calculates hamming distances and stores its incides for which the index is minimum
g=np.zeros(30000)
for j in range(0,30000):
	c3=[]
	ham=[]
	for i in range (0,16):
		
		for k  in range (8*j,8*(j+1)):
			
			c3.append(c1[k])
		
	        
	
		ham.append(hamm(c3,p[i]))
	g[j]=ham.index(min(ham))
	

new=g


#assigning the minimum index to corresponding matrix M  for which it is minimum and stores all M's in a matrix
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

# Reshapes the array c to matrix(400,300)
d = ox1.reshape(400,300) 
#Calculates bit error rate and no.of error bits

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
	
# plots the final image				
plt.imshow(d,'gray')
plt.show()


