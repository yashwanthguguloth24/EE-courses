#channel code 2
#This code prints MSS, Energy per information bit, no.of error bits, Bit error rate and Final Image of MSS for  different values of (E_b/N_o) in dB 
#In this we took variance of AWGN channel as N_o/2
#In this code We considered E_b as (T/2)*(n/k)

import numpy as np
import matplotlib.pyplot as plt
MSS=np.load('mss.npy') 			

b = np.reshape(MSS, (1,np.product(MSS.shape)))[0] 


# CHANNEL CODING(Channel Code 2)


c=[]
for i in range(0,120000):
	for k in range (0,3): # repetetion of each bit 3 times
		c.append(b[i])

	

T=(10)**(-6)				
T_s=2*(10**(-8))				# Defining Time period of signal		
f_c=2*(10**(6))					# Defining Frequency of the signal	
f_s=50*(10**(6))				# Defining Sampling frequency	

x=np.zeros(360000)				# Creates a row matrix with 240000 elements as zeroes 

# This for loop encodes bits into constellations 

for i in range (0,360000):
	if (c[i]==0):
		x[i]=1
	else:
		x[i]=(-1)



# Modulation
# Discrete time model

s_t=[]			# Creates a waveform to transmit
for j in range (0,180000):		
	for n in range(50*(j),50*(j+1)):
			s_t.append(x[2*j]*(np.cos(2*(np.pi)*(f_c)*(n*(T_s))))+(x[(2*j)+1]*(np.sin(2*(np.pi)*(f_c)*(n*(T_s))))))



# Energy Calculation per informtion bit
l=0
for k in range(0,9000000):
	l=l+((s_t[k])**2)
print(l) 
m=(l/18000000)*(T)*3
print('The energy per information bit is :',m)


# Calculating power spectral Density of Gaussian Random process from Obtained E_b


k=int(input('enter the value of E_b/N_o:'))
v=m/2
k_1=float(k/10)
sigma=np.sqrt((v)*(f_s)/((10)**(k_1)))

# Creates Discrete AWGN channel with mean:0 and variance:N_o/2
w=np.random.normal(0,sigma,9000000) 

# Recieved wave form Through AWGN channel
r=s_t+w


# Demodulation by minimum distance decoding
# Creating four signals s1,s2,s3,s4 as constellation in 4-QAM modulation scheme
s1=np.zeros(9000000)
s2=np.zeros(9000000)
s3=np.zeros(9000000)
s4=np.zeros(9000000)
# u1,u2,u3,u4 are distances of r from constellations s1,s2,s3,s4 respectively
u1=np.zeros(180000)
u2=np.zeros(180000)
u3=np.zeros(180000)
u4=np.zeros(180000)

# This for loop calculate distances u1,u2,u3,u4
# Calcualting distances of 50 samples from both s_i(i=1,2,3,4) and r and storing it in u1,u2,u3,u4 respectively

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
# Taking the minimum values from [u1[i],u2[i],u3[i],u4[i] ] for i from 0 to 5499

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

# assigning bits to corresponding constellation points and stores it in array c1

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
#Channel Decoding


c2=np.zeros(120000)
for i in range(0,120000):
	r=[]
	for k in range (3*i,3*(i+1)):
		r.append(c1[k])
	
	if(r.count(1)>r.count(0)): #Majority bit counter
		c2[i]=1
	else:
		c2[i]=0	
		 
# Reshapes the array c to matrix(400,300)
d = c2.reshape(400,300) 
#Calculates bit error rate and no.of error bits

z_1=np.zeros(120000)
for q in range(0,120000):
	if (c2[q]==b[q]):
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

