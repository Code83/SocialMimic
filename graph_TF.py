from matplotlib import pyplot as plt
import math
import numpy as np



font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 22,
        }


def sigmoid1(gamma):     #convert to probability
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid1i(gamma):     
	gamma = -gamma
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid2(gamma):
	gamma /= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))
		
def sigmoid3(gamma):
	gamma /= 3
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))

def sigmoid4(gamma):
	gamma *= 2
	if gamma < 0:
		return 1 - 1/(1 + math.exp(gamma))
	else:
		return 1/(1 + math.exp(-gamma))


def Vfunction1(gamma):
	return abs(np.tanh(gamma))

def Vfunction2(gamma):
	val = (math.pi)**(0.5)
	val /= 2
	val *= gamma
	val = math.erf(val)
	return abs(val)

def Vfunction3(gamma):
	val = 1 + gamma*gamma
	val = math.sqrt(val)
	val = gamma/val
	return abs(val)

def Vfunction4(gamma):
	val=(math.pi/2)*gamma
	val=np.arctan(val)
	val=(2/math.pi)*val
	return abs(val)

x=np.arange(-6,7,0.01)
y1 = [Vfunction1(i) for i in x]
y2 = [Vfunction2(i) for i in x]
y3 = [Vfunction3(i) for i in x]
y4 = [Vfunction4(i) for i in x]

# y1 = [sigmoid1(i) for i in x]
# # y11 = [sigmoid1i(i) for i in x]
# y2 = [sigmoid2(i) for i in x]
# y3 = [sigmoid3(i) for i in x]
# y4 = [sigmoid4(i) for i in x]

# plt.axvline(x=0, color='k')
# plt.axhline(y=0, color='k')
plt.plot(x,y1,label = 'V1')
# plt.plot(x,y11,label = 'S1i')
plt.plot(x,y2,label = 'V2')
plt.plot(x,y3,label = 'V3')
plt.plot(x,y4,label = 'V4')
plt.xlabel('x',fontsize=20)
plt.ylabel('T(x)',fontsize=20)
plt.title("V-Shaped Transfer Functions",fontsize=20)
# plt.text(5,0.9,r'$ \frac{1}{1 + e^{-x}} $',fontdict=font)
# plt.text(-6,0.8,r'$ \frac{1}{1 + e^{x}} $',fontdict=font)

plt.legend()# loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=True, ncol=5
# plt.show()
plt.savefig('VShapedFunction.png', bbox_inches='tight')