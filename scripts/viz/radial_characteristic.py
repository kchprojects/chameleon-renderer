import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import json

def polynomial4(x,a,b,c,d,e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


def polynomial8(x,a,b,c,d,e,f,g,h,i):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e + f * x**5 + g * x**6 + h * x**7 + i * x**8



X = [0,5,10,30,50,70,90]
X = [-x for x in X[::-1]] + X
Y = [1,0.985,0.975,0.86,0.6,0.2,0]
Y = Y[::-1] + Y

popt, pcov = curve_fit(polynomial8, np.array(X), np.array(Y))

xdata = np.array(range(0,91))
#print(xdata)

out = polynomial8(xdata,*popt)


with open("led_characteristic.json","w") as file:
    out_json = {
        "led_model": "GT-P6PRGB4303",
        "radial_attenuation":out.tolist()
    }
    json.dump(out_json,file,indent=4)

plt.plot(xdata,out)
plt.xticks(range(-90,91,10))
plt.show()

