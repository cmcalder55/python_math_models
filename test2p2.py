# -*- coding: utf-8 -*-


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import optimize as opt

def plotContour(fx,pts):
    # make contour plot of x1 vs. x2 from vertices in feasible set
    d = np.linspace(0.2, 1, 300)

    x1, x2 = np.meshgrid(d, d)
    f = (x1-1)**2 + 2*(x2-2)**2

    fig, ax = plt.subplots(1, 1)
    contours = plt.contour(x1, x2, f)
    
    # add a colorbar and labels
    ax.set_title('f(x1,x2) Contour Plot and Feasible Region')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    extent=(x1.min(), x1.max(), x2.min(), x2.max())
    
    plt.imshow(((x1+x2 >= 0) & (1-x1**2-x2**2>=0)),interpolation='nearest',
               extent=extent, origin="lower", cmap="Greys",alpha=0.3)
    
    plt.clabel(contours, inline=1, fontsize=10)

    # plt.imshow(((x1+x2 >= 0) & (minfunc>=0)),
    # interpolation='nearest',extent=extent, origin="lower", cmap="Greys",alpha=0.3)
    
    plt.plot(*zip(*pts),marker='^')


def getPoints(r,arr,pts,t_stop,cons):
    while r >= t_stop:
        
        c=lambda x:(x[0]-1)**2 + 2*(x[1]-2)**2-r*np.log(1-x[0]**2-x[1]**2)-r*np.log(x[0]+x[1])
        
        res = opt.minimize(c, arr, method='SLSQP',constraints=cons)
        
        arr=res.x
        pts.append(tuple(arr))
        
        r=r/2
    # print(r)
    return pts

if __name__ == '__main__':

    x, y = sp.symbols('x,y', real=True)
    fx = [(x-1)**2, 2*(y-2)**2]
    hx = [1-x**2-y**2, x+y]

    h1= lambda x,y: 1-x**2-y**2
    h2= lambda x,y: x+y
    f= lambda x: (x[0]-1)**2 + 2*(x[1]-2)**2
    
    cons = ({'type': 'ineq', 'fun': lambda x: 1-x[0]**2-x[1]**2},
            {'type': 'ineq', 'fun': lambda x: x[0]+x[1]})
    
    r = 1
    t_stop = 0.002
    arr = np.array([0.5,0.5])
    pts = [(0.5,0.5)]

    p = getPoints(r,arr,pts,t_stop,cons)

    plotContour(fx,p)
    
        
        