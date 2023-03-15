# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sympy as sp

def minSln(arr):
    sln = lambda x: (x[0]+5)**2 + (x[1]+8)**2 + (x[2]+7)**2 + 2*x[0]**2*(x[1]**2 + 2*x[2]**2)
    return sln(arr)

def gradientDirections(eq,start):
    ## STEP 2
    # Find the gradient matrix
    partials = sp.Matrix([[sp.diff(eq,x)], [sp.diff(eq,y)], [sp.diff(eq,z)]])
    # calculate dk and gk at the start vector
    d = sp.Matrix([p.subs({x:start[0],y:start[1],z:start[2]}) for p in partials])
    
    return partials, d   

def getAlpha(partials,start,gk):
    ## STEP 3
    # form Hessian matrix and solve at start vector
    H = sp.Matrix(
        [[sp.diff(partials[0],x),sp.diff(partials[0],y),sp.diff(partials[0],z) ],
         [sp.diff(partials[1],x),sp.diff(partials[1],y),sp.diff(partials[1],z) ],
         [sp.diff(partials[2],x),sp.diff(partials[2],y),sp.diff(partials[2],z) ]])
    h_dict = {x:start[0],y:start[1],z:start[2]}
    
    h = sp.Matrix(
        [[h.subs( h_dict) for h in H[0:3] ],
         [h.subs( h_dict) for h in H[3:6] ],
         [h.subs( h_dict) for h in H[6:9] ]])
    # calculate alpha step size
    num = np.array(gk.T*gk).astype(np.float64).item()
    den = np.array(gk.T*h*gk).astype(np.float64).item()
    alpha = num/den
    
    return alpha,h    

def steepestDescent(minfunc,start,k):
    """Minimizes a function by iteratively moving towards the opposite direction 
    of the gradient at a given point. Input the objective function f(x1,x2,...xn), 
    a 1xn initial input vector, and maximum number of iterations to output
    the final input vector and the resulting objective function and Hessian."""
   
    costfxn = {}
    for i in range(k):
        # get partial derivatives and negative gradient at current x
        partials, dk = gradientDirections(minfunc,start)
        gk = dk*-1
        # calculate the Hessian matrix to find alpha for iteration
        alpha,H = getAlpha(partials,start,gk)
        # calculate step to move in direction of steepest descent
        delta = alpha*np.array(gk).flatten()
        # check that the step size is larger than the stopping threshold
        if np.all(abs(delta) >= t_stop):
            start = start + delta
            fx = minSln(start)
        else:
            break
        # store value of the objective function at each iteration
        costfxn[i] = fx
    
    return start, costfxn, H

def plotIter(costfxn, x_final, H):
    data = np.array(list(costfxn.values())).astype(np.float64)
    df = pd.DataFrame(data,columns=['Objective Function'])
    df.plot(y='Objective Function',use_index=True,
            title='Minimization with Steepest Gradient Descent',
            xlabel='Iterations',ylabel='f(x)')
    
    print('\nMinimizing x vector: \n'+ str(x_final))
    print('\nObjective function value: ' + str(minSln(x_final)))
    print('\nFinal Hessian matrix: \n'+ str(np.array(H).astype(np.float64)))
    
if __name__ == '__main__':

    x,y,z = sp.symbols('x,y,z',real=True)
    # max number of iterations
    k = 100
    # initialize x vector
    start = np.array([1.0, 1.0, 1.0])
    # objective function to be minimized
    eq = (x+5)**2 + (y+8)**2 + (z+7)**2 + 2*x**2*(y**2 + 2*z**2)
    # set break point
    t_stop = 1e-6  
    # optimize using steepest gradient descent
    x_final, costfxn, H = steepestDescent(eq,start,k)
    # plot objective function value over time
    plotIter(costfxn, x_final, H)
    