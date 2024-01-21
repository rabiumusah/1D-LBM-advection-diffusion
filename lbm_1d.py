#########################################################################
# 1D advection-diffusion
# Author: Dr. Rabiu Musah
# Address: University for Development Studies, Tamale - Ghana
           Department of Physics
# NB: The code is not optimized
#########################################################################

from numpy import zeros, ones, sqrt, exp, power
import numpy as np

from matplotlib.pyplot import plot, show, legend

#defined parameters in the question
Diff_coeff = 5.0e-8; #Diffusion coefficient
vel_max = 0.15; #macroscopic fluid velocity

n     = 1000; #number of grids
t_max = 4000; #maximum time/iteration to be reached

c_s = 1/sqrt(3); #speed of sound
ma  = vel_max/c_s; #mach number
tau = Diff_coeff/(c_s*c_s); #relaxation time
beta = 1/(2*tau+1); #factor considering the effect of relaxation time

# define the exact solution
rho_act = lambda x: 1.0 +1.0/sqrt(2*n-2)*exp(-power(((x-n/2)/(n-1)), 2));

#initializing initial populations
rho = zeros((n));
for i in range(n):
  rho[i] = rho_act(i)

# defining equilibrium populations
feq = zeros((3,n));
feq[0,:] = 2*rho*(2 - sqrt(1 + ma*ma))/3;
feq[1,:] = rho*(( vel_max - c_s*c_s)/(2*c_s*c_s) + sqrt(1 + ma*ma))/3;
feq[2,:] = rho*((-vel_max - c_s*c_s)/(2*c_s*c_s) + sqrt(1 + ma*ma))/3;

#initial population is in equilibrium 
f = feq;

# t = time/iteration
for t in range(1,t_max):   
    # density kinetic equation
    rho = f[0,:] + f[1,:] + f[2,:];

    # defining equilibrium populations
    feq[0,:] = 2*rho*(2 - sqrt(1 + ma*ma))/3;
    feq[1,:] = rho*(( vel_max - c_s*c_s)/(2*c_s*c_s) + sqrt(1 + ma*ma))/3;
    feq[2,:] = rho*((-vel_max - c_s*c_s)/(2*c_s*c_s) + sqrt(1 + ma*ma))/3;
    
    # performing collision 
    f[0,:] = f[0,:] - 2*beta*(f[0,:] - feq[0,:]); 
    f[1,:] = f[1,:] - 2*beta*(f[1,:] - feq[1,:]);
    f[2,:] = f[2,:] - 2*beta*(f[2,:] - feq[2,:]);

    # periodic boundary conditions
    temp_left  = f[2, 0];
    temp_right = f[1,-1];
    
    #advection across direction of c
    for i in np.arange(n-1, 0, -1):
        f[1,i] = f[1,i-1];
    
    for i in np.arange(0, n):
        f[2,i] = f[2,i];
    
    f[1, 0] = temp_right;
    f[2,-1] = temp_left;
    

#postprocessing
x = np.linspace(0, n, n)
plot(x,rho_act(x),'k--',label='exact');
plot(x, rho,'k-', label='lbm');
legend()
show()
