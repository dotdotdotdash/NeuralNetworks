import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from math import sin, cos

g=9.8
length=0.5
m_cart=1.0
m_pend=0.1
time_step = 0.1
runs = 0
max_runs = 100

friction_cart=0.0005;
friction_pend=0.000002;

total_m=m_cart+m_pend
momt_pend=m_pend*length

#initialization
init_angle = 90.0
prev_angle = init_angle

def vel_model(curr_angle,prev_angle):
    return (prev_angle-curr_angle)/time_step

def acc_model(angle,position,t):
    global prev_angle
    dth2dt = g*sin(angle) + cos(angle)*(-m_pend*length*vel_model(angle,prev_angle)**2*sin(angle))/(length*(4/3-m_pend*cos(angle)**2)/total_m)
    prev_angle = angle
    return dth2dt

def run():
    solver = ode(acc_model)
    solver.set_integrator('dop853')
