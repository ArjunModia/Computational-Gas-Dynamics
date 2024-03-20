# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:56:30 2024

@author: arjun-modia
"""

"""
The entry parameters must be the left values WL = (ρL, uL, pL), the
right values WR = (ρR, uR, pR), the specific heat ratio γ (assume it to be the same on
both sides of the initial discontinuity), and the sampling point x/t where the solution
is to be determined. The utility should produce the solution of the Riemann problem
WRS = (ρRS , uRS , pRS ) at the requested space-time location defined by x/t.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define global variable speeds
global speeds
speeds = ['SHL', 'STL', 'SHR', 'SHR']

# Define left and right state variables
rho_left, u_left, p_left = [8., 0., 480]          # Left State
rho_right, u_right, p_right = [1., 0., 1]        # Right State

# Define gamma and combinations used as constants
gamma = 5.0/3

g1 = 2 / (gamma + 1)
g2 = (gamma - 1) / (gamma + 1)
g3 = 2 / (gamma - 1)
g4 = (gamma - 1) / (2*gamma)
g5 = -(gamma + 1) / (2*gamma)
g6 = 1 / gamma
g7 = 2 * gamma / (gamma - 1)
g8 = gamma - 1.0
g9 = (gamma - 1.0) / 2.0

# Calculate speeds of sound on left and right sides
c_left = np.sqrt(gamma * p_left / rho_left)
c_right = np.sqrt(gamma * p_right / rho_right)

# Input values for p* calculation
p_old = 0
error = 1
tol = 10e-6
itr = 1

# Input values for s=x/t sampling
x0 = 0.0
t0 = 0.0
t = 0.04

# File paths for data and plots
file_path = "RSproblem3.txt"
plot1_location = "RhoUPproblem3.png"
plot2_location = "XTproblem3.png"

# **1. Pressure Algorithm (p*)**

def initialize_press(rho_left, u_left, p_left, c_left, rho_right, u_right, p_right, c_right):       # Initializes the pressure
    return (p_left * rho_right * c_right + p_right * rho_left * c_left + (u_left - u_right) * rho_left * c_left * rho_right * c_right) / (
                rho_left * c_left + rho_right * c_right)

p_star = initialize_press(rho_left, u_left, p_left, c_left, rho_right, u_right, p_right, c_right)
print(p_star)

def fun_p(p_star, u_k, rho_k, p_k, c_k):               # f(p) for the pressure calculation
    if p_star > p_k:
        f = (p_star - p_k) * ((g1 / rho_k) / (p_star + g2 * p_k)) ** 0.5
        print("f(p) =", f)
        return f
    else:
        f = g3 * c_k * ((p_star / p_k) ** g4 - 1)
        print("f(p) =", f)
        return f

def diff_fun_p(p_star, rho_k, p_k, c_k):               # f'(p) for the pressure calculation
    if p_star > p_k:
        df = (1 - (p_star - p_k) / 2 / (p_star + g2 * p_k)) * (g1 / rho_k / (p_star + g2 * p_k)) ** 0.5
        print("df/dp =", df)
        return df
    else:
        df = ((p_star / p_k) ** g5) / rho_k / c_k
        print("df/dp =", df)
        return df

# Main loop for pressure iteration
flag = 1
while error > tol:

    if p_star < 0 and flag == 1:    # Condition for vacuum pressure check -> set to 0.00001 for vacuum
        print("Error -ve pressure -> p* = 0.00001 ........SET")
        p_star = 0.00001
        flag = 0

    p_old = p_star
    print("Iteration [", itr, "] ----------------------------------------")

    fL = fun_p(p_star, u_left, rho_left, p_left, c_left)
    fR = fun_p(p_star, u_right, rho_right, p_right, c_right)
    d_fL = diff_fun_p(p_star, rho_left, p_left, c_left)
    d_fR = diff_fun_p(p_star, rho_right, p_right, c_right)

    p_star = p_old - (fL + fR + u_right - u_left) / (d_fL + d_fR)
    error = abs((p_star - p_old) / p_old)
    itr += 1

print("p* =", p_star)

# **2. Velocity in Star Region (u*)**

u_star = (u_left + u_right) / 2 + (fun_p(p_star, u_right, rho_right, p_right, c_right) - fun_p(p_star, u_left, rho_left, p_left, c_left)) / 2
print("u* =", u_star)

# **3. Sampling Algorithm**

# Define arrays to store results
rho_result, u_result, p_result, c_result = np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000)

# Define the x/t sampling
x = np.linspace(-0.4995, 0.4995, 1000)
s = (x - x0) / (t - t0)

def right_rarefaction(s, u_star, p_star, rho_right, u_right, p_right, c_right):

  rho_3 = rho_right*((p_star/p_right)**g6)
  c_3 = c_right*((p_star/p_right)**g4)

  S_HR = u_right + c_right
  S_TR = u_star + c_3
  
  speeds[3] = round(S_TR,3)
  speeds[2] = round(S_HR,3)
  
  if s >= S_HR:
    #print("Region -> 4  | ", rho_right, u_right, p_right, c_right)
    return rho_right, u_right, p_right, c_right
  else:
    if s < S_TR:
      #print("Region -> 3*  | ", rho_3, u_star, p_star, c_3)
      return rho_3, u_star, p_star, c_3
    else:
      rho_fan = rho_right*(g1 - (g2*(u_right-s)/c_right))**g3
      u_fan = g1*(-c_right + (u_right/g3) + s)
      p_fan = p_right*(g1 - (g2*(u_right-s)/c_right))**g7
      c_fan = c_right*((p_fan/p_right)**g4)     
      #print("Region -> fanR  | ", rho_fan, u_fan, p_fan, c_fan)
      return rho_fan, u_fan, p_fan, c_fan


def right_shock(s, u_star, p_star, rho_right, u_right, p_right, c_right):
  
  rho_3 = rho_right*( (p_star/p_right + g2)/( g2*p_star/p_right + 1) )
  c_3 = np.sqrt(gamma*p_star/rho_3)
  Ms = np.sqrt(g4 - g5*p_star/p_right)
  S_R = u_right + c_right*Ms

  speeds[2] = round(S_R,3)

  if s >= S_R:
    #print("Region -> 4  | ", rho_right, u_right, p_right, c_right)
    return rho_right, u_right, p_right, c_right
  else:
    #print("Region -> 3*  | ", rho_3, u_star, p_star, c_3)
    return rho_3, u_star, p_star, c_3

"""---"""

def left_rarefaction(s, u_star, p_star, rho_left, u_left, p_left, c_left):

  rho_2 = rho_left*((p_star/p_left)**g6)

  c_2 = c_left*((p_star/p_left)**g4)

  S_HL = u_left - c_left

  S_TL = u_star - c_2
  
  speeds[1] = round(S_TL,3)
  speeds[0] = round(S_HL,3)

  if s <= S_HL:
      #print("Region -> 1  | ", rho_left, u_left, p_left, c_left)
      return rho_left, u_left, p_left, c_left
  else:  
    if s > S_TL:
      #print("Region -> 2*  | ", rho_2, u_star, p_star, c_2)
      return rho_2, u_star, p_star, c_2
    else:
      # print("Region -> fanL")
      rho_fan = rho_left*(g1 + (g2*(u_left-s)/c_left))**g3
      u_fan = g1*(c_left + (u_left/g3) + s)  
      p_fan = p_left*(g1 + (g2*(u_left-s)/c_left))**g7
      c_fan = c_left*((p_fan/p_left)**g4)
      
      #print("Region -> fanL  | ", rho_fan, u_fan, p_fan, c_fan)
      return rho_fan, u_fan, p_fan, c_fan

def left_shock(s, u_star, p_star, rho_left, u_left, p_left, c_left):
  
  rho_2 = rho_left*( (p_star/p_left + g2)/( g2*p_star/p_left + 1) )
  c_2 = np.sqrt(gamma*p_star/rho_2)
  Ms = np.sqrt(g4 - g5*p_star/p_left)
  S_L = u_left - c_left*Ms
  
  speeds[0] = round(S_L,3)
  
  if s <= S_L:
    #print("Region -> 1  | ", rho_left, u_left, p_left, c_left)
    return rho_left, u_left, p_left, c_left
  else:
    #print("Region -> 2*  | ", rho_2, u_star, p_star, c_2)
    return rho_2, u_star, p_star, c_2
        
"""---"""


# Main loop for the sampling algorithm

rho_result, u_result, p_result, c_result = np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000)

for i in range(1000):
  #print(i)
  if s[i] <= u_star: # Left part of graph
    if p_star > p_left: # Left shock
      #print("left shock")
      rho_result[i], u_result[i], p_result[i], c_result[i] = left_shock(s[i], u_star, p_star, rho_left, u_left, p_left, c_left)
    else: # Left expansion fan
      #print("left_rarefaction")
      rho_result[i], u_result[i], p_result[i], c_result[i] = left_rarefaction(s[i], u_star, p_star, rho_left, u_left, p_left, c_left)
  else:  # Right part of graph
    if p_star > p_right: # Right shock
      #print("right_shock")
      rho_result[i], u_result[i], p_result[i], c_result[i] = right_shock(s[i], u_star, p_star, rho_right, u_right, p_right, c_right)
    else: # Right expansion fan
      #print("right_rarefaction")
      rho_result[i], u_result[i], p_result[i], c_result[i] = right_rarefaction(s[i], u_star, p_star, rho_right, u_right, p_right, c_right)

# # """# **4. Writing Results to Text File**"""
# # Combine the four arrays into tuples
# combined_values = np.column_stack((x, rho_result, u_result, p_result))
# # Write the values to the text file with the desired format
# np.savetxt(file_path, combined_values, fmt="%+0.5E", delimiter='\t', newline='\n',header='\t\t   x\t\t\t rho\t\t\t  ux\t\t\t   p')

# """# **5. Plotting the graphs**"""
# Creating rho, u, p plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))
fig.suptitle(r'$\rho,\ u,\ p\ vs\ x\ at\ t=$'+str(t))

# Plot the first subplot
ax1.plot(x, rho_left*(x<=x0) + rho_right*(x>x0), '--k', linewidth=0.9)
ax1.plot(x, rho_result, color='blue')
ax1.set_ylabel(r'$density\ (\rho)$')
ax1.set_xlim(-0.5, 0.5)
ax1.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
ax1.grid(True)

# Plot the second subplot
ax2.plot(x, u_left*(x<=x0) + u_right*(x>x0), '--k', linewidth=0.9)
ax2.plot(x, u_result,color='red')
ax2.set_ylabel(r'$velocity\ (u)$')
ax2.set_xlim(-0.5, 0.5)
ax2.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
ax2.grid(True)

# Plot the third subplot
ax3.plot(x, p_left*(x<=x0) + p_right*(x>x0), '--k', linewidth=0.9)
ax3.plot(x, p_result,color='green')
ax3.set_xlabel(r'$x\ location$')
ax3.set_ylabel(r'$pressure\ (p)$')
ax3.set_xlim(-0.5, 0.5)
ax3.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
ax3.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Saving the plot1
plt.gcf().set_size_inches(7.2,6.3)
plt.savefig(plot1_location, format='png')

# Show the plot1
plt.show()

# print("\n----------------------------------------\nConditions for the problem")

# # Creating x/t plots for different velocities
# fig2 = plt.figure()
# ax = plt.subplot(111)

# # Check and plot left side lines
# if type(speeds[1]) != np.float64:
#     print("Left shock")
#     # Plot speeds[0]
#     t_values = (x-x0) / float(speeds[0])
#     t_values = np.where(t_values > 0, t_values, np.nan)  # Filter t_values for t > 0
#     ax.plot(x, t_values, color='orange', label=f'Shock L = {speeds[0]}')
#     ax.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
# else:
#     print("Left fan")
#     # Plot speeds[0:1]
#     t_values1 = (x-x0) / float(speeds[0]) 
#     t_values2 = (x-x0) / float(speeds[1])
#     t_values1 = np.where(t_values1 > 0, t_values1, np.nan)  # Filter t_values1 for t > 0
#     t_values2 = np.where(t_values2 > 0, t_values2, np.nan)  # Filter t_values2 for t > 0
#     ax.plot(x, t_values1, color='purple', label=f'Fan HL = {speeds[0]}')
#     ax.plot(x, t_values2, color='darkcyan', label=f'Fan TL = {speeds[1]}')
#     ax.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])

# # Check and plot right side lines
# if type(speeds[3]) != np.float64:
#     print("Right shock")
#     # Plot speeds[2]
#     t_values = (x-x0) / float(speeds[2])
#     t_values = np.where(t_values > 0, t_values, np.nan)  # Filter t_values for t > 0
#     ax.plot(x, t_values, color='r', label=f'Shock R = {speeds[2]}')
#     ax.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
# else:
#     print("Right fan")
#     # Plot speeds[2:3]
#     t_values1 = (x-x0) / float(speeds[2])
#     t_values2 = (x-x0) / float(speeds[3])
#     t_values1 = np.where(t_values1 > 0, t_values1, np.nan)  # Filter t_values1 for t > 0
#     t_values2 = np.where(t_values2 > 0, t_values2, np.nan)  # Filter t_values2 for t > 0
#     ax.plot(x, t_values1, '-b', label=f'Fan HR = {speeds[2]}')
#     ax.plot(x, t_values2, '-g', label=f'Fan TR = {speeds[3]}')
#     ax.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])

# # Plot u* on x\t diagram
# if u_star <= 0.001:
#     t_u_star = np.array([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
#     x_star = np.zeros(len(t_u_star)) + x0
#     ax.plot(x_star, t_u_star, 'k--',label=f'CS = {round(u_star,2)}')
#     ax.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])
# else:
#     t_u_star = (x-x0) / float(u_star) 
#     t_u_star = np.where(t_u_star > 0, t_u_star, np.nan)  # Filter t_values for t > 0
#     ax.plot(x, t_u_star, 'k--',label=f'CS = {round(u_star,2)}')
#     ax.xaxis.set_ticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])

# # Add labels and title to x\t diagram
# plt.xlabel('x')
# plt.ylabel('t')
# plt.title('X/t diagram for different Speeds')

# # Display the legend for x\t diagram
# box = ax.get_position()
# ax.set_position([box.x0*0.9, (box.y0 - box.height * 0.1)*0.9,
#                   box.width, box.height * 0.9])
# legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
#                     fancybox=True, shadow=True, ncol=5,
#                     handlelength=0.8, handletextpad=0.3)

# # Set scientific notation for the y-axis for x\t diagram
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# # Display the grid
# plt.grid(True)

# # Saving the plot
# plt.savefig(plot2_location, format='png', bbox_inches='tight')

# # Show the plot
# plt.show()