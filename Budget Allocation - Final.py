import pandas as pd
import numpy as np

#############################
#Part 1: Notations Summary: #
############################# 
   
# b_i, b_j are the available budgets to countries i,j 
# c_i, c_j are the consumption goods capabilities to countries i,j
# x_i_s, y_i_s, x_j_s, y_j_s: are the starting hybrid (x) or conventional (y) capabilities of country i,j
# s_t_1 is the threshold beyond which there is an increased risk of conventional retaliation
# s_t is the conventional  escalation threshold
# res_d, res_a represents the resolve of the attacker as estimated by the defender (d) vs actual (a) 
# theta_i is the risk aversion of the defender
# p_h_i, p_h_j, p_c_i, p_c_j are the prices for each type of capability (hybrid - h, conventional -c) and for each country (i,j)

###################################
#Part 2: Functions and Parameters #
###################################

# 1. Define the necessary functions
#---------------------------------

#    1.1 Probability Function Calculation: 
#    See paper for the discussion about this function. It accounts for the respective relative capabilities of both attacker and defender
#    the lambda factor allows to allocate more weight to the differential of hybrid or conventional cability when it comes to probality of war.
#    Default Value for lambda is 0.5 (i.e. equal weights).
 
def prob_fct_def(s, k_lbda, s_0, x_i,x_j,y_i,y_j,pw):
    lbda = 1 - 1/(1+(np.exp(-k_lbda*(s-s_0)))) 
    pb = lbda * (x_j / (x_i+x_j))**(1-pw) + (1-lbda) * y_j / (y_i+y_j)
    return(pb)

#   1.2 Utility Function for Gains/Losses of attacker
#   Utility is defined here as the Consumption level + or - gains or losses.
#   res parameter is the resolve factor of the attacker:
#   - >1 means risk taker
#   - <1 means risk averse 

def at_utility(C_j, b_j, x, res):
    if x >= 0 and C_j >= 0:
        ut = C_j + res * x * b_j
    elif x < 0 and C_j >= 0:
        ut = C_j +  x * b_j
    else:
        ut = 0
    return (ut)

# 2. Parameters
#-------------

# 2.1 Logistic Function Parameters. 

s_t_1 = 0.50 # Risk of conventional retaliation threshold
s_t = 0.65 # Conventional Warfare Threshold

s_0 = (s_t_1 + s_t)/2   #Transition parameter for Lambda Function.
k_lbda_pj = 20      #Speed of transition for Lambda Function 

# 2.2 Budget (An arbirtrary monetary value)
b_i = 200
b_j = 200

# 2.3 Utility function Parameters as per our theoretical paper.

# 2.3.1 Resolve of the attacker

res_d = 1  # as guessed by the defender
res_a = 1  # true resolve of the attacker

# We originally allowed to account for error in the defender estimate of the
# attacker resolve, but this is not used in the results, hence the two values should
# always be the same as if it was one single resolve parameter that is set. 

# 2.3.2 Risk aversion of defender
theta_i = 2 # 1 risk averse / < 1 risk taker

#  2.4 Discretization parameter for the intensity, the step in the space will be calculated as 1/dst
dst_s = 100  # when dst_s = 100 --> Increments of 1% of intensity.

# 2.5 Prices of military capability

p_h_i = 0.5 # price of hybrid for player i 
p_c_i = 1 # price of conventional for player i

p_h_j = 0.5 # price of hybrid for player j
p_c_j = 1 # price of conventional for player j

# 2.6 Mu parameter for exponential in the calculation of P_i (see paper)

mu = 0.1

# 2.7 a and b parameters for the calculation of gain/ loss functions (see paper)

gain_fact = 0.5
loss_fact = 1

# 2.8 Initial Stock of capabilities before allocation of budget. 

x_i_s = 10
y_i_s = 15
x_j_s = 15
y_j_s = 30

####################
#Part 3: The Model #
####################

# 3.1 Create the Vectors that are necessary to construct the optimization space. 
# Note from coder --> The most superfine discretization to cover the whole budget requires too much computing power.  

vector_superfine = np.linspace(0, 14, num = 15)
vector_fine = np.linspace(16,30, num = 8)
vector_coarse = np.linspace(40,200, num = 17)
vector_full = np.concatenate((vector_superfine,vector_fine, vector_coarse),axis = 0)

x_i = vector_full + x_i_s # --> Total Hybrid Capability defender, i.e. Initial Stock + allocation
y_i = vector_full + y_i_s # --> Total Conventional Capability defender, i.e. Initial Stock + allocation
x_j = vector_full + x_j_s # --> Hybrid Capability attacker, i.e. Initial Stock + allocation
y_j = vector_full + y_j_s # --> Conventional Capability attacker, i.e. Initial Stock + allocation

S = np.arange(start=1/dst_s, stop = 1+1/dst_s, step = 1/dst_s) # --> Vector of intensity levels (0-1)

p_j = np.zeros((len(S), len(x_i),len(y_i),len(x_j), len(y_j)),dtype = float) # --> Probability of a successful attack (Attacker perspective)
p_i = np.zeros((len(S), len(x_i),len(y_i),len(x_j), len(y_j)),dtype = float) # --> Probability of an attack of given intensity by attacker (Defender perspective)
d_i_c = np.zeros((len(S), len(x_i),len(y_i),len(x_j), len(y_j)),dtype = float) # --> Damage vector to be applied on the defender's consumption

#This matrix will contain the utility of [C_i - x_i - y_i - D_i].

u_i = np.zeros((len(S), len(x_i), len(y_i), len(x_j), len(y_j)), dtype = float)

#This matrix contains the Integrated values of [C_i - x_i - y_i - D_i] over the intensity dimention

integ_ui = np.zeros((len(x_i), len(y_i), len(x_j), len(y_j)),dtype = float)

#This matrix contains the Integrated values of the attacker utilities of 
#gains and losses multiplied by their respective probabilities.

integ_pj = np.zeros((len(x_i), len(y_i), len(x_j), len(y_j)),dtype = float)

#This matrix congtains the maximum over the intensity dimension of the attacker 
#utilities of gains and losses by their respe65ctive probabilities

w_j = np.zeros((len(x_i), len(y_i), len(x_j), len(y_j)),dtype = float)

#Gain, Losses and Damage Vectors

g = gain_fact * S[:]**2 
l = loss_fact * -(S[:]**4)   
dmg = S**2

# pw is the exponent intervening in the Pj formula. Given the exponent changes with the level of intensity to optimize the speed
# this is pre-calculated here.

pw = np.exp(-5 * S[:])

# Utility of Gains and losses from defender's perspective

u_g_d = np.zeros((len(g), len(x_j), len(y_j)), dtype = float)
u_l_d = np.zeros((len(l), len(x_j), len(y_j)), dtype = float)

#Utility of Gains and losses from attacker's perspective

u_g_a = np.zeros((len(g), len(x_j), len(y_j)), dtype = float)
u_l_a = np.zeros((len(l), len(x_j), len(y_j)), dtype = float)

# expected utility function of the attacker

eu_j = np.zeros((len(S), len(x_i), len(y_i), len(x_j), len(y_j)), dtype = float)

# 3.2 Based on the Vectors above, evaluate every component of the calculation process. This is done iteratively
# By scanning the full space of possibilities.

for a in range(len(x_i)):
    print(a)
    for b in range(len(y_i)):
        for c in range(len(x_j)):
            for d in range(len(y_j)):
                for s in range (len(S)):
                    
                    C_j = (b_j - p_h_j * (x_j[c]-x_j_s) - p_c_j * (y_j[d]-y_j_s))
                                        
                    p_j[s,a,b,c,d] = prob_fct_def(S[s],k_lbda_pj,s_0,x_i[a],x_j[c],y_i[b],y_j[d],pw[s])
                    d_i_c[s,a,b,c,d] = dmg[s] * p_j[s,a,b,c,d]
                    u_g_d[s,c,d] = at_utility(C_j, b_j, g[s], res_d)
                    u_l_d[s,c,d] = at_utility(C_j, b_j, l[s], res_d)
                    u_g_a[s,c,d] = at_utility(C_j, b_j, g[s], res_a)
                    u_l_a[s,c,d] = at_utility(C_j, b_j, l[s], res_a)
                    eu_j[s,a,b,c,d] = p_j[s,a,b,c,d] * u_g_a[s,c,d] + (1-p_j[s,a,b,c,d]) * u_l_a[s,c,d]

                integ_pj[a,b,c,d] = np.sum((np.exp(mu*(p_j[:,a,b,c,d] * u_g_d[:,c,d] + (1-p_j[:,a,b,c,d]) * u_l_d[:,c,d])))) * 1/dst_s
                w_j[a,b,c,d] = max(eu_j[:,a,b,c,d])

for a in range(len(x_i)):
    print(a)
    for b in range(len(y_i)):
        for c in range(len(x_j)):
            for d in range(len(y_j)):
                for s in range (len(S)):                    
                    p_i[s,a,b,c,d] = np.exp(mu*((p_j[s,a,b,c,d] * u_g_d[s,c,d] + (1-p_j[s,a,b,c,d]) * u_l_d[s,c,d]))) / integ_pj[a,b,c,d]
                    u_i[s,a,b,c,d] = p_i[s,a,b,c,d] * (max((b_i - p_h_i * (x_i[a]-x_i_s) - p_c_i * (y_i[b]-y_i_s)),0) - max((b_i - p_h_i * (x_i[a]-x_i_s) - p_c_i * (y_i[b]-x_i_s)),0) * theta_i * d_i_c[s,a,b,c,d])
                integ_ui[a,b,c,d] = np.sum(u_i[:,a,b,c,d] * 1/dst_s)


########################
# Part 4: Maximization #
########################

# This section allows to locate the best responses for both the defender and the challenger.

max_for_given_i = np.zeros((len(x_i),len(y_i),2), dtype = float)
max_for_given_j = np.zeros((len(x_j),len(y_j),2), dtype = float)

for c in range(len(x_j)):
    print(c)
    for d in range(len(y_j)):
        print(d)
        mm_wi = pd.DataFrame(integ_ui[:,:,c,d])
        ind = np.unravel_index(np.argmax(mm_wi, axis = None), mm_wi.shape)
        max_for_given_j[c,d,0] = x_i[ind[0]]
        max_for_given_j[c,d,1] = y_i[ind[1]]


for a in range(len(x_i)):
    print(a)
    for b in range(len(y_i)):
        print(b)
        mm_wj = pd.DataFrame(w_j[a,b,:,:])
        ind = np.unravel_index(np.argmax(mm_wj, axis = None), mm_wj.shape)
        max_for_given_i[a,b,0] = x_j[ind[0]]
        max_for_given_i[a,b,1] = y_j[ind[1]]
        
# Once the best responses are known for each allocation set, we can calculate the nash-equilibrium

euclidian_old = 1000000000000

total_len = len(x_i) * len (y_i) * len(x_j) * len(y_j)

euclidian = np.zeros((total_len, 5), dtype = float)
counter = 0

for a in range(len(x_i)):
    for b in range(len(y_i)): 
        for c in range(len(x_j)):
            for d in range(len(y_j)):

                
                euclidian_new = ((x_i[a] - max_for_given_j[c,d,0])**2 + (y_i[b] - max_for_given_j[c,d,1])**2 + (x_j[c] - max_for_given_i[a,b,0])**2 + (y_j[d] - max_for_given_i[a,b,1])**2)**(1/2) 
                euclidian[counter,0] = euclidian_new
                euclidian[counter,1] = x_i[a]
                euclidian[counter,2] = y_i[b]
                euclidian[counter,3] = x_j[c]
                euclidian[counter,4] = y_j[d]
                counter = counter + 1
                
                if euclidian_new <= euclidian_old: 
                    print(euclidian_new)
                    print(a,b,c,d)
                    a_eq = a
                    b_eq = b
                    c_eq = c
                    d_eq = d
                    nash = [x_i[a], y_i[b], x_j[c], y_j[d]]
                    euclidian_old = euclidian_new


###############################                    
# Part 5: Diagnostic Section  #
###############################
            
# This section allows to find specific best responses if the flow is given.
# e.g. we know that the attacker will invest 5 units in hybrid and 0 in conventional
# and we want to know what is the defender best reaction to this specific move of the challenger.

# 1. Defender best reaction
# 1.1. Fix the challenger final levels (start stock + flow)

x_j_f = x_j_s + 5
y_j_f = y_j_s + 10

# 1.2. Find in the final levels which index to use
x_j_f_ind = np.where(x_j == x_j_f)
y_j_f_ind = np.where(y_j == y_j_f)

# 1.3. Check Best Response
defender_best_rep_h = max_for_given_j[x_j_f_ind,y_j_f_ind,0]
defender_best_rep_c = max_for_given_j[x_j_f_ind,y_j_f_ind,1]

def_b_rep_h_ind = np.where(x_i == defender_best_rep_h[0,0])
def_b_rep_c_ind = np.where(y_i == defender_best_rep_c[0,0])

# 1.4. Exports

x_i_nash_ind = np.where(x_i == nash[0])
y_i_nash_ind = np.where(y_i == nash[1])
x_j_nash_ind = np.where(x_j == nash[2])
y_j_nash_ind = np.where(y_j == nash[3])

nash_p_i = p_i[:,x_i_nash_ind, y_i_nash_ind, x_j_nash_ind, y_j_nash_ind]
diag_p_i = p_i[:, def_b_rep_h_ind, def_b_rep_c_ind, x_j_f_ind, y_j_f_ind]

# 2. Challenger best reaction
# 1.1. Fix the challenger final levels

x_i_f = x_j_s + 0
y_i_f = y_j_s + 5

# 1.2. Fix the challenger final levels
x_i_f_ind = np.where(x_i == x_i_f)
y_i_f_ind = np.where(y_i == y_i_f)

# 1.3. Fix the challenger final levels
challenger_best_rep_h = max_for_given_i[x_i_f_ind,y_i_f_ind,0]
challenter_best_rep_c = max_for_given_i[x_i_f_ind,y_i_f_ind,1]