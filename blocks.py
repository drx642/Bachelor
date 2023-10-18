import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        A = path.A[ncol,:] #aggregate savings
        B = path.B[ncol,:] #Bonds (aggregate)
        C = path.C[ncol,:] #Aggregate consumption
        C_N = path.C_N[ncol,:] #necessity consumption
        C_L = path.C_L[ncol,:] #luxury consumption
        clearing_A = path.clearing_A[ncol,:] #clearing bonds/savings market
        clearing_C = path.clearing_C[ncol,:] #clearing aggregate consumption market
        clearing_C_N = path.clearing_C_N[ncol,:] #clearing necessity market
        clearing_C_L = path.clearing_C_L[ncol,:] #clearing luxury market
        clearing_N = path.clearing_N[ncol,:] #clearing labour market
        d = path.d[ncol,:] #dividends
        d_N = path.d_N[ncol,:] #necessity dividends
        d_L = path.d_L[ncol,:] #luxury dividends
        G = path.G[ncol,:] #public spending - originally nothing
        i = path.i[ncol,:] #nominal interest rate
        N = path.N[ncol,:] #aggregate labour
        N_N = path.N_N[ncol,:] #labour in necessity sector
        N_L = path.N_L[ncol,:] #labour in luxury sector
        M_N = path.M_N[ncol,:] #materials used in necessity sector
        M_L = path.M_L[ncol,:] #materials used in luxury sector
        pm_L = path.pm_L[ncol,:] #price of materials in luxury sector
        pm_N = path.pm_N[ncol,:] #price of materials in necessity sector
        NKPC_res_N = path.NKPC_res_N[ncol,:] #New keynesian philips curve necessity (describes inflation)
        NKPC_res_L = path.NKPC_res_L[ncol,:] #New keynesian philips curve luxury (describes inflation)
        pi = path.pi[ncol,:] #full/average inflation
        pi_N = path.pi_N[ncol,:] #inflation in necessity sector
        pi_L = path.pi_L[ncol,:] #inflation in luxury sector
        adjcost = path.adjcost[ncol,:] #full price adjustment costs
        adjcost_N = path.adjcost_N[ncol,:] #price adjustment costs in necessity sector
        adjcost_L = path.adjcost_L[ncol,:] #price adjustment costs in luxury sector
        r = path.r[ncol,:] #interest rate
        istar = path.istar[ncol,:] #equilibrium nominal interest rate
        rstar = path.rstar[ncol,:] #equilibrium real interest rate? is not referenced anywhere OBS
        tau = path.tau[ncol,:] #tax rate
        w = path.w[ncol,:] #real wage
        w_N = path.w_N[ncol,:] #real wage in necessity sector
        w_L = path.w_L[ncol,:] #real wage in luxury sector
        mc_N = path.mc_N[ncol,:] #marginal costs necessity sector
        mc_L = path.mc_L[ncol,:] #marginal costs luxury sector
        Y = path.Y[ncol,:] #aggregate production
        Y_N = path.Y_N[ncol,:] #necessity production
        Y_L = path.Y_L[ncol,:] #Luxury production
        Y_star = path.Y_star[ncol,:] #equilibrium/potential production, used for Taylor rule
        Q = path.Q[ncol,:] #relative price between L and N, not defined?, OBS
        Q_check = path.Q_check[ncol,:] #defined as relative price between L and N, but not used? OBS
        p_N = path.p_N[ncol,:] #price of necessity good
        p_L = path.p_L[ncol,:] #price of luxury good
        P = path.P[ncol,:] #Standard CES price index
        Z_N = path.Z_N[ncol,:] #TFP, necessity sector
        Z_L = path.Z_L[ncol,:] #TFP, luxury sector
        A_hh = path.A_hh[ncol,:] #savings, households
        C_hh = path.C_hh[ncol,:] #Consumption households
        C_HAT_N_hh = path.C_HAT_N_hh[ncol,:] #net of subsistence consumption for necessity goods, households
        C_N_hh = path.C_N_hh[ncol,:] #consumption of necessity goods, households
        C_L_hh = path.C_L_hh[ncol,:] #consumption of luxury goods, households
        ELL_hh = path.ELL_hh[ncol,:] #Labor for households
        N_hh = path.N_hh[ncol,:] #Labour households
        P_hh = path.P_hh[ncol,:] #IPI for households
        tau_pm = path.tau_pm[ncol,:] #tax rate
        pm_f = path.pm_f[ncol,:] #tax rate
        M_test = path.M_test[ncol,:]

        #################
        # implied paths #
        #################

        # inflation
        #Q_lag = lag(ini.Q,Q)
        pi_L[:] = (Q/ss.Q)*(1+pi_N)-1 #Appendix A5, Central Bank block, eq 1
        #pi_L[:] = (Q/Q_lag)*(1+pi_N)-1
        
        # back out prices option 1 - inflation is multiplicative, standard solution goes wrong after 5 periods?

        #p_N_lag = lag(ini.p_N,p_N)
        #p_N[:] = (pi_N+1)*p_N_lag
        #p_L[:] = Q*p_N


        # back out prices option 2 - inflation is multiplicative, and needs to be looped?              
        #for t in range(par.T):
        #
        #    # i. lag
        #    p_N_lag = p_N[t-1] if t > 0 else ss.p_N
        #    p_N[t] = (pi_N[t]+1)*p_N_lag
        #    p_L[t] = Q[t]*p_N[t]
        
                
        # back out prices option 3 - inflation is just relative to steady state
        tau_pm[:]=0
        tau_pm[2]=par.tax_rate_base*pm_N[2]
        tau_pm[3]=par.tax_rate_base*pm_N[3]
        pm_f[:]=pm_N-tau_pm
        #if np.any(pm_N==0.01*ss.Q):
        #    tau_pm[2]=par.tax_rate_base*pm_N[2]
        #else:
        #    tau_pm[2]=0
        #elif pm_N==0.01*ss.Q:
        #    tau_pm[:]=par.tax_rate_base*pm_N
        #else:
        #    tau_pm[:]=0  
        p_N[:] = (1+pi_N)*ss.p_N #price equals ss price times inflation (same for necessity and luxury)
        p_L[:] = (1+pi_L)*ss.p_L # - What goes wrong here? Model still runs fine, but not as it should
        #p_L[:] = Q*p_N
        Q_check[:] = p_L/p_N #definition of Q, (section 2.2)
        
        #Inflation option one
        #P_hh_lag = lag(ini.P_hh,P_hh)
        #pi[:] = P_hh/P_hh_lag-1 #preliminary inflation indexing - only works if inflation is as above
        
        #Inflation option two - standard, #net aggregate inflation, geometric mean (2.23)        
        pi[:] = (1+pi_N)**par.epsilon*(1+pi_L)**(1-par.epsilon)-1 #preliminary inflation indexing
        
        # prices
        P[:] = (par.alpha_hh*p_N**(1-par.gamma_hh)+(1-par.alpha_hh)*p_L**(1-par.gamma_hh))**(1/(1-par.gamma_hh)) # price index, appendix A5, firms block, eq 1, OBS not defined exactly the same?
        #P[:] = ((1+pi_N)**(1-par.gamma_hh)*par.alpha_hh+(1-par.alpha_hh)*Q**(1-par.gamma_hh))**(1/(1-par.gamma_hh)) # price index
        w_L[:] = (1/Q)*w_N # real wage rate, reverse of w_N definition (section 2.2)
        pm_L[:] = (1/Q)*pm_f # real raw material price, reverse of pm_N definition (section 2.2)

        # production
        mc_N[:] = ((1-par.alpha_N)*(w_N/Z_N)**(1-par.gamma_N)+par.alpha_N*pm_f**(1-par.gamma_N))**(1/(1-par.gamma_N)) # marginal cost sector N, Appendix A5, firm block eq 4 (2.18)
        mc_L[:] = ((1-par.alpha_L)*(w_L/Z_L)**(1-par.gamma_L)+par.alpha_L*pm_L**(1-par.gamma_L))**(1/(1-par.gamma_L)) # marginal cost sector L, Appendix A5, firm block eq 4 (2.18)

        Y_N[:] = N_N/((1-par.alpha_N)*(w_N/mc_N)**(-par.gamma_N)*Z_N**(par.gamma_N-1)) #Appendix A5, firm block, eq 6 (2.16), reverse definition
        Y_L[:] = N_L/((1-par.alpha_L)*(w_L/mc_L)**(-par.gamma_L)*Z_L**(par.gamma_L-1)) #Appendix A5, firm block, eq 6 (2.16), reverse definition

        adjcost_N[:] = Y_N*(par.mu_N/(par.mu_N-1))*(1/(2*par.kappa_N))*(np.log(1+pi_N))**2 # adjustment cost sector N, Appendix A5, firm block, eq 5 (2.19)
        adjcost_L[:] = Y_L*(par.mu_L/(par.mu_L-1))*(1/(2*par.kappa_L))*(np.log(1+pi_L))**2 # adjustment cost sector L, Appendix A5, firm block, eq 5 (2.19)

        Y[:] = (Y_N+Q*Y_L)*(p_N/P) # overall production, appendix A5, firm block, eq 9, p_N is a 1 in paper OBS
        Y_star[:] = (ss.Y_N+Q*ss.Y_L)*(p_N/P) # potential production, ss version of above

        M_N[:] = par.alpha_N*(pm_f/mc_N)**(-par.gamma_N)*Y_N # M_N demand, Appendix A5, firm block, eq 7 (2.17)
        M_L[:] = par.alpha_L*(pm_L/mc_L)**(-par.gamma_L)*Y_L # M_L demand, Appendix A5, firm block, eq 7 (2.17)

        d_N[:] = Y_N-w_N*N_N-pm_f*M_N-adjcost_N # dividends sector N, Appendix A5, firm block, eq 8 (2.21)
        d_L[:] = Y_L-w_L*N_L-pm_L*M_L-adjcost_L # dividends sector L, Appendix A5, firm block, eq 8 (2.21)

        # b. monetary policy
        i[:] = istar + par.phi*pi + par.phi_y*(Y-(Y_star)) # taylor rule, Appendix A5, Central bank block, eq 3 (2.22)
        i_lag = lag(ini.i,i) #lag variable of nominal interest rate
        r[:] = (1+i_lag)/(1+pi)-1 ## Fix these taylor rule weights, Appendix A5, Central bank block, eq 4
        #r[:] = i-pi # fisher equation

        #test
        M_test[:] = (1 + (((M_N - ss.M_N) / ss.M_N) - ((pm_f - ss.pm_N) / ss.pm_N))) * ss.M_N

        # c. government
        B[:] = ss.B #a constant amount of bonds, SS value
        tau[:] = r*B + par.chi+tau_pm*(M_N+M_L) #tax rate is cost of having sold the bonds (r*B) and the "survival payment" (chi), (section 2.3)
        G[:] = tau-r*B - par.chi #Government spending equals whats left of the tax income after above payments, (2.24 rewritten)
        
        # d. aggregates
        A[:] = ss.B #aggregate savings must always equal the ss value of bonds (as constant) (2.25)
        C_N[:] = Y_N-adjcost_N-pm_f*M_N #agg. N consumption equals production minus expenses of production/price adj, Appendix A5, firm block, eq 11 (2.27) why expenses included here? OBS
        C_L[:] = Y_L-adjcost_L-pm_L*M_L #agg. L consumption equals production minus expenses of production/price adj, Appendix A5, firm block, eq 11 (2.28) why expenses included here? OBS
        C[:] = (C_N + Q*C_L)*(p_N/P) #agg. consumption, Appendix A5, firm block, eq 9, seems like it is the same as for Y, OBS
        N[:] = N_N + N_L #agg. labour, appendix A5, firm block, eq 10

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in nb.prange(ncols):

        A = path.A[ncol,:]
        B = path.B[ncol,:]
        C = path.C[ncol,:]
        C_N = path.C_N[ncol,:]
        C_L = path.C_L[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_C = path.clearing_C[ncol,:]
        clearing_C_N = path.clearing_C_N[ncol,:]
        clearing_C_L = path.clearing_C_L[ncol,:]
        clearing_N = path.clearing_N[ncol,:]
        d = path.d[ncol,:]
        d_N = path.d_N[ncol,:]
        d_L = path.d_L[ncol,:]
        G = path.G[ncol,:]
        i = path.i[ncol,:]
        N = path.N[ncol,:]
        N_N = path.N_N[ncol,:]
        N_L = path.N_L[ncol,:]
        M_N = path.M_N[ncol,:]
        M_L = path.M_L[ncol,:]
        pm_L = path.pm_L[ncol,:]
        pm_N = path.pm_N[ncol,:]
        NKPC_res_N = path.NKPC_res_N[ncol,:]
        NKPC_res_L = path.NKPC_res_L[ncol,:]
        pi = path.pi[ncol,:]
        pi_N = path.pi_N[ncol,:]
        pi_L = path.pi_L[ncol,:]
        adjcost = path.adjcost[ncol,:]
        adjcost_N = path.adjcost_N[ncol,:]
        adjcost_L = path.adjcost_L[ncol,:]
        r = path.r[ncol,:]
        istar = path.istar[ncol,:]
        rstar = path.rstar[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        w_N = path.w_N[ncol,:]
        w_L = path.w_L[ncol,:]
        mc_N = path.mc_N[ncol,:]
        mc_L = path.mc_L[ncol,:]
        Y = path.Y[ncol,:]
        Y_N = path.Y_N[ncol,:]
        Y_L = path.Y_L[ncol,:]
        Y_star = path.Y_star[ncol,:]
        Q = path.Q[ncol,:]
        P = path.P[ncol,:]
        Z_N = path.Z_N[ncol,:]
        Z_L = path.Z_L[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        C_HAT_N_hh = path.C_HAT_N_hh[ncol,:]
        C_N_hh = path.C_N_hh[ncol,:]
        C_L_hh = path.C_L_hh[ncol,:]
        ELL_hh = path.ELL_hh[ncol,:]
        N_hh = path.N_hh[ncol,:]
        p_N = path.p_N[ncol,:]
        P_hh = path.P_hh[ncol,:]
        tau_pm = path.tau_pm[ncol,:]
        pm_f = path.pm_f[ncol,:]
        M_test = path.M_test[ncol,:]

        #################
        # check targets #
        #################

        # a. phillips curve
        r_plus = lead(r,ss.r) #lead variable for interest rate, if nothing preceding, then ss value? OBS
        pi_N_plus = lead(pi_N,ss.pi_N) #lead variable for necessity inflation
        pi_L_plus = lead(pi_L,ss.pi_L) #lead variable for luxury inflation
        Y_N_plus = lead(Y_N,ss.Y_N) #lead variable for necessity production
        Y_L_plus = lead(Y_L,ss.Y_L) #lead variable for luxury production

        NKPC_res_N[:] = par.kappa_N*(mc_N-1/par.mu_N) + Y_N_plus/Y_N*np.log(1+pi_N_plus)/(1+r_plus) - np.log(1+pi_N) #New Keynesian philips curve for necessity (2.20)
        NKPC_res_L[:] = par.kappa_L*(mc_L-1/par.mu_L) + Y_L_plus/Y_L*np.log(1+pi_L_plus)/(1+r_plus) - np.log(1+pi_L) #New Keynesian philips curve for luxury (2.20)

        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_C_N[:] = C_N-C_N_hh
        clearing_C_L[:] = C_L-C_L_hh
        clearing_N[:] = N-N_hh
        clearing_C[:] = C-C_hh