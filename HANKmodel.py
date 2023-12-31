
import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks

class HANKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']
        
        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['w_N','r','i','d_N','d_L','tau','p_N','p_L','Q','P'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','c_hat_N','c_N','c_L','ell','n','p','u','e'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['istar','Z_N','Z_L','pm_N'] # exogenous inputs
        self.unknowns = ['pi_N','Q','w_N','N_L','N_N'] # endogenous inputs
        self.targets = ['NKPC_res_N','NKPC_res_L','clearing_A','clearing_N','clearing_C_N'] # targets
        
        # d. all variables
        self.varlist = [ # all variables
            'A',
            'B',
            'C',
            'C_N',
            'C_L',
            'clearing_A',
            'clearing_C',
            'clearing_C_N',
            'clearing_C_L',
            'clearing_N',
            'd',
            'd_N',
            'd_L',
            'G',
            'i',
            'N',
            'N_N',
            'N_L',
            'M_N',
            'M_L',
            'pm_L',
            'pm_N',
            'NKPC_res_N',
            'NKPC_res_L',
            'p_N',
            'p_L',
            'pi',
            'pi_N',
            'pi_L',                        
            'adjcost',
            'adjcost_N',
            'adjcost_L',            
            'r',
            'istar',
            'rstar',
            'tau',
            'w',
            'w_N',
            'w_L',            
            'mc_N',
            'mc_L',
            'Y',
            'Y_N',
            'Y_L',
            'Y_star',
            'Q',
            'Q_check',
            'P',
            'Z_N',
            'Z_L',
            'tau_pm',
            'tax_rate_base',
            'pm_f',
            'M_test']

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        self.block_pre = blocks.block_pre
        self.block_post = blocks.block_post
        
    def setup(self):
        """ set baseline parameters """

        par = self.par

        par.Nfix               = 1                     # number of fixed discrete states (either work in L or N sector)
        par.Nz                 = 7                     # number of stochastic discrete states (here productivity)
        par.r_target_ss        = 1.02**(1/4)-1

        # a. preferences
        par.beta               = 0.9875                # discount factor (guess, calibrated in ss)
        par.varphi             = 1.0                   # disutility of labor
        par.alpha_hh           = 1/3                   # Net of subsistence weight on necessity goods
        par.gamma_hh           = 0.2                   # Elasticity of substitution
        par.c_bar              = 0.05           
        par.chi                = 0.05                  #Government transfer

        par.sigma              = 2.0                   # inverse of intertemporal elasticity of substitution
        par.nu                 = 2.0                   # inverse Frisch elasticity
        # c. income parameters     
        par.rho_z              = 0.9777                # AR(1) parameter
        par.sigma_psi          = 0.1928                # std. of psi

        #par.rho_z = 0.966                                     
        #par.sigma_psi = np.sqrt(0.50**2*(1-par.rho_z**2))      
    
        # d. price setting           
        par.alpha_L            = 0.31                    # cobb-douglas for sector L
        par.alpha_N            = 0.63                    # cobb-douglas for sector N
        par.gamma_L            = 0.815                    # substitution elasticity for sector L
        par.gamma_N            = 0.25                   # substitution elasticity for sector N
        par.mu_L               = 1.8                    # mark-up for sector L
        par.mu_N               = 1.2                    # mark-up for sector N
        par.kappa_L            = 0.04                   # price rigidity for sector L
        par.kappa_N            = 0.22                   # price rigidity for sector N
    
        # e. government
        par.phi                = 1.5                    # Taylor rule coefficient on inflation
        par.phi_y              = 0.0                    # Taylor rule coefficient on output
        par.epsilon            = 0.454                  # Taylor rule inflation weights

        par.tax_rate_base       =0.00                    #basic tax rate
              
        par.G_target_ss        = 0.0                    # government spending
        par.B_target_ss        = 5.6                    # bond supply 
     
        # f. grids               
        par.a_min              = 0.0                    # maximum point in grid for a
        par.a_max              = 100.0                  # maximum point in grid for a
        par.Na                 = 500                    # number of grid points
     
        # g. shocks      
        par.jump_Z_N           = 0.0                    # initial jump
        par.rho_Z_N            = 0.00                   # AR(1) coefficeint
        par.std_Z_N            = 0.00                   # std.

        par.jump_Z_L           = 0.0                    # initial jump
        par.rho_Z_L            = 0.00                   # AR(1) coefficeint
        par.std_Z_L            = 0.00                   # std.
     
        #par.jump_istar =       0.0025
        par.jump_istar         = 0.0
        par.rho_istar          = 0.61
        par.std_istar          = 0.0025
     
        #par.jump_pm_N          = 0.01
        par.jump_pm_N          = 0.0
        par.rho_pm_N           = 0.75
        par.std_pm_N           = 0.0025

        # h. misc.
        par.T                  = 500                    # length of path        
        par.max_iter_solve     = 50_000                 # maximum number of iterations when solving
        par.max_iter_simulate  = 50_000                 # maximum number of iterations when simulating
        par.max_iter_broyden   = 100                    # maximum number of iteration when solving eq. system
        par.tol_ss             = 1e-11                  # tolerance when finding steady state
        par.tol_solve          = 1e-11                  # tolerance when solving
        par.tol_simulate       = 1e-11                  # tolerance when simulating
        par.tol_broyden        = 1e-10                  # tolerance when solving eq. system

        
    def allocate(self):
        """ allocate model """

        par = self.par

        # b. solution
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss        