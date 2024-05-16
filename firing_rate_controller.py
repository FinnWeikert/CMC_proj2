"""Network controller"""

import numpy as np
from scipy.interpolate import CubicSpline
import scipy.stats as ss
import farms_pylog as pylog



class FiringRateController:
    """zebrafish controller"""

    def __init__(
            self,
            pars
    ):
        super().__init__()

        self.n_iterations = pars.n_iterations
        self.n_neurons = pars.n_neurons
        self.n_muscle_cells = pars.n_muscle_cells
        self.timestep = pars.timestep
        self.times = np.linspace(
            0,
            self.n_iterations *
            self.timestep,
            self.n_iterations)
        self.pars = pars

        self.n_eq = self.n_neurons*4 + self.n_muscle_cells*2 + self.n_neurons * \
            2  # number of equations: number of CPG eq+muscle cells eq+sensors eq
        self.muscle_l = 4*self.n_neurons + 2 * \
            np.arange(0, self.n_muscle_cells)  # muscle cells left indexes
        self.muscle_r = self.muscle_l+1  # muscle cells right indexes
        self.all_muscles = 4*self.n_neurons + \
            np.arange(0, 2*self.n_muscle_cells)  # all muscle cells indexes
        # vector of indexes for the CPG activity variables - modify this
        # according to your implementation
        self.all_v = range(self.n_neurons*2)
        self.rL = 2*np.arange(0,self.n_neurons)
        self.rR = self.rL + 1
        self.aL = 2*np.arange(self.n_neurons,self.n_neurons*2)
        self.aR = self.aL + 1
        self.sL = 2*(np.arange(self.n_neurons*2,self.n_neurons*3)+self.n_muscle_cells)
        self.sR = self.sL + 1
        #self.all_r = np.concatenate([self.rL,self.rR])
        #self.all_a = np.concatenate([self.aL,self.aR])
        self.all_r = np.arange(0, 2*self.n_neurons)
        self.all_a = 2*self.n_neurons + np.arange(0, 2*self.n_neurons)
        self.all_s = 4*self.n_neurons + 2*self.n_muscle_cells + np.arange(0, 2*self.n_neurons)

        # muscle cells parameters
        self.n_joints = self.pars.n_joints
        self.taum_a = self.pars.taum_a
        self.taum_d = self.pars.taum_d
        self.w_V2a2muscle = self.pars.w_V2a2muscle
        # conversion weight from muscle cell activity to muscle activations
        self.act_strength = self.pars.act_strength

        # CPG pars
        self.I = self.pars.I
        self.Idiff = self.pars.Idiff
        self.n_asc = self.pars.n_asc
        self.n_desc = self.pars.n_desc
        self.tau = self.pars.tau
        self.taua = self.pars.taua
        self.b = self.pars.b
        self.gamma = self.pars.gamma
        self.w_inh = self.pars.w_inh

        # stretch pars
        self.w_stretch = self.pars.w_stretch
        self.n_asc_str = self.pars.n_asc_str
        self.n_desc_str = self.pars.n_desc_str
        self.tau_str = self.pars.tau_str

        # noise pars
        self.noise_sigma = self.pars.noise_sigma

        # Added by Tristan
        self.gin = self.pars.gin
        self.gss = self.pars.gss
        self.gmc = self.pars.gmc
        self.rho = self.pars.rho

        self.state = np.zeros([self.n_iterations, self.n_eq])  # equation state
        self.dstate = np.zeros([self.n_eq])  # derivative state
        self.state[0] = np.random.rand(self.n_eq)  # set random initial state

        self.poses = np.array([
            0.007000000216066837,
            0.00800000037997961,
            0.008999999612569809,
            0.009999999776482582,
            0.010999999940395355,
            0.012000000104308128,
            0.013000000268220901,
            0.014000000432133675,
            0.014999999664723873,
            0.01600000075995922,
        ])  # active joint distances along the body (pos=0 is the tip of the head)
        self.poses_ext = np.linspace(
            self.poses[0], self.poses[-1], self.n_neurons)  # position of the sensors

        # to keep track of the active join angles (for Ex6 angles posotions plot)
        self.angle_poses = np.zeros((self.n_iterations, 10)) # added Finn (maybe other way makes more sense)

        # initialize ode solver
        self.f = self.ode_rhs

        # stepper function selection
        if self.pars.method == "euler":
            self.step = self.step_euler
        elif self.pars.method == "noise":
            self.step = self.step_euler_maruyama
            # vector of noise for the CPG voltage equations (2*n_neurons)
            self.noise_vec = np.zeros(self.n_neurons*2)

        # zero vector activations to make first and last joints passive
        # pre-computed zero activity for the first 4 joints
        self.zeros8 = np.zeros(8)
        # pre-computed zero activity for the tail joint
        self.zeros2 = np.zeros(2)

        def calculate_w(i, j, ndesc, nasc):
            if i <= j and j - i <= ndesc:
                return 1 / (j - i + 1)
            elif i > j and i - j <= nasc:
                return 1 / (i - j + 1)
            else:
                return 0
            
        def calculate_wcm(i, j, ncm):
            if ncm*i <= j and ncm*(i+1) - 1 >= j:
                return 1
            else:
                return 0

        self.Wsc = np.zeros((self.n_neurons,self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                self.Wsc[i,j] = calculate_w(i,j,self.n_desc_str,self.n_asc_str)

        self.Win = np.zeros((self.n_neurons,self.n_neurons))
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                self.Win[i,j] = calculate_w(i,j,self.n_desc,self.n_asc)

        self.Wcm = np.zeros((self.n_muscle_cells,self.n_neurons))
        for i in range(self.n_muscle_cells):
            for j in range(self.n_neurons):
                self.Wcm[i,j] = calculate_wcm(i,j,5) # ncm = 5

    def get_ou_noise_process_dw(self, timestep, x_prev, sigma):
        """
        Implement here the integration of the Ornstein-Uhlenbeck processes
        dx_t = -0.5*x_t*dt+sigma*dW_t
        Parameters
        ----------
        timestep: <float>
            Timestep
        x_prev: <np.array>
            Previous time step OU process
        sigma: <float>
            noise level
        Returns
        -------
        x_t{n+1}: <np.array>
            The solution x_t{n+1} of the Euler Maruyama scheme
            x_new = x_prev-0.1*x_prev*dt+sigma*sqrt(dt)*Wiener
        """

        dx_process = np.zeros_like(x_prev)

    def step_euler(self, iteration, time, timestep, pos=None):
        """Euler step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        d = 1

        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def step_euler_maruyama(self, iteration, time, timestep, pos=None):
        """Euler Maruyama step"""
        self.state[iteration+1, :] = self.state[iteration, :] + \
            timestep*self.f(time, self.state[iteration], pos=pos)
        self.noise_vec = self.get_ou_noise_process_dw(
            timestep, self.noise_vec, self.pars.noise_sigma)
        self.state[iteration+1, self.all_v] += self.noise_vec
        self.state[iteration+1,
                   self.all_muscles] = np.maximum(self.state[iteration+1,
                                                             self.all_muscles],
                                                  0)  # prevent from negative muscle activations
        return np.concatenate([
            self.zeros8,  # the first 4 passive joints
            self.motor_output(iteration),  # the active joints
            self.zeros2  # the last (tail) passive joint
        ])

    def motor_output(self, iteration):
        """
        Here you have to final muscle activations for the 10 active joints.
        It should return an array of 2*n_muscle_cells=20 elements,
        even indexes (0,2,4,...) = left muscle activations
        odd indexes (1,3,5,...) = right muscle activations
        """
        activation = self.act_strength*self.state[iteration][self.all_muscles]
        #return np.zeros(
        #    2 *
        #    self.n_muscle_cells)  # here you have to final active muscle equations for the 10 joints
        return activation

    def ode_rhs(self,  _time, state, pos=None):
        """Network_ODE
        You should implement here the right hand side of the system of equations
        Parameters
        ----------
        _time: <float>
            Time
        state: <np.array>
            ODE states at time _time
        Returns
        -------
        dstate: <np.array>
            Returns derivative of state
        """

        # Implement (11) here QUESTION: did they forget Idiff in (11)
        # coupling for rL  (Transpose is wierd...)
        xL = self.I + self.Idiff- self.b*state[self.aL] - self.w_inh*np.dot(self.Win.T, state[self.rR]) \
             - self.w_stretch*np.dot(self.Wsc.T, state[self.sR]) # Transpose or not is the big question...
        FL = np.sqrt(np.maximum(xL,0))
        # coupling for rR
        xR = self.I - self.Idiff - self.b*state[self.aR] - self.w_inh*np.dot(self.Win.T, state[self.rL]) \
             - self.w_stretch*np.dot(self.Wsc.T, state[self.sL]) # Transpose or not is the big question...
        FR = np.sqrt(np.maximum(xR,0))

        # derivatives of the firing rates of the CPG neurons
        self.dstate[self.rL] = (-state[self.rL] + FL) / self.tau
        self.dstate[self.rR] = (-state[self.rR] + FR) / self.tau

        # derivatives of the firing rate adaptions
        self.dstate[self.all_a] = (-state[self.all_a] + self.gamma*state[self.all_r]) / self.taua 

        # derivatives of the muscle cell activities
        self.dstate[self.muscle_l] = self.gmc * np.dot(self.Wcm, state[self.rL]) \
                                              * (1-state[self.muscle_l])/self.taum_a \
                                              - state[self.muscle_l]/self.taum_d
        self.dstate[self.muscle_r] = self.gmc * np.dot(self.Wcm, state[self.rR]) \
                                        * (1-state[self.muscle_r])/self.taum_a \
                                        - state[self.muscle_r]/self.taum_d
        
        # ex 6 implementation                  !!! Pas sûr de cette implementation !!!
        # Perform cubic spline interpolation
        if self.w_stretch != 0:
            # self.poses is the actual position of the joints while pos is
            interp_func = CubicSpline(self.poses, pos)
            # Compute interpolated joint angles at the positions of the 50 sensors
            interpolated_joint_positions = interp_func(self.poses_ext)

            # derivative of the strech sensory neurons
            self.dstate[self.sL] = (np.sqrt(np.maximum(interpolated_joint_positions,0)) \
                                    * (1 - state[self.sL]) - state[self.sL]) / self.tau_str
            self.dstate[self.sR] = (np.sqrt(np.maximum(-interpolated_joint_positions,0)) \
                                    * (1 - state[self.sR]) - state[self.sR] ) / self.tau_str
        
        return self.dstate


