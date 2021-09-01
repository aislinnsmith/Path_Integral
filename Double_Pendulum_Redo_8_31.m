
%function LQR_Toy_Problem_with_Noise
%Here I am comparing the optimal trajectories given by the backwards
%Ricatti LQR formulation v.s. the path integral formulation on the inverted
%pendulum problem
close; clc; clear; 
%SDE := dx = Axdt+B(udt+dw); s.t. dE = BdW ----> E[dx^2] = E[BdW] = B*nu*B^T (affine
%transformation of random variable 
%Cost functional: phi(T) + int(u'*R*u + x'*B*x)dt 
%reference is LQR matlab site: https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace
%in this model u is a 1x1, B is a 4x1, A is a 4x4, and dW is a 1x1
n = 4;
lambda = 2;
dt = 0.001;
A = [0,1,0,0;0,0,-1,0;0,0,0,1;0,0,9,0]; 
B = [0;0.1;0;-0.1];
[A_d,B_d] = c2d(A,B,dt); %B_d = int_{0,dt}exp(A*dt)
R = 2*eye(n,n); 
R_d = R/dt;
Q = diag([1,1,10,10]); 
Q_d = diag([1,1,10,10])*dt; 
nu = B*B';
fun = @(t)exp(A*t)*nu*exp(A'*t);
nu_d = integral(fun,0,dt,'ArrayValued',true);
%This is the discretization of the covariance matrix 
sigma = real(sqrtm(nu_d));% The initial time step in the simulation
T0 = 0;
% The final time step in the simulation [units: hrs]
Tf = 0.005;
% Create a vector of times to run the simulation over using previously set time step size
Tspan = T0:dt:Tf;

% Determine the total number of time steps the simulation is evaluated at.
Tsize = length(Tspan)-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Important: for each time step, we are completely redoing the sampling because we need to sample ateach initial time, which
%will change based on what control imput we use in the previous time step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Construct the noise terms to be used for the Monte Carlo Simulation

% We use Wiener Process noise (form of white noise?) as it has the specific
% form necessary for the path integr%al control discussed in [Kappen, "Path
% Integrals and Symmetry Breaking for Optimal Control Theory", 2008. Page 3,7]

% The probability of the Wiener is Gaussian with mean zero and variance

x_t_c = 0.5*ones(n,Tsize); %n dimensional column vectors

u_t_c = zeros(n,Tsize); %scalar
running_cost = 0; 
rng default 
Discrete_Noise = zeros(n,Tsize);
Continuous_Noise = zeros(n,Tsize); 
for k = 1:Tsize
m = 100000; %initial condition of state vect

Random_Path_Cost = zeros(m,1,'gpuArray');% mx1 GPU array that stores the terminal cost of each sample path starting at initial state
Random_Path_Noise = zeros(n,m,'gpuArray'); %mx1 GPU array that stores the initial noise of each sample path starting at initial state
x_o = x_t_c(:,k); %state at current time step is assigned to be initial state for sample paths 
parfor j = 1:m
    
    Sample_Path = zeros(n,Tsize-(k-1)); %each column is 4 dim
    Sample_Path(:,1) = x_o;   
    Sample_Cost = 0;
    E = zeros(n,Tsize-(k-1)); %each samples is 4x1
    for i = 1:(Tsize-(k-1)) %every time step will shorten the trajectory path
        E(:,i) = randn(1,1);% 4 x 1 noise with adjusted variance 
        Sample_Path(:,i+1) = A_d*Sample_Path(:,i) + sigma*E(:,i); 
        Sample_Cost = Sample_Cost + (Sample_Path(:,i)'*Q_d*Sample_Path(:,i)); 
    end
    phi_Tf = (Sample_Path(:,Tsize-(k-1))'*Q_d* Sample_Path(:,Tsize-(k-1))) + Sample_Cost; 
    Random_Path_Cost(j,:) = (1/m)*exp(-phi_Tf/lambda);
    %Gives us terminal cost for the mth path
    Random_Path_Noise(:,j) = E(:,1);   
 end %Gives us noise at first step for mth path
  
%Goal is to apply function m times, and in each iteration simulate the
%entire sample path from intial state, return the final cost and initial noise 
psi_real = sum(Random_Path_Cost);
psi_real = gather(psi_real); 
 
Random_Path_Noise = gather(Random_Path_Noise); 
[Noise_Real] = arrayfun(@weighted_avg,Random_Path_Noise,Random_Path_Cost');
 

E_real = (sum(Noise_Real,2)); %weighted average of the rows (noise from each sample path) 
E_real = gather(E_real);
dE_real = sqrt(dt)*B*E_real; %covariance of dE is dt*B*B^T -> sigma = B*sqrt(dt)


Discrete_Noise(:,k) = sigma*Random_Path_Noise(:,m);
Continuous_Noise(:,k) = sqrt(dt)*B*Random_Path_Noise(:,m); 
%First we need to compute weights for the trajectories from the noise 
u_t_c(:,k) = ((B)*(dE_real))/((psi_real*dt)); %u_t_c = B*u_t_c

%Currently, the issue is that the control term is not doing much to deviate
%from uncontrolled path 
%By HJB: u = (-R^{-1}*B^T*d_xJ) = -R^{-1}*B^T*(-P)) (-P is defined in eqn
%20) 
%eqn 29 Kappen: sum(w_i*dE_i)/(psi*dt)
%in discretization, we assume the control is constant for each time step
%interval 

x_t_c(:,k+1) = A_d*x_t_c(:,k)+ u_t_c(:,k) + Discrete_Noise(:,k); %using discretized matrices 

running_cost = x_t_c(:,k)'*Q_d*x_t_c(:,k) + (1/2)*(u_t_c(:,k)'*R_d*u_t_c(:,k)) + running_cost;

end
running_cost
R = eye(n,n); 
R_d = R/dt;  % have to redefine R becaue cost functional has form $u^T*R*u for backwards ricatti (factor of 1/2 not included) 


y_t = 0.5*ones(n,Tsize);
u_t_2 = zeros(n,Tsize); 
K_t = zeros(n,n,Tsize);
[K,P] = dlqr(A_d,B_d,Q_d,R_d);
[K_1,P_1] = lqr(A,B,Q,R); %This is the steady state solution 
S = zeros(n,n,Tsize); 
S(:,:,Tsize) = P; 
running_cost_2 = 0; 
for i = Tsize:-1:2
    scalar = R_d + B_d'*S(:,:,i)*B_d;
    scalar_inv = scalar^(-1);
    S(:,:,i-1) = Q_d + A_d'*S(:,:,i)*A_d - A_d'*S(:,:,i)*B_d*scalar_inv*B_d'*S(:,:,i)*A_d;   
end

for t = 1:Tsize-1
    factor = R_d + B_d'*S(:,:,i)*B_d;
    factor_inv = scalar^(-1);
    K_t(:,:,t) = -factor_inv*B_d'*S(:,:,i+1)*A_d;
    u_t_2(:,t) = -K*y_t(:,t);
    y_t(:,t+1) =  A_d*y_t(:,t) + B_d*u_t_2(:,t) + Discrete_Noise(:,t);
    running_cost_2 = running_cost_2 + (y_t(:,t)'*Q_d*y_t(:,t)) + (u_t_2(:,t)'*R_d*u_t_2(:,t));
end
running_cost_2

mean_diff = 0; 
for i = 1:Tsize
    mean_diff = mean_diff + (1/Tsize)*norm((u_t_c(:,i) - u_t_2(:,k))); 
    
end
figure(1)
a = num2str(running_cost);
d = num2str(running_cost_2); 
e = strcat('y_t(1)',' ', d); 
b = strcat('x_t_c(1)',' ',a);
t = 1:Tsize;
tiledlayout(4,1)

nexttile
plot(t,x_t_c(1,t),'-o');
title(b)

nexttile
plot(t,u_t_c(1,t),'-b');
title('u_t_c(1,t)')

nexttile
plot(t,y_t(1,t),'-r'); 
title(e)

nexttile
plot(t,u_t_2(1,t),'-g'); 
title('u_t_2(1,t)'); 

