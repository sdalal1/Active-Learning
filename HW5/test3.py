import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import copy
from scipy.stats import multivariate_normal as mvn

# Define the Gaussian Mixture Model parameters
mean1 = np.array([0.35, 0.38])
cov1 = np.array([
    [0.01, 0.004],
    [0.004, 0.01]
])
w1 = 0.5

mean2 = np.array([0.68, 0.25])
cov2 = np.array([
    [0.005, -0.003],
    [-0.003, 0.005]
])
w2 = 0.2

mean3 = np.array([0.56, 0.64])
cov3 = np.array([
    [0.008, 0.0],
    [0.0, 0.004]
])
w3 = 0.3

# Initialize lists for ILQR
phi_list = np.empty(0)
fk_list = np.empty(0)
h_list = np.empty(0)
c_list = np.empty(0)
f_traj = np.empty(0)
lam_list = np.empty(0)
dfkdxdt_list = np.empty(0)

# Initial control trajectory to move right initially
# init_u_traj = np.tile(np.array([1.0, 0.0]), reps=100)

# Function to calculate the PDF of the Gaussian Mixture Model
def pdf(x, w, dists, dx, dy):
    r = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(w.shape[0]):
            r[i] += w[j] * dists[j].pdf(x[i])
    r /= np.sum(r * dx * dy)
    return r

# Simulation parameters
dt = 0.1
T = 15.0
tlist = np.arange(0, T, dt)
tsteps = tlist.shape[0]

x0 = np.array([0.3, 0.3, np.pi/2.0])
q = 0.1

R_u=np.diag([0.01, 0.01])
Q_x = np.diag([0.01, 0.01])
P1=np.diag([2.0, 2.0])

Q_z = np.diag([0.01, 0.01, 0.001]),
R_v = np.diag([0.01, 0.01])

weights = np.array([w1, w2, w3])
means = np.array([mean1, mean2, mean3])
covs = np.array([cov1, cov2, cov3])

# Create distribution objects for each Gaussian component
dist = []
for mu, cov in zip(means, covs):
    dist.append(mvn(mu, cov))

# Define the Fourier basis functions
num_k_per_dim = 10
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(num_k_per_dim), np.arange(num_k_per_dim)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T  # this is the set of all index vectors

# Create a grid for evaluation
L_list = np.array([1.0, 1.0])
grids_x, grids_y = np.meshgrid(
    np.linspace(0, L_list[0], 1001),
    np.linspace(0, L_list[1], 1001)
)

grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

N = 1000 + 1
dx = dy = 1 / 1000
grid_pdf = pdf(grids, weights, dist, dx, dy)

# Initialize lists for Fourier coefficients
phi_list = np.zeros((ks.shape[0]))
fk_list = np.zeros(shape=(ks.shape[0], N * N))
h_list = np.zeros((ks.shape[0]))

for i, k_vec in enumerate(ks):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)
    h_temp = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
    fk_vals /= h_temp
    fk_list[i, :] = fk_vals

    phi_list[i] = np.sum(grid_pdf * fk_vals * dx * dy)
    h_list[i] = h_temp

c_list = np.zeros(ks.shape[0])
f_traj = np.zeros((ks.shape[0], tsteps))
lam_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3 / 2.0)
# dfdxdt = np.zeros((ks.shape[0], 2))
dfkdxdt_list = np.zeros((ks.shape[0], 2))

# Define the system dynamics
def dyn(xt, ut):
        # xdot = np.zeros(3)  # replace this
        theta = xt[2]
        u1 = ut[0]
        u2 = ut[1]
        x1dot = np.cos(theta) * u1
        x2dot = np.sin(theta) * u1
        x3dot = u2
        # x1dot = xt[2]
        # x2dot = xt[3]
        # x1ddot = ut[0]
        # x2ddot = ut[1]

        xdot = np.array([x1dot, x2dot, x3dot])
        # xdot = copy.deepcopy(ut)
        # xdot = np.array([x1dot, x2dot, x1ddot, x2ddot])
        return xdot

def get_A(t, xt, ut):
    theta = xt[2]
    u1 = ut[0]
    A_mat = np.zeros((3, 3))  # replace this
    # A_mat = np.zeros((4, 4))
    # A_mat[0, 2] = 1
    # A_mat[1, 3] = 1
    A_mat[0, 2] = -np.sin(theta) * u1
    A_mat[1, 2] = np.cos(theta) * u1
    return A_mat

def get_B(t, xt, ut):
    theta = xt[2]
    B_mat = np.zeros((3, 2))  # replace this
    # B_mat[2, 0] = 1
    # B_mat[3, 1] = 1
    # B_mat = np.eye(2)
    B_mat[0, 0] = np.cos(theta)
    B_mat[1, 0] = np.sin(theta)
    B_mat[2, 1] = 1
    return B_mat

# Define the integration step
def step(xt, ut):
    k1 = dt * dyn(xt, ut)
    k2 = dt * dyn(xt + 0.5 * k1, ut)
    k3 = dt * dyn(xt + 0.5 * k2, ut)
    k4 = dt * dyn(xt + k3, ut)
    xt_new = xt + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return xt_new

# Simulate the trajectory
def traj_sim(x0, ulist):
    # tsteps = ulist.shape[0]
    x_traj = np.zeros((tsteps + 1, x0.shape[0]))
    x_traj[0, :] = x0
    xt = copy.deepcopy(x0)
    for t in range(tsteps):
        xt_new = step(xt, ulist[t])
        x_traj[t + 1] = copy.deepcopy(xt_new)
        xt = copy.deepcopy(xt_new)
    return x_traj

# Define the gradients for the cost function
def dldx(t, xt, ut):
    index = np.where(tlist == t)[0][0]
    dvec = np.zeros(2)
    for lam_k, ck, phi_k, hk, k_vec, fk_traj, dfkdxdt in zip(
            lam_list, c_list, phi_list, h_list, ks, f_traj, dfkdxdt_list
        ):
        fk = fk_traj[index]
        dfdx = -1 / hk * np.pi * np.sin(k_vec * np.pi * xt[:2]) * \
               np.cos(k_vec[::-1] * np.pi * xt[1::-1])
        dvec += q * lam_k * 2 * (ck - phi_k) * 1 / T * dfdx
    return np.append(dvec, 0)

def dldu(t, xt, ut):
    rlist = np.diag(R_u)
    dvec = rlist * ut
    return dvec

# Define the cost function
def J(x, u):
    J_v = 0.0
    J_v += q * np.sum(lam_list * np.square(c_list - phi_list))
    # print(q * np.sum(lam_list * np.square(c_list - phi_list)))
    # print(u.shape)
    for a in u:
        # a = np.atleast_1d(a)  # Ensure a is a 1D array
        J_v += a.T @ R_u @ a * dt
    return J_v

# ILQR iteration
def ilqr_iter(x0, u_traj):
    x_traj = traj_sim(x0, u_traj)
    c_list = np.zeros(ks.shape[0])
    f_t = np.zeros((ks.shape[0], x_traj.shape[0]))
    for i, (k, h) in enumerate(zip(ks, h_list)):
        f_val = np.prod(np.cos(np.pi * k / L_list * x_traj[:,:2]), axis=1)
        dfdxdt = np.zeros(2)
        f_val /= h
        ck = np.sum(f_val) * dt / T
        for xt in x_traj:
            dfdx = -1 / h * np.pi * np.sin(k * np.pi * xt[:2]) * np.cos(k[::-1] * np.pi * xt[1::-1])
            dfdxdt += dfdx * dt
        dfkdxdt_list[i, :] = dfdxdt
        # dfkdxdt_list[i] = dfdxdt
        
        f_t[i, :] = f_val
        c_list[i] = ck

    A_list = np.zeros((tsteps, 3, 3))
    B_list = np.zeros((tsteps, 3, 2))
    a_list = np.zeros((tsteps, 3))
    b_list = np.zeros((tsteps, 2))

    for idx, t in enumerate(tlist):
        A_list[idx] = get_A(t, x_traj[idx], u_traj[idx])
        B_list[idx] = get_B(t, x_traj[idx], u_traj[idx])
        a_list[idx] = dldx(t, x_traj[idx], u_traj[idx])
        b_list[idx] = dldu(t, x_traj[idx], u_traj[idx])

    xT = x_traj[-1, :]
    xT_2 = x_traj[int(x_traj.shape[0] / 2), :]
    # p1 = 2 * P1 * (xT - mean2) * (xT_2 - mean3)
    p1 = np.zeros(3)

    def zp_dyn(t, zp):
        zt = zp[:3]
        pt = zp[3:]
        t_idx = (t / dt).astype(int)
        At = A_list[t_idx]
        Bt = B_list[t_idx]
        at = a_list[t_idx]
        bt = b_list[t_idx]
        M11 = At
        M12 = np.zeros((3, 3))
        M21 = np.zeros((3, 3))
        M22 = -At.T
        dyn_mat = np.block([[M11, M12], [M21, M22]])
        # print(Bt.shape, R_v.T.shape, pt.shape, bt.shape)
        m1 = -Bt @ np.linalg.inv(R_v.T) @ (pt.T @ Bt + bt.T)
        m2 = -at - zt @ Q_z
        # print("m1", m1)
        # print("m2", m2[0])
        dyn_vec = np.hstack([m1, m2[0]])
        return dyn_mat @ zp + dyn_vec

    def zp_dyn_list(t_list, zp_list):
        list_len = len(t_list)
        zp_dot_list = np.zeros((6, list_len))
        for _i in range(list_len):
            zp_dot_list[:, _i] = zp_dyn(t_list[_i], zp_list[:, _i])
        return zp_dot_list

    def zp_bc(zp0, zpT):
        z0 = zp0[:3]
        p0 = zp0[3:]
        zT = zpT[:3]
        pT = zpT[3:]
        bc = np.zeros(6)
        bc[:3] = z0
        bc[3:] = np.abs(pT - p1)
        return bc

    res = solve_bvp(zp_dyn_list, zp_bc, tlist, np.zeros((6, tsteps)), verbose=1, max_nodes=100)
    zp_t = res.sol(tlist).T
    # z_t = zp_t[:2, :]
    # p_t = zp_t[2:, :]
    z_t = zp_t[:, :3]
    p_t = zp_t[:, 3:]

    u_traj_new = np.zeros((tsteps, 2))
    for _i in range(tsteps):
        At = A_list[_i]
        Bt = B_list[_i]
        at = a_list[_i]
        bt = b_list[_i]
        zt = z_t[_i]
        pt = p_t[_i]
        u_traj_new[_i, :] = -np.linalg.inv(R_v.T) @ (pt.T @ Bt + bt.T)

    return u_traj_new, c_list

# Run ILQR iterations
# u_traj = init_u_traj.copy()
u_traj = np.tile(np.array([0.02 * np.pi, -np.pi / 10.0]), reps=(tsteps, 1))
init_u = np.tile(np.array([0.02 * np.pi, -np.pi / 10.0]), reps=(tsteps, 1))
init_x_traj = traj_sim(x0, init_u)

# J_list = np.array([J(traj_sim(x0, u_traj), u_traj)])
J_list = np.array([J(traj_sim(x0, u_traj), u_traj)])

for iter in range(100):
    x_traj = traj_sim(x0, u_traj)
    v_traj, c_list = ilqr_iter(x0, u_traj)
    gamma = 0.001
    alpha = 1e-3
    beta = 0.5
    while J(x_traj, u_traj + gamma * v_traj) > J(x_traj, u_traj) + alpha * gamma * np.abs(np.trace(v_traj.T @ v_traj)):
        gamma *= beta
    u_traj += gamma * v_traj
    
    J_list = np.hstack([J_list, J(x_traj, u_traj)])


# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(15, 3))
# print(x_traj)
# Plot trajectories
ax[0].plot(init_x_traj[:, 0], init_x_traj[:, 1], linestyle='-', color='C0', label='Initial')
ax[0].plot(x_traj[:, 0], x_traj[:, 1], linestyle='-', color='k', label='Final')
ax[0].scatter(x_traj[0, 0], x_traj[0, 1], c='r', alpha=1.)
ax[0].imshow(grid_pdf.reshape(grids_x.shape), extent=(0, 1, 0, 1), origin='lower', cmap='Reds', alpha=1.)
ax[0].legend(loc='upper right')
ax[0].set_title('Trajectories')

# Plot control inputs
ax[1].plot(np.arange(0, T, dt), u_traj[:, 0], linestyle='-', color='C0', label='u1')
ax[1].plot(np.arange(0, T, dt), u_traj[:, 1], linestyle='-', color='k', label='u2')
ax[1].legend(loc='upper right')
ax[1].set_title('Control Inputs')

# Plot cost function
ax[2].plot(J_list, linestyle='-', color='C0')
ax[2].set_title('Cost Function')

plt.show()