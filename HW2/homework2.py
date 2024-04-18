import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = (20, 20)
x_source, y_source = np.random.randint(0, GRID_SIZE[0], 2)
# x_source = 12
# y_source = 13
robot_pos = (np.random.randint(0, GRID_SIZE[0]), np.random.randint(0, GRID_SIZE[1]))  # Random initial position
# robot_pos = (12, 12)    
x_range = np.arange(0, GRID_SIZE[0])
y_range = np.arange(0, GRID_SIZE[1])
X, Y = np.meshgrid(x_range, y_range)

# likelihood_values = np.zeros_like(X, dtype=float)

def likelihood(x, y, x_source, y_source):
    # if x == x_source and y == y_source:
    #     return 1.0
    # elif y == y_source+1 or y == y_source-1:
    #     if x == x_source+1 or x == x_source-1 or x == x_source:
    #         return 0.5
    #     else:
    #         return 0.0
    # elif y == y_source+2 or y == y_source-2 :
    #     if x == x_source+2 or x == x_source-2 or x == x_source+1 or x == x_source-1 or x == x_source:
    #         return 1/3
    #     else:
    #         return 0.0
    # elif y == y_source+3 or y == y_source-3 :
    #     if x == x_source+3 or x == x_source-3 or x == x_source+2 or x == x_source-2 or x == x_source+1 or x == x_source-1 or x == x_source:
    #         return 0.25
    #     else:
    #         return 0.0
    # else:
    #     return 0.0

    dx = abs(x - x_source)
    dy = abs(y - y_source)
    if dy ==3 :
        if dx <=3:
            return 0.25
        else:
            return 0.0
    elif dy ==2 :
        if dx <=2:
            return 1/3
        else:
            return 0.0
    elif dy ==1 :
        if dx <=1:
            return 0.5
        else:
            return 0.0
    elif dy ==0 :
        if dx ==0:
            return 1.0
        else:
            return 0.0
    else:
        return 0.0


# Function to simulate a measurement
def simulate_measurement(robot_position):
    u = np.random.rand()
    return u < likelihood(robot_position[1], robot_position[0], x_source, y_source)
    
def simulate_measurement_grid(robot_position, x_grid, y_grid, m):
    if m == True:
        return likelihood(robot_position[1], robot_position[0], x_grid, y_grid)
    else:
        return 1 - likelihood(robot_position[1], robot_position[0], x_grid, y_grid)

def likelihood_grid(robot_position, m):
    likelihood_values = np.zeros_like(X, dtype=float)
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            likelihood_values[i, j] = simulate_measurement_grid(robot_position, j, i, m)
    
    # print(likelihood_values)
            
    return likelihood_values

# Function to update belief based on measurement
def update_belief(belief, measurement):
    updated_belief = (belief * measurement)/np.sum(belief * measurement)

    # print(np.sum(updated_belief))
    # updated_belief /= np.sum(updated_belief)
    return updated_belief

# Random exploration strategy
def random_exploration(num_steps, belief, robot_pos=robot_pos):
    robot_trajectory = []
    # robot_trajectory.append(robot_pos)
    belief = belief
    for t in range(num_steps):
        m = simulate_measurement(robot_pos)
        measurement = likelihood_grid(robot_pos, m)
        # measurement = simulate_measurement(robot_pos)
        belief = update_belief(belief, measurement)
        
        robot_trajectory.append((robot_pos[1], robot_pos[0]))
        dx, dy = np.random.choice([-1, 0, 1], 2, replace=False)  # Randomly choose motion direction
        if dx != 0 and dy != 0:  # If both dx and dy are non-zero, force one of them to be zero
            if np.random.rand() > 0.5:  # Randomly choose which one to zero out
                dx = 0
            else:
                dy = 0
        new_x = max(0, min(robot_pos[0] + dx, GRID_SIZE[0] - 1))  # Ensure within grid bounds
        new_y = max(0, min(robot_pos[1] + dy, GRID_SIZE[1] - 1))
        robot_pos = (new_x, new_y)
        if t % 10 == 0: 
            plt.imshow(belief, cmap='hot', interpolation='nearest', origin='lower')
            plt.title('Belief after Random Exploration')
            plt.colorbar(label='Belief')
            # plt.scatter(robot_pos[1], robot_pos[0], color='red', marker='o', label='Robot Position')
            plt.scatter(x_source, y_source, color='blue', marker='x', label='Source Position')
            plt.plot(*zip(*robot_trajectory), color='green', label='Robot Trajectory')
            plt.legend()
            plt.xlim(0, GRID_SIZE[1]-1)
            plt.ylim(0, GRID_SIZE[0]-1)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig('./HW2/assets/random_exploration_'+str(t)+'.png')
            plt.show()
            
    return belief, robot_trajectory

# Initialize belief
# belief = np.ones(GRID_SIZE).astype(float) / np.prod(GRID_SIZE)

# # belief = np.zeros(GRID_SIZE)

# # Run random exploration for 100 time steps
# belief, robot_trajectory = random_exploration(100, belief)

def entropy(belief):
    np.clip(belief, 1e-15, 1.0, belief)
    return -np.sum(belief * np.log(belief))

def entropy_calc(belief, robot_position):
    states = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
    h_bar = entropy(belief)
    estimate_entropy = []

    for state in states:
        robot_position_temp = (robot_position[0] + state[0], robot_position[1] + state[1])
        robot_position_temp = (max(0, min(robot_position_temp[0], GRID_SIZE[0] - 1)), max(0, min(robot_position_temp[1], GRID_SIZE[1] - 1)))
        measurement_pos = likelihood_grid(robot_position_temp, True)
        measurement_zero = likelihood_grid(robot_position_temp, False)
        belief_pos = update_belief(belief, measurement_pos)
        belief_zero = update_belief(belief, measurement_zero)
        entropy_pos = entropy(belief_pos)
        entropy_zero = entropy(belief_zero)
        
        print('Entropy:', entropy_pos, entropy_zero, h_bar)
        
        del_entropy_pos = entropy_pos - h_bar
        del_entropy_zero = entropy_zero - h_bar
        
        estimate_entropy_l= np.sum(belief*measurement_pos) * del_entropy_pos + np.sum(belief*measurement_zero) * del_entropy_zero 
        estimate_entropy.append(estimate_entropy_l)
    #     if estimate_entropy > h_bar:
    #         h_bar = estimate_entropy
    #         robot_position = robot_position_temp

    # h_bar = np.amin(estimate_entropy)
    h_index = np.argmin(estimate_entropy)
    robot_position = (robot_position[0] + states[h_index][0], robot_position[1] + states[h_index][1])
    #         print('Entropy:', h_bar)
    return h_bar, robot_position, belief


# define infotaxis

def infotaxis(belief, robot_pos=robot_pos):
    robot_trajectory = []
    h_bar = 1e20
    plot=[]
    # fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for t in range(110):
        print("h+bar", h_bar)
        if h_bar < 1e-6:
            print('Converged at time step:', t)
            break
        else:
            
        # measurement = simulate_measurement(robot_pos)
            h_bar, robot_pos, belief = entropy_calc(belief, robot_pos)
            m = simulate_measurement(robot_pos)
            measurement = likelihood_grid(robot_pos, m)
            belief = update_belief(belief, measurement)
            robot_pos = (robot_pos[0], robot_pos[1])
            robot_trajectory.append((robot_pos[1], robot_pos[0]))
            # if t % 10 == 0: 
            plt.imshow(belief, cmap='hot', interpolation='nearest', origin='lower')
            plt.title('Belief after Infotaxis')
            plt.colorbar(label='Belief')
            plt.scatter(x_source, y_source, color='blue', marker='x', label='Source Position')
            plt.plot(*zip(*robot_trajectory), color='green', label='Robot Trajectory')
            plt.legend()
            plt.xlim(0, GRID_SIZE[1]-1)
            plt.ylim(0, GRID_SIZE[0]-1)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.savefig('./HW2/assets/infotaxis_'+str(t)+'.png')
            plot.append(plt)
            plt.show()
            plt.clf()   
    #             idx = t // 10
    #             ax = axs[idx // 5, idx % 5]
    #             ax.imshow(belief, cmap='hot', interpolation='nearest', origin='lower')
    #             ax.scatter(x_source, y_source, color='blue', marker='x', label='Source Position')
    #             ax.plot(*zip(*robot_trajectory), color='green', label='Robot Trajectory')
    #             ax.set_title('Time Step {}'.format(t))
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    # plt.tight_layout()
    # plt.savefig('./HW2/assets/infotaxis_grid.png')
    # plt.show() 
    
        
            
    return belief, robot_trajectory

# def infotaxis(belief, robot_pos=robot_pos):
#     robot_trajectory = []
#     h_bar = 1e20
#     fig, axs = plt.subplots(2, 5, figsize=(15, 6))
#     for t in range(1000):
#         if h_bar < 1e-6:
#             print('Converged at time step:', t)
#             break
#         else:
#             h_bar, robot_pos, belief = entropy_calc(belief, robot_pos)
#             m = simulate_measurement(robot_pos)
#             measurement = likelihood_grid(robot_pos, m)
#             belief = update_belief(belief, measurement)
#             robot_pos = (robot_pos[0], robot_pos[1])
#             robot_trajectory.append((robot_pos[1], robot_pos[0]))
#             if t % 10 == 0 and t < 100: 
#                 idx = t // 10
#                 ax = axs[idx // 5, idx % 5]
#                 ax.imshow(belief, cmap='hot', interpolation='nearest', origin='lower')
#                 ax.scatter(x_source, y_source, color='blue', marker='x', label='Source Position')
#                 ax.plot(*zip(*robot_trajectory), color='green', label='Robot Trajectory')
#                 ax.set_title('Time Step {}'.format(t))
#                 ax.set_xticks([])
#                 ax.set_yticks([])
                
    # plt.tight_layout()
    # plt.savefig('./HW2/assets/infotaxis_grid.png')
    # plt.show()

                
    # plt.tight_layout()
    # plt.savefig('./HW2/assets/infotaxis_grid.png')
    # plt.show()

        
belief = np.ones(GRID_SIZE).astype(float) / np.prod(GRID_SIZE)
belief, robot_trajectory = infotaxis(belief)




        
# def entropy_calc(belief):

# x_range = np.arange(0, GRID_SIZE[0])
# y_range = np.arange(0, GRID_SIZE[1])
# X, Y = np.meshgrid(x_range, y_range)

# # Calculate likelihood values for each grid cell
# likelihood_values = np.zeros_like(X, dtype=float)
# for i in range(GRID_SIZE[0]):
#     for j in range(GRID_SIZE[1]):
#         likelihood_values[i, j] = likelihood(j, i, x_source, y_source)


# # Plot likelihood function
# plt.imshow(likelihood_values, cmap='hot', interpolation='nearest', origin='lower')
# plt.title('Likelihood Function')
# plt.colorbar(label='Likelihood')
# plt.scatter(x_source, y_source, color='red', marker='o', label='Robot Position')
# plt.scatter(x_source, y_source, color='blue', marker='x', label='Source Position')
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
