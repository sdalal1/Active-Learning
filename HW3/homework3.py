import numpy as np
import matplotlib.pyplot as plt

# def motion_model(state, u , dt):
    
#     # state = [x, y, theta]
#     # u = [u1, u2]
    
#     x_dot = u[0] * np.cos(state[2]) 
#     y_dot = u[0] * np.sin(state[2]) 
#     theta_dot = u[1]
    
#     state[0] += x_dot * dt + np.random.normal(0, np.sqrt(0.002))
#     state[1] += y_dot * dt + np.random.normal(0, np.sqrt(0.002))
#     state[2] += theta_dot * dt + np.random.normal(0, np.sqrt(0.002))

#     return state

# def measurement_model(state, var):
    
#     # state = [x, y, theta]
#     # z = [x, y, theta]
#         x = state[0] + np.random.normal(0, np.sqrt(var))
#         y = state[1] + np.random.normal(0, np.sqrt(var))
#         theta = state[2] + np.random.normal(0, np.sqrt(var))
        
#         return [x, y, theta]
    
# num_particle = 100
# particles = np.zeros((num_particle, 3))
# particles[:, 2] = np.random.normal(np.pi/2,np.sqrt(0.02) ,num_particle)
# particles[:, 0] = np.random.normal(0.0, np.sqrt(0.02), num_particle)
# particles[:, 1] = np.random.normal(0.0, np.sqrt(0.02), num_particle)

# state = [0, 0, np.pi/2]

# u = [1, -1/2]

# # T = 2 * np.pi
# T = 6.1

# dt = 0.1

# # var = np.sqrt(0.02)

# var  = 0.002
# state_estimate = [0, 0, np.pi/2]
# state_array = []
# ground_truth_array = []
# for t in np.arange(0, T, dt):
#     z = measurement_model(state, var)
     
#     for i in range(num_particle):
#         particles[i] = motion_model(particles[i], u, dt)
        
#     # likelihood = np.ones(num_particle)/num_particle
    
#     # z = measurement_model([np.mean(particles[:, 0]), np.mean(particles[:, 1]), np.mean(particles[:, 2])], var)
#     weigth = np.zeros(num_particle)
#     ground_truth_array.append(z)
#     for i in range(num_particle):
#         weigth[i] = np.exp(-0.5 * np.sum((particles[i] - z)**2)/np.sqrt(0.02))
        
#     weigth /= np.sum(weigth)
    
#     index = np.random.choice(np.arange(num_particle), num_particle, p = weigth)
#     particles = particles[index]
#     # state_estimate = np.array([np.mean(particles[:, 0]), np.mean(particles[:, 1]), np.mean(particles[:, 2])])
#     state_estimate = np.average(particles, axis=0, weights=weigth)


#     state = motion_model(state_estimate, u, dt)
#     if t == 0 or t == 1 or t == 2 or t == 3 or t == 4 or t == 5 or t == 6:

#         plt.scatter(particles[:, 0], particles[:, 1], color = plt.cm.hsv(t/T), alpha = 0.2)
    
#     #plot the state estimate in red as a curve
#     state_array.append(state_estimate)

#     # plt.line(state_estimate[0], state_estimate[1], s = 0.2, c = 'red')
#     # plt.scatter(ground_truth[0], ground_truth[1], s = 100, c = 'blue')

# state_array = np.array(state_array)
# ground_truth_array = np.array(ground_truth_array)
# plt.plot(ground_truth_array[:, 0], ground_truth_array[:, 1], color = 'black')
# plt.plot(state_array[:, 0], state_array[:, 1], color = 'red')
# plt.show()


#### problem 3
samples = 100


w1 = 0.5
w2 = 0.2
w3 = 0.3

w = np.array([w1, w2, w3])



u1 = np.array([0.35, 0.38]).T
u2 = np.array([0.68, 0.25]).T
u3 = np.array([0.56, 0.64]).T

u = np.array([u1, u2, u3])


sigma1 = np.array([[0.01, 0.004], [0.004, 0.01]])
sigma2 = np.array([[0.005, -0.003], [-0.003, 0.005]])
sigma3 = np.array([[0.008, 0.0], [0.0, 0.004]])

sigma = np.array([sigma1, sigma2, sigma3])




# plt.scatter(x[0, :, 0], x[0, :, 1], color = 'red')
# plt.scatter(x[1, :, 0], x[1, :, 1], color = 'blue')
# plt.scatter(x[2, :, 0], x[2, :, 1], color = 'green')

gamma = np.zeros((samples, 3))

a=0


while a < 6:
    # x1 = np.random.multivariate_normal(u_list[0].T, sigma1[0], samples)
    # x2 = np.random.multivariate_normal(u_list[1].T, sigma2[0], samples)
    # x3 = np.random.multivariate_normal(u_list[2].T, sigma3[0], samples)

    # x1 = np.random.multivariate_normal(u1, sigma1, samples)
    # x2 = np.random.multivariate_normal(u2, sigma2, samples)
    # x3 = np.random.multivariate_normal(u3, sigma3, samples)
    print(u1)
    x1 = np.random.multivariate_normal(u1, sigma1, samples)
    x2 = np.random.multivariate_normal(u2, sigma2, samples)
    x3 = np.random.multivariate_normal(u3, sigma3, samples)
    
    x = []
    for i in range(samples):
        x.append(np.array([x1[i], x2[i], x3[i]]))
    
    x = np.array(x)
    print(x.shape)
    
    gamma = []
    
    # convergence criteria
    

    for i in range(samples):
        den = w1 * x1[i] + w2 * x2[i] + w3 * x3[i]
        gamma.append(np.array([w1 * x1[i]/den, w2 * x2[i]/den, w3 * x3[i]/den]))
    
    gamma = np.array(gamma)
        
    for k in range(3):
        gamma_sum = 0
        gamma_x_sum = np.zeros_like(u[k])
        gamma_x_xT_sum = np.zeros_like(sigma[k])

        for i in range(samples):
            gamma_sum += gamma[i, k]
            gamma_x_sum += gamma[i, k] * x[i, k]
            gamma_x_xT_sum += gamma[i, k] * np.outer((x[i, k] - u[k]), (x[i, k] - u[k]))

        if k == 0:
            w1 = gamma_sum / samples
            u1 = gamma_x_sum / gamma_sum
            sigma1 = gamma_x_xT_sum / gamma_sum
        elif k == 1:
            w2 = gamma_sum / samples
            u2 = gamma_x_sum / gamma_sum
            sigma2 = gamma_x_xT_sum / gamma_sum
        else:
            w3 = gamma_sum / samples
            u3 = gamma_x_sum / gamma_sum
            sigma3 = gamma_x_xT_sum / gamma_sum
        
    # for k in range(3):
    #     # w[k] = np.sum(gamma[:, k])/samples
    #     # u[k] = np.sum(gamma[:, k] * x[k])/np.sum(gamma[:, k])
    #     # sigma[k] = np.sum(gamma[:, k] * (x[k] - u[k]) * (x[k] - u[k]))/np.sum(gamma[:, k])
    #     gamma_sum = 0
    #     gamma_x_sum = 0
    #     gamma_x_xT_sum = 0
    #     print("gamma_shape", gamma[i].shape)
    #     print("x_shape", x[i, k].shape)
    #     print("u_shape", u[k].shape)

    #     for i in range(samples):
    #         gamma_sum += gamma[i,k]
    #         gamma_x_sum += gamma[i,k] * x[i]
    #         gamma_x_xT_sum += gamma[i,k] * (x[i, k] - u[k]) * (x[i, k] - u[k]).T
    
        
    #     if k == 0:
    #         w1 = gamma_sum/samples
    #         u1 = gamma_x_sum/gamma_sum
    #         sigma1 = gamma_x_xT_sum/gamma_sum
    #     elif k == 1:
    #         w2 = gamma_sum/samples
    #         u2 = gamma_x_sum/gamma_sum
    #         sigma2 = gamma_x_xT_sum/gamma_sum
    #     else:
    #         w3 = gamma_sum/samples
    #         u3 = gamma_x_sum/gamma_sum
    #         sigma3 = gamma_x_xT_sum/gamma_sum
        # if k == 0:
        #     w1 = np.sum(gamma[:, k])/samples
        #     u1 = np.sum(np.sum(gamma[:, k] * x[k], axis=0))/np.sum(gamma[:, k])
        #     sigma1 = np.sum(gamma[:, k] * (x[k] - u[k]) * (x[k] - u[k]))/np.sum(gamma[:, k])
        # elif k == 1:
        #     w2 = np.sum(gamma[:, k])/samples
        #     u2 = np.sum(gamma[:, k] * x[k])/np.sum(gamma[:, k])
        #     sigma2 = np.sum(gamma[:, k] * (x[k] - u[k]) * (x[k] - u[k]))/np.sum(gamma[:, k])
        # else:
        #     w3 = np.sum(gamma[:, k])/samples
        #     u3 = np.sum(gamma[:, k] * x[k])/np.sum(gamma[:, k])
        #     sigma3 = np.sum(gamma[:, k] * (x[k] - u[k]) * (x[k] - u[k]))/np.sum(gamma[:, k])
        

    # while sigma_list[0][-1] < 0.0001 and sigma_list[1][-1] < 0.0001 and sigma_list[2][-1] < 0.0001:
    #     for i in range(samples):
    #         den = w1 * x1[i] + w2 * x2[i] + w3 * x3[i]
    #         gamma[i] = np.array([w1 * x1[i]/den, w2 * x2[i]/den, w3 * x3[i]/den])
            
    #     for k in range(3):
    #         w[k] = np.sum(gamma[:, k])/samples
    #         u[k] = np.sum(gamma[:, k] * x[k])/np.sum(gamma[:, k])
    #         sigma[k] = np.sum(gamma[:, k] * (x[k] - u[k]) * (x[k] - u[k]).T)/np.sum(gamma[:, k])

    #         w_list[k] = np.append(w_list[k], w[k])
    #         u_list[k] = np.append(u_list[k], u[k])
    #         sigma[k] = np.append(sigma[k], sigma[k])
            
    a += 1


    # plt.scatter(x[0, :, 0], x[0, :, 1], color = 'red')
    # plt.scatter(x[1, :, 0], x[1, :, 1], color = 'blue')
    # plt.scatter(x[2, :, 0], x[2, :, 1], color = 'green')
    for i in range(samples):
        plt.scatter(x[i, 0, 0], x[i, 0, 1], color='red')  # x1
        plt.scatter(x[i, 1, 0], x[i, 1, 1], color='blue')  # x2
        plt.scatter(x[i, 2, 0], x[i, 2, 1], color='green')  # x3

    plt.show()  

  