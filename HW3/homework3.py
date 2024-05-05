import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# # scipy.stats.multivariate_normal

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
# plt.plot(ground_truth_array[:, 0], ground_truth_array[:, 1], color = 'black', label = 'Ground Truth')
# plt.plot(state_array[:, 0], state_array[:, 1], color = 'red', label = 'State Estimate')
# plt.legend()
# plt.show()


#### problem 3
samples = 100


w1 = 0.5
w2 = 0.2
w3 = 0.3

w = np.array([w1, w2, w3])



u1 = np.array([0.35, 0.38])
u2 = np.array([0.68, 0.25])
u3 = np.array([0.56, 0.64])

u = np.array([u1, u2, u3])


sigma1 = np.array([[0.01, 0.004], [0.004, 0.01]])
sigma2 = np.array([[0.005, -0.003], [-0.003, 0.005]])
sigma3 = np.array([[0.008, 0.0], [0.0, 0.004]])

sigma = np.array([sigma1, sigma2, sigma3])


def generate_samples(weights, means, covs, num_samples):
    samples = np.zeros(shape=(num_samples, 3))

    indeces = np.arange(0, len(weights))

    for i in range(num_samples):
        index = np.random.choice(a=indeces, p=weights, size=1)[0]

        mean = means[index, :]
        cov = covs[index, :, :]

        s = np.random.multivariate_normal(mean=mean, cov=cov)
        samples[i, :2] = s
        samples[i, 2] = index

    return samples

def point_in_gaussian(samples,gamma, mean, cov, weight, ground_truth):
    first_gaussian = []
    second_gaussian = []
    third_gaussian = []
    
    for i in range(len(samples)):
        # print(samples[i])
        # print(gamma[i])
        if gamma[i, 0] > gamma[i, 1] and gamma[i, 0] > gamma[i, 2]:
            first_gaussian.append(samples[i])
        elif gamma[i, 1] > gamma[i, 0] and gamma[i, 1] > gamma[i, 2]:
            second_gaussian.append(samples[i])
        else:   
            third_gaussian.append(samples[i])
    
    first_gaussian = np.array(first_gaussian)
    second_gaussian = np.array(second_gaussian)
    third_gaussian = np.array(third_gaussian)
    
    print("second",second_gaussian.shape)
    print("third",third_gaussian.shape)
    print("first",first_gaussian.shape)

    Z_tot = np.zeros((100, 100))
    for i in range(3):
        x = np.linspace(0, 1.0, 100)
        y = np.linspace(0, 1.0, 100)
        X, Y = np.meshgrid(x, y)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                # pos = np.array([X[j,k], Y[j,k]])
                Z = weight[i] * np.exp(-0.5 * np.dot(np.dot(np.array([X[j,k], Y[j,k]]) - mean[i], np.linalg.inv(cov[i])), np.array([X[j,k], Y[j,k]]) - mean[i]))
                Z_tot[j,k] += Z

    Z_groud_truth = np.zeros((100, 100))
    for i in range(3):
        x = np.linspace(0, 1.0, 100)
        y = np.linspace(0, 1.0, 100)
        X, Y = np.meshgrid(x, y)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                # pos = np.array([X[j,k], Y[j,k]])
                Z = multivariate_normal.pdf([X[j,k], Y[j,k]], mean=u[i], cov=sigma[i])
                Z_groud_truth[j,k] += Z
    
    colors = ['red', 'blue', 'green']
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[2].title.set_text('Predicted')
    ax[2].contourf(X, Y, Z_tot, 10, cmap='gray_r')
    ax[2].scatter(first_gaussian[:, 0], first_gaussian[:, 1], color = colors[0])
    ax[2].scatter(second_gaussian[:, 0], second_gaussian[:, 1], color = colors[1])
    ax[2].scatter(third_gaussian[:, 0], third_gaussian[:, 1], color = colors[2])
    for i in range(3):
        w, v = np.linalg.eig(cov[i])
        angle = np.arctan2(v[1,0], v[0,0])
        ellipse = plt.matplotlib.patches.Ellipse(xy=mean[i], width=5*np.sqrt(w[0]), height=5*np.sqrt(w[1]), angle=angle*180/np.pi, edgecolor=colors[i], facecolor='none')
        ax[2].add_patch(ellipse)
    
    ax[0].title.set_text('Samples')
    ax[0].scatter(ground_truth[:, 0], ground_truth[:, 1], color = 'black')
    ax[1].title.set_text('Ground Truth')
    ax[1].contourf(X, Y, Z_groud_truth, 10, cmap='gray_r')
    for i in range(3):
        ax[1].scatter(ground_truth[ground_truth[:, 2] == i, 0], ground_truth[ground_truth[:, 2] == i, 1], color = colors[i])
    # ax[1].scatter(ground_truth[:, 0], ground_truth[:, 1], c=ground_truth[:, 2])
    for i in range(3):
        w, v = np.linalg.eig(sigma[i])
        angle = np.arctan2(v[1,0], v[0,0])
        ellipse = plt.matplotlib.patches.Ellipse(xy=u[i], width=5*np.sqrt(w[0]), height=5*np.sqrt(w[1]), angle=angle*180/np.pi, edgecolor=colors[i], facecolor='none')
        ax[1].add_patch(ellipse)
    plt.show()
    

ground_truth = generate_samples(w, u, sigma, samples)
gamma = np.zeros((samples, 3))

a=0

sample_mean = np.mean(ground_truth, axis=0)
# sample_cov = np.cov(ground_truth[].T)
# point_in_gaussian(ground_truth, gamma)

pred_weight = np.ones(3)/3
pred_mean = np.array([[0.2, 0.2], [0.8, 0.2], [0.6, 0.8]])
# pred_mean = np.array([sample_mean[0:2]+np.random.uniform(-0.1,0.1), sample_mean[0:2]+np.random.uniform(-0.1,0.1), sample_mean[0:2]+np.random.uniform(-0.1,0.1)])
# pred_cov = np.array([np.eye(2)*0.01, np.eye(2)*0.01, np.eye(2)*0.01])
pred_cov = np.array([[[0.01, 0.004], [0.004, 0.01]], [[0.005, -0.003], [-0.003, 0.005]], [[0.008, 0.0], [0.0, 0.004]]])
pred_cov += np.random.uniform(low=-0.001, high=0.001, size=pred_cov.shape)
while a < 5:
    gamma = np.zeros((samples, 3))
    # convergence criteria
    for i in range(samples):
        for j in range(3):

            gamma[i, j] = pred_weight[j] * multivariate_normal.pdf(ground_truth[i, 0:2], mean=pred_mean[j], cov=pred_cov[j])

        gamma[i] /= np.sum(gamma[i])
    
    gamma = np.array(gamma)
    # print(gamma.shape)
    point_in_gaussian(ground_truth, gamma, pred_mean, pred_cov, pred_weight, ground_truth)

    for k in range(3):
        
        pred_weight[k] = np.sum(gamma[:, k]) / samples
        pred_weight[k] /= np.sum(pred_weight)
        print(k,pred_weight)
        
        pred_mean[k] = np.sum(ground_truth[:, 0:2] * gamma[:,k][:, np.newaxis], axis=0) / np.sum(gamma[:, k])
        
        print(k,pred_mean)
        
        pred_cov[k] = np.zeros_like(pred_cov[k])

        pred_cov[k] = np.dot((ground_truth[:, 0:2] - pred_mean[k]).T, (ground_truth[:, 0:2] - pred_mean[k]) * gamma[:, k][:, np.newaxis]) / np.sum(gamma[:, k])
        
        print(k,pred_cov)
  
    a += 1

    # plt.show()
    

  