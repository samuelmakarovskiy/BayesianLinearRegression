#Samuel Makarovskiy, Bayesian ML HW2 (Figure 3.7 Simulation)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from math import sqrt

def main():

    #Known Parameters
    a0 = -0.3
    a1 = 0.5
    noiseSD = .2
    Beta = (1/noiseSD)**2
    alpha = 2
    density = 100 #how dense the mesh is for contour plots (arbitrary)

    #Plot Configuration - Bunch of Beautification
    fig = plt.figure(figsize=(6,9))
    ax = fig.subplots(4,3)
    plt.setp(ax, aspect = 'equal', xlim = (-1,1), ylim = (-1,1), \
             xticks = [-1,0,1],  yticks = [-1,0,1])
    plt.setp(ax[0:4,0:2], xlabel = 'w\u2080', ylabel = 'w\u2081')
    plt.setp(ax[0:4,2], xlabel = 'x', ylabel = 'y')
    ax[0,0].axis('off')         #Turn off subplot 1,1
    ax[0,0].set_title('Likelyhood')
    ax[0,1].set_title('Prior/Posterior')
    ax[0,2].set_title('Data Space')
    fig.tight_layout()          #Plot so that subplot labels don't overlap 




    #Generation of 20 data points
    #generate random gaussian noise on sample targets (y):
    noise_sample = np.random.normal(0,noiseSD,size = 20)
    #generate random x-samples:
    x_sample = np.random.uniform(low = -1.0, high = 1.0, size = 20)
    #generate y samples based on actual weights and generated gaussian noise
    y_sample = np.add(a0,a1*x_sample+noise_sample)  
    phi = np.empty([20,2])      #define max size of capital phi matrix for 20 samples
    phi[:,0] = 1                #First column corresponds to w0(x_i) which is constant
    phi[:,1] = x_sample         #Second column corresponds to w1(x_i) = x_i  (linear model)
    
    
    #Initial Prior Distribution (Plot in index 0,1):
    PriorProb = np.empty([density,density]) #initialize PriorProb for each mesh point
    xy = np.linspace(-1,1,density)          #mesh density along one dimension (x or y)
    X, Y = np.meshgrid(xy,xy)               #define meshgrid for all contours used henceforth
    for i in range(density):
        for j in range(density):
            #calculate prior prob at all pts in meshgrid:
            PriorProb[i][j] = \
                            multivariate_normal(np.array([0,0]),1/alpha*np.eye(2)).pdf([xy[j],xy[i]])
            
    ax[0,1].contourf(X,Y,PriorProb,150,cmap = 'jet') #plot prior contour plot on subplot 1,2
    

    #Likelyhood plots: p(t|x,w,beta)
    #initialize LikeyhoodProb for each mesh point and for each of the three samples:
    LikelyhoodProb = np.empty([3,density,density]) 
    for i in range(density):
        for j in range(density):
                #eqn 3.10 for likelyhood of weights given single draw, for samples 1,2, and 20
                LikelyhoodProb[0][i][j] = \
                    norm(np.dot(np.array([[xy[j],xy[i]]]),phi[0].T),sqrt(1/Beta)).pdf(x_sample[0])  
                LikelyhoodProb[1][i][j] = \
                    norm(np.dot(np.array([[xy[j],xy[i]]]),phi[1].T),sqrt(1/Beta)).pdf(x_sample[1]) 
                LikelyhoodProb[2][i][j] = \
                    norm(np.dot(np.array([[xy[j],xy[i]]]),phi[19].T),sqrt(1/Beta)).pdf(x_sample[19])
    ax[1,0].contourf(X,Y,LikelyhoodProb[0],150,cmap = 'jet')       #plot likelyhood contours
    ax[2,0].contourf(X,Y,LikelyhoodProb[1],150,cmap = 'jet')
    ax[3,0].contourf(X,Y,LikelyhoodProb[2],150,cmap = 'jet')
    ax[1,0].plot(a0,a1,'w+')            #White crosshairs for actual weights
    ax[2,0].plot(a0,a1,'w+')
    ax[3,0].plot(a0,a1,'w+')

    #Posterior Calculation p(w|t,alpha,beta)
   
    Sn_sample1 = np.linalg.inv(alpha*np.eye(2) + Beta*np.dot(phi[:1].T,phi[:1]))    #Eqn 3.54
    Sn_sample2 = np.linalg.inv(alpha*np.eye(2) + Beta*np.dot(phi[:2].T,phi[:2]))
    Sn_sample20 = np.linalg.inv(alpha*np.eye(2) + Beta*np.dot(phi.T,phi))

    mN_sample1 = Beta*np.dot(np.dot(Sn_sample1,phi[:1].T),y_sample[:1].reshape(-1,1))   #Eqn 3.53
    mN_sample2 = Beta*np.dot(np.dot(Sn_sample2,phi[:2].T),y_sample[:2].reshape(-1,1))
    mN_sample20 = Beta*np.dot(np.dot(Sn_sample20,phi.T),y_sample.reshape(-1,1))

    #pdfs of posterior dist for recall later:
    Posteriorpdfs = [multivariate_normal(mN_sample1.flatten(),Sn_sample1),     
                     multivariate_normal(mN_sample2.flatten(),Sn_sample2),
                     multivariate_normal(mN_sample20.flatten(),Sn_sample20)]
    #initialize PosteriorProb for each mesh point and for each of the three samples:
    PosteriorProb = np.empty([3,density,density])
    for i in range(density):
        for j in range(density):
            #calculate posterior prob at every x,y coord from -1 to 1:
            PosteriorProb[0][i][j] =   Posteriorpdfs[0].pdf([xy[j],xy[i]])  
            PosteriorProb[1][i][j] =   Posteriorpdfs[1].pdf([xy[j],xy[i]])
            PosteriorProb[2][i][j] =   Posteriorpdfs[2].pdf([xy[j],xy[i]])
    ax[1,1].contourf(X,Y,PosteriorProb[0],150,cmap = 'jet') #plot posterior prob contours
    ax[2,1].contourf(X,Y,PosteriorProb[1],150,cmap = 'jet')
    ax[3,1].contourf(X,Y,PosteriorProb[2],150,cmap = 'jet')
    ax[1,1].plot(a0,a1,'w+')            #White crosshairs for actual weights
    ax[2,1].plot(a0,a1,'w+')
    ax[3,1].plot(a0,a1,'w+')


    #Data Space Point Generation

    x_data = np.array([-1,1])           #enpoints of line
    
    #generate 6 random weight vectors based on appropriate posterior dist for each sample#:
    w0random, w1random = \
    np.random.multivariate_normal(np.array([0,0]),1/alpha*np.eye(2),size = 6).T                    
    w0random_sample1, w1random_sample1 = \
    np.random.multivariate_normal(mN_sample1.flatten(),Sn_sample1,size = 6).T
    w0random_sample2, w1random_sample2 = \
    np.random.multivariate_normal(mN_sample2.flatten(),Sn_sample2,size = 6).T
    w0random_sample20, w1random_sample20 = \
    np.random.multivariate_normal(mN_sample20.flatten(),Sn_sample20,size = 6).T   

    #generate data based on eq: y = w0 + w1*x:
    y_data = np.stack((w1random*x_data[0]+w0random,w1random*x_data[1]+w0random),axis = -1)                              
    y_data_sample1 = \
        np.stack((w1random_sample1*x_data[0]+w0random_sample1,\
                  w1random_sample1*x_data[1]+w0random_sample1),axis = -1)
    y_data_sample2 = \
        np.stack((w1random_sample2*x_data[0]+w0random_sample2,\
                  w1random_sample2*x_data[1]+w0random_sample2),axis = -1)
    y_data_sample20 = \
        np.stack((w1random_sample20*x_data[0]+w0random_sample20,\
                  w1random_sample20*x_data[1]+w0random_sample20),axis = -1)
    for i in range(6):
        #plot 6 random lines with these parameters along with sample pts
        ax[0,2].plot(x_data,y_data[i],'r-') 
        ax[1,2].plot(x_data,y_data_sample1[i],'r-',x_sample[:1],y_sample[:1],'bo', mfc = 'none')
        ax[2,2].plot(x_data,y_data_sample2[i],'r-',x_sample[:2],y_sample[:2],'bo', mfc = 'none')
        ax[3,2].plot(x_data,y_data_sample20[i],'r-',x_sample,y_sample,'bo', mfc = 'none')

    #Show Plot
    plt.show()



if __name__ == '__main__':
    main()
