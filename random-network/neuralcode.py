#!/usr/bin/python
import os
import sys
import pickle
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from collections import Counter
from sklearn import preprocessing
from scipy.stats.stats import pearsonr

import seaborn as sns
sns.set_style("white")

RandomMatrix = np.random.randn
monotonic_nonlinear = np.tanh

def Normalise(x):
    """Function to normalise and mean centre matrix.
    """
    x -= np.mean(x)
    x /= np.std(x)
    return x

def Remap(x, new_min = -1, new_max = 1):
    """Function to put matrix values in range [new_min, new_max].


    Keyword arguments:
    new_min -- default value: -1
    new_max -- default value: 1
    """
    old_min = np.amin(x)
    old_max = np.amax(x)
    old_value = x
    new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    return new_value

class NeuralCode():
  """A set of possible models of the neural code.
  """
  def __init__(self, representations_file = 'representations.pkl', save_representations = True):
    """Initialise the NeuralCode class. If no filenames provided, two random normally distributed category prototypes will be generated.


    Keyword arguments:
    representations_file -- the filename of pickle file from a previous run; default value is "representations.pkl". If the file exists, the representations matrix will be loaded from it. If provided, but file does not exist, new representations will be written to it.
    save_representations --- to save the output of the models to a pickle file; default value is True.
    """
    self.model_labels = []
    self.categories = 100 # how many categories/prototypes
    self.levels_of_noise = range(20) #0th level has no noise
    self.sd_increment = 0.05 # how much sd of noise to add per level
    self.models = 20 # the number of models we are interested in
    self.runs = 100
    self.prototypes = np.zeros((self.categories, 100, 1)) # create a matrix with Gaussian noise in it
    self.prototypes= np.random.standard_normal(self.prototypes.shape) # these are the basis for each category
    self.representations = np.zeros((self.runs, self.models, self.prototypes.shape[0], len(self.levels_of_noise), self.prototypes.shape[1] * self.prototypes.shape[2])) # where we will store the internal representations for each model
    self.save_representations = save_representations

    if not os.path.isfile(representations_file):
      self.RunModels(representations_file)
    else:
      f = open(representations_file, "rb" )
      print 'Loading representations from: ', representations_file
      self.representations, self.model_labels = pickle.load(f)
      print 'Done!'
      f.close()

  def RunModels(self, output = 'representations.pkl'):
    """ Run and save representations generated by all models.


    Keyword arguments:
    output -- name of pickle file to save the output of all the models, default value is "representations.pkl". If the file exists, it will be overwritten.
    """

    noise = np.zeros((self.prototypes.shape[0], len(self.levels_of_noise), self.prototypes.shape[1], self.prototypes.shape[2])) # initialise noise matrix to add to prototypes
    self.representations = np.zeros((self.runs, self.models, self.prototypes.shape[0], len(self.levels_of_noise), self.prototypes.shape[1] * self.prototypes.shape[2])) # where we will store the internal representations for each model

    corrx1x2 = np.empty((self.runs, self.categories, len(self.levels_of_noise)-1))
    corry1y2 = np.empty((self.runs, self.categories, len(self.levels_of_noise)-1))
    corrz1z2 = np.empty((self.runs, self.categories, len(self.levels_of_noise)-1))

    for run in range(self.runs):

        print 'Generating categories...'

        self.model_labels = [] # this was more useful in the past for some graphs, now just for terminal output

        # Create the prototypes
        self.prototypes= np.random.standard_normal(self.prototypes.shape)

        # Create the levels of noise
        for i, prototype in enumerate(self.prototypes):
          for level in self.levels_of_noise:
            noise[i, level] = np.random.standard_normal(noise[i, level].shape) * level*self.sd_increment
            if level == 0:
            #    cv2.randn(noise[i, level], 0, 0.0001) # imperceptible noise
               noise[i, 0] = np.random.standard_normal(noise[i, 0].shape) * 0.0001

        # Model 1: just pass along the pixel intensities plus the noise as a vector.
        model = 0
        self.model_labels.append('Unmodified Pixel Intensities')
        print '\tDone!\nRun ', run+1,' of ',self.runs, '\nModel', model+1,  self.model_labels[model], '\n\tRunning...'
        for i, prototype in enumerate(self.prototypes):
          for level in self.levels_of_noise:
            self.representations[run, model, i, level] = Remap(Normalise((prototype + noise[i, level]).flatten()))

            if level:
                 corrx1x2[run][i][level-1] = pearsonr(self.representations[run, model, i, 0], self.representations[run, model, i, level])[0]

        #Model 2: pass through a monotonic non-linearity.
        model += 1 # model = 1
        self.model_labels.append('Monotonic Non-linear Function')
        print '\tDone!\nRun ', run+1,' of ',self.runs, '\nModel', model+1,  self.model_labels[model], '\n\tRunning...'

        for i, prototype in enumerate(self.prototypes):
          for level in self.levels_of_noise:
            self.representations[run, model, i, level] = monotonic_nonlinear(Normalise((prototype + noise[i, level]).flatten()))

        # Model 3: Multiply by an invertible matrix.
        model += 1 # model = 2
        self.model_labels.append('Multiplication by Invertible Matrix')
        print '\tDone!\nRun ', run+1,' of ',self.runs, '\nModel', model+1,  self.model_labels[model], '\n\tRunning...'

        # I chose a matrix with normally distributed values with a mean of 0 and an SD of +/-1
        matrix = RandomMatrix(self.prototypes[0].shape[1] * self.prototypes[0].shape[0], self.prototypes[0].shape[1] * self.prototypes[0].shape[0]) # create a 100*100 matrix with Gaussian noise in it
        for i, prototype in enumerate(self.prototypes):
          for level in self.levels_of_noise:
            self.representations[run, model, i, level] = Remap(np.dot(Remap(Normalise((prototype + noise[i, level]).flatten())), matrix).flatten())#dot product of input times invertible matrix

        # Model 4: Consists of model 2 and 3 combined. Multiply by an invertible matrix then put thorugh a non-linear function.
        model += 1 # model = 3
        self.model_labels.append('Multiplication by Invertible Matrix Followed by Monotonic Non-linear Function')
        print '\tDone!\nRun ', run+1,' of ',self.runs, '\nModel', model+1,  self.model_labels[model], '\n\tRunning...'

        weights = RandomMatrix(self.prototypes[0].shape[1] * self.prototypes[0].shape[0], self.prototypes[0].shape[1] * self.prototypes[0].shape[0]) # create a matrix with Gaussian noise in it, for the weights of this random network
        print weights.shape
        for i, prototype in enumerate(self.prototypes):
          for level in self.levels_of_noise:
            self.representations[run, model, i, level] = monotonic_nonlinear(np.dot(Remap(Normalise((prototype + noise[i, level]).flatten())), matrix).flatten()) #dot product first then scaling to 0 mean and 1 SD then monotonic non-linear function then scaling again
            if level:
                 corry1y2[run][i][level-1] = pearsonr(self.representations[run, model, i, 0], self.representations[run, model, i, level])[0]
        # Models: n-layer versions of Model 4
        n_layers = 8
        for layer in range(n_layers): # this is a loop over the layers of a random networks with weights calculated below per layer
          model += 1
          self.model_labels.append(str(layer+2)+'-layer version of Model 4')
          print '\tDone!\nRun ', run+1,' of ',self.runs, '\nModel', model+1,  self.model_labels[model], '\n\tRunning...'

          weights = RandomMatrix(self.prototypes[0].shape[1] * self.prototypes[0].shape[0], self.prototypes[0].shape[1] * self.prototypes[0].shape[0]) # create a matrix with Gaussian noise in it, for the weights at this layer
          for i, prototype in enumerate(self.prototypes):
            for level in self.levels_of_noise:
              self.representations[run, model, i, level] = monotonic_nonlinear(np.dot(self.representations[run, model-1, i, level], weights)) #dot product first then monotonic non-linear function
              if level and layer == n_layers-1: # on final (8th) layer we are interested in
                  corrz1z2[run][i][level-1] = pearsonr(self.representations[run, model, i, 0], self.representations[run, model, i, level])[0]

    print '\tDone!'

    input_smoothness = pearsonr(corrx1x2.flatten(), corry1y2.flatten())
    output_smoothness = pearsonr(corrx1x2.flatten(), corrz1z2.flatten())
    print 'First layer functional smoothness:\n\trho = ', input_smoothness[0], 'p = ', input_smoothness[1]
    print '8th layer functional smoothness:\n\trho = ', output_smoothness[0], 'p = ', output_smoothness[1]

    if self.save_representations:
      print 'Saving Representations...'
      f = open(output, "wb" )
      pickle.dump([self.representations, self.model_labels], f)
      f.close()
      print 'Done!'

  def LineGraphs(self):
    line_color = range(10)
    line_color[0] = '#00441b'
    line_color[1] = '#16542F'
    line_color[2] = '#2C6543'
    line_color[3] = '#437658'
    line_color[4] = '#59866C'
    line_color[5] = '#709781'
    line_color[6] = '#86A895'
    line_color[7] = '#9DB8AA'
    line_color[8] = '#B3C9BE'

    num_x = range(10)
    num_x[0] = 10.4 # 1,2,3
    num_x[1] = 9.25 # 4
    num_x[2] = 7.8 # 5
    num_x[3] = 6.5 # 6
    num_x[4] = 5.2 # 7
    num_x[5] = 4.1 # 8
    num_x[6] = 3.27 # 9
    num_x[7] = 2.6 # 10
    num_x[8] = 1.3 # 11

    c = range(10)
    c[0] = 'w'
    c[1] = 'w'
    c[2] = 'w'
    c[3] = 'w'
    c[4] = 'w'
    c[5] = 'k'
    c[6] = 'k'
    c[7] = 'k'
    c[8] = 'k'

    slope = 0.0773722627737
    intercept = 0.0421284671533

    # which gives us the equation of the (seemingly 45ish degree) line: y = mx + b
    #that we will use further down to calculate points and move them around

    """ Create and save line graphs of correlations as .pdf and .jpg files in /fig directory. """
    #graph of correlation between matrices for each model for each noisy self.representations to the self.representations with no noise
    print 'Creating Line Graphs...'
    #plt.style.use('fivethirtyeight')
    rho_within = np.zeros((self.categories, len(self.levels_of_noise)))
    x_labels = []
    for level in self.levels_of_noise:
      x_labels.append(str(level))

      #if level == 0:
        #x_labels.append('None')
      #else:
        #x_labels.append(r'$\pm$'+str(level*self.sd_increment))
    fig = plt.figure(figsize=(6, 4)) #create a new plot per model
    ax = fig.add_subplot(111)
    ax.set_xlabel('Level of Distortion', fontsize = 14)
    ax.set_ylabel(r'Pearson Correlation Coefficient $\rho$ ', fontsize = 14)

    for m, model in enumerate(range(2, 11)):
        rho_within = np.zeros((self.categories, len(self.levels_of_noise)))

        print '\tModel:', model+1, 'of', self.models
        #now gather up the correlation coefficients per level of noise/distortion for each image
        for level in self.levels_of_noise:
            for i in range(self.categories):
                for run in range(self.runs):
                    rho_within[i, level] += pearsonr(self.representations[run, model, i, 0].flatten(), self.representations[run, model, i, level].flatten())[0] #within category correlation
                rho_within[i, level] = rho_within[i, level] / self.runs

        combined_within = np.mean(rho_within, axis =0)

        line_within = ax.plot(self.levels_of_noise, combined_within, color = 'w', lw=1, alpha = 0.1)

        ax.fill_between(self.levels_of_noise, 0, combined_within, color = line_color[m])#, alpha = 0.2)

        if m == 0:
            label = '1, 2, 3'
        else:
            label = str(m+3)

        ax.annotate(label, xy=(num_x[m], slope * num_x[m] + intercept), xycoords='data',
                      xytext=(0, -7), textcoords='offset points',
                      size= 10, va="center", ha="center",
                      color = c[m]
        )

        plt.locator_params(axis = 'x', nbins=len(self.levels_of_noise))
        ax.set_xticklabels(x_labels)#, fontsize = 12)
        ax.set_ylim([0, 1.05])
        sns.despine()

    fig.savefig('ann_models_correlation.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig('ann_models_correlation.png', format='png', bbox_inches='tight', pad_inches=0.05)

    plt.close("all")
    print 'Done!'

#------------------------------------------------------------------------------#

if __name__ == "__main__":

  Model = NeuralCode() # Initialise, and run if nothing saved is found, or load saved file
  Model.LineGraphs() # Save line graphs of the correlations within categories
