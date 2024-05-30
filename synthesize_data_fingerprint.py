#Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Global variables for number of data points and wavenumber axis
min_wavenumber = 0.1
max_wavenumber = 2000
n_points = 640
step = (max_wavenumber-min_wavenumber)/(n_points)
wavenumber_axis = np.arange(min_wavenumber, max_wavenumber, step)
nu = np.linspace(0,1,n_points)

#Global variables for benchmarking (number of peaks and FWHM width of peaks)
#CASE 1 - 6-15cm-1 width
#CASE A - 2-3      peaks
def key_parameters(a=1,b='a'):
    if a == 1 and b == 'a':
        min_features = 1
        max_features = 15
        min_width = 2
        max_width = 10
    else:
        print('Case not defined correctly')
    return (min_features,max_features,min_width,max_width)


#Define functions for generating suseptibility
def random_parameters_for_chi3(min_features,max_features,min_width,max_width):
    """
    generates a random spectrum, without NRB. 
    output:
        params =  matrix of parameters. each row corresponds to the [amplitude, resonance, linewidth] of each generated feature (n_lor,3)
    """
    n_lor = np.random.randint(min_features,max_features+1) #the +1 was edited from bug in Paper 1.
    a = np.random.uniform(0,1,n_lor) #these will be the amplitudes of the various lorenzian function (A) and will vary between 0 and 1
    # w = np.random.uniform(min_wavenumber+300,max_wavenumber-300,n_lor) #these will be the resonance wavenumber poisitons
    # these will be the resonance wavenumber poisitons
    w = np.random.uniform(min_wavenumber+300,max_wavenumber-300,n_lor)
    g = np.random.uniform(min_width,max_width, n_lor) # and tehse are the width

    params = np.c_[a,w,g]
#    print(params)
    return params

def generate_chi3(params):
    """
    buiilds the normalized chi3 complex vector
    inputs: 
        params: (n_lor, 3)
    outputs
        chi3: complex, (n_points, )
    """
    chi3 = np.sum(params[:,0]/(-wavenumber_axis[:,np.newaxis]+params[:,1]-1j*params[:,2]),axis = 1)
    return chi3/np.max(np.abs(chi3))  

#Define functions for generating nrb
def sigmoid(x,c,b):
    return 1/(1+np.exp(-(x-c)*b))

def generate_nrb():
    """
    Produces a normalized shape for the NRB
    outputs
        NRB: (n_points,)
    """
    nu   = np.linspace(0,1,n_points)
    bs   = np.random.normal(10/max_wavenumber,5/max_wavenumber,2)
    c1   = np.random.normal(0.2*max_wavenumber,0.3*max_wavenumber)
    c2   = np.random.normal(0.7*max_wavenumber,.3*max_wavenumber)
    cs   = np.r_[c1,c2]
    sig1 = sigmoid(wavenumber_axis, cs[0], bs[0])
    sig2 = sigmoid(wavenumber_axis, cs[1], -bs[1])
    nrb  = sig1*sig2
    return nrb

#Define functions for generating bCARS spectrum 
def generate_bCARS(min_features,max_features,min_width,max_width):
    """
    Produces a cars spectrum.
    It outputs the normalized cars and the corresponding imaginary part.
    Outputs
        cars: (n_points,)
        chi3.imag: (n_points,)
    """
    chi3 = generate_chi3(random_parameters_for_chi3(min_features,max_features,min_width,max_width))*np.random.uniform(0.3,1) #add weight between .3 and 1 
    nrb = generate_nrb() #nrb will have valeus between 0 and 1
    noise = np.random.randn(n_points)*np.random.uniform(0.0005, 0.003)
    bcars = ((np.abs(chi3+nrb)**2)/2+noise)
    return bcars, chi3.imag

def generate_batch(min_features, max_features, min_width, max_width, size = 10000):
    BCARS = np.empty((size,n_points))
    RAMAN = np.empty((size,n_points))
    for i in range(size):
        BCARS[i, :], RAMAN[i, :] = generate_bCARS(min_features, max_features, min_width, max_width)
    return BCARS, RAMAN

def generate_all_data(min_features, max_features, min_width, max_width, N_train, N_valid):
    BCARS_train, RAMAN_train = generate_batch(min_features, max_features, min_width, max_width, N_train) # generate bactch for training
    BCARS_valid, RAMAN_valid = generate_batch(min_features, max_features, min_width, max_width, N_valid) # generate bactch for validation
    return BCARS_train, RAMAN_train, BCARS_valid, RAMAN_valid


#save batch to memory for training and validation - this is optional if we want to make sure the same data was used to train different methods
#it is obviously MUCH faster to generate data on the fly and not read to/write from RzOM
def generate_and_save_data(N_train,N_valid,fname='./data/',a=1,b='a'):

    (min_features,max_features,min_width,max_width) = key_parameters(a,b)

    print('min_features=',min_features,'max_features=',max_features,'min_width=',min_width,'max_width=',max_width)

    BCARS_train, RAMAN_train, BCARS_valid, RAMAN_valid = generate_all_data(min_features,max_features,min_width,max_width,N_train,N_valid)

    pd.DataFrame(RAMAN_valid).to_csv(fname+str(a)+b+'Raman_spectrums_valid.csv')
    pd.DataFrame(BCARS_valid).to_csv(fname+str(a)+b+'CARS_spectrums_valid.csv')
    pd.DataFrame(RAMAN_train).to_csv(fname+str(a)+b+'Raman_spectrums_train.csv')
    pd.DataFrame(BCARS_train).to_csv(fname+str(a)+b+'CARS_spectrums_train.csv')

    return BCARS_train, RAMAN_train, BCARS_valid, RAMAN_valid


def load_data(raman, cars):
    # load training set
    RAMAN = pd.read_csv(raman)
    BCARS = pd.read_csv(cars)

    RAMAN = RAMAN.values[:,1:]
    BCARS = BCARS.values[:,1:]

    return BCARS, RAMAN

if __name__ == '__main__':
    generate_and_save_data(N_train=30000, N_valid=6000, fname='./data03/', a=1, b='a')















