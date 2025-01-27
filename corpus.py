"""
This script is used to create a corpus of documents.

datatype = WIND
The corpus is saved as a binary file. 
It is useful to avoid using gensim_corpus everytime
 in LDA functions, as this takes huge amounts of time. 
N.B.: results are going to have large size, make 
sure to have lots of available space (and RAM).
"""

# %% Import packages and logging config
import xarray as xr
import logging
import numpy as np
import pickle

# Set up logging to file
logging.basicConfig(filename='corpus.log', format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)
log = logging.getLogger(__name__)

# %% Define functions needed to suitably manipulate .nc files
def digitizer(ds, step, eps, min_val=0.):
    """
    Digitize dataset.

    Dataset ds is digitized according to tolerance eps, threshold min_val 
    and step (bin_width). 
    Return type: numpy array.
    Better to first filter snapshots and load it before (no dask array).
    """
    log.debug("FUNCTION: digitizer")
    dsdigit = np.zeros_like(ds, dtype=int)  # .compute()
    ind = np.logical_and(ds >= eps, ds >= min_val)

    # if value<step but >eps and minval, digit = 1
    dsdigit[ind] = np.ceil(ds[ind]/step)
    return dsdigit


def prepare_corpus(dsdigit, nsnap=1, corpus=None):
    """
    Create gensim compatible corpus from digitized array.

    Corpus form: (list of docs as list of tuples (word_index, n_occurences))
    nsnap is the time dimension of documents: 
        =1 by default for space motifs, 
        >1 for space-time motifs (not implemented yet)
    """
    log.debug("FUNCTION: prepare_corpus")

    shape = dsdigit.shape
    if nsnap == 1:                      ### unnecessary yet
        if corpus is None:
            corpus = np.empty(shape[0], dtype=object)  # [None]*shape[0]
            append = False
        else:
            append = True
        # flatten  space dimensions
        dsdigit_flat = dsdigit.reshape([shape[0], np.prod(shape[1:])])      ### np.prod unnecessary
        space_dim = dsdigit_flat.shape[1]
        for it in range(shape[0]):
            if append:
                corpus[it] = corpus[it] + [(j+space_dim, dsdigit_flat[it, j])
                                           for j in range(space_dim) if dsdigit_flat[it, j] > 0]
            else:
                corpus[it] = [(j, dsdigit_flat[it, j])
                              for j in range(space_dim) if dsdigit_flat[it, j] > 0]
    return corpus


def gensim_corpus(dataraw, ndocs, step=None, eps=0, maxwordcount=10):
    """
    Create gensim compatible corpus from raw data. 

    Corpus form: (list of documents as list of tuples (word_index, n_occurences))
    In particular:
        a) normalize data
        b) digitize data
        c) prepare actual corpus
    """
    log.debug("FUNCTION: gensim_corpus")
    nsize = dataraw.shape[1]*dataraw.shape[2]
    datanormalized = dataraw.reshape(ndocs, nsize)

    if step is None:
        step = np.amax(dataraw)/maxwordcount

    # flatten data (reshape as vector) + digitize
    X = np.zeros([ndocs, nsize])
    X = digitizer(datanormalized, step=step, eps=eps)

    corpus = prepare_corpus(X)

    return corpus


##### MAIN #####
log.debug("NEW-EXECUTION")

# %% ACQUIRE DATA
dataset_paths = ["ERA5_1979.nc"]

for path in dataset_paths:
    ds_temp = xr.open_dataset(path)
    u_temp = ds_temp.variables['u10'].data  # x component
    v_temp = ds_temp.variables['v10'].data  # y component
    data = np.sqrt(u_temp**2 + v_temp**2)
    del u_temp, v_temp
    ds_temp.close()

ndocs = data.shape[0]
# threshold on the minimum value of wind for us to consider it
# => sparsify data
nwthreshold = 0.1

# %% Build corpus
flcorpus = gensim_corpus(data, ndocs, eps=nwthreshold, maxwordcount=20)
del data

# %% Save corpus
with open("toy_corpus", "wb") as f:
    pickle.dump(flcorpus, f)

log.debug("THE END")
