"""
This script executes LDA from gensim.

Data are assumed to have already been properly preprocessed.
See
https://radimrehurek.com/gensim/models/ldamodel.html 
for more information on lda implementation.
"""

#%% Import packages 
import logging
import numpy as np
import os
import pickle
import time as tm

from gensim import models

#%% Functions 
def setup_logger(name, log_file, level=logging.DEBUG, mode = 'a'):
    """To setup as many loggers as I want"""

    formatter = logging.Formatter('%(asctime)s : %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file, mode)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def get_doc_topic(corpus, model, eps=0):
        """
        Obtain document-topic distribution.

        We obtain a matrix of shape (ndocs,ntopics) containing
        weights attributed to topics generating each document.
        """
        ldoc_topic = list()
        for doc in corpus:
            # get document as a tuple of len 2 (topic index, probability) 
            # eps is a threshold on probability
            best_topics = model.__getitem__(doc, eps=eps)
            all_topics = [(i,0) for i in range(model.get_topics().shape[0])]
            for t in best_topics:
                all_topics[t[0]] = t
            ldoc_topic.append(all_topics)
        # keep only probability [1] and not topic index [0] and convert into numpy array
        doc_topic=np.array(ldoc_topic)[:,:,1] 
        return doc_topic

def save_model(lda, parameters, duration, pdt_gensim, path):
    """
    Save model for future usage.
    """
    # save model
    lda.save(path + '/' + 'model') 
    # get topic - word distribution 
    ptw_gensim=lda.get_topics()  
    # save distribution 
    with open(path + '/' + 'ptw-atm',"wb") as filename:
        pickle.dump(ptw_gensim, filename)
    # save document - topic distribution
    with open(path + '/' + 'pdt-atm',"wb") as filename:
        pickle.dump(pdt_gensim,filename)
    #save duration
    with open(path + '/' + 'duration',"wb") as filename:
        pickle.dump(duration,filename)
    #save parameters
    with open(path + '/' + 'parameters',"wb") as filename:
        pickle.dump(parameters,filename)

#################################################
#################################################
#################################################

##### MAIN #####

# Local logging for specific number of topics
log = setup_logger('LDA_%dtopics.log' % (ntopics),'LDA_updating/LDA_updating_%dtopics.log' % (ntopics),level=logging.DEBUG)
# General logging, shared between all executions
logtot = setup_logger('LDA','LDA_updating.log',level=logging.INFO)

tot_start_time = tm.time()

#%% ACQUIRE DATA
corpus_dir = "."
#datatype = "SLP"
datatype = "WIND"

#%% PARAMETERS
ndocstot = 0
# choose number of documents in range (0:ndocstot)
#ndocs = ndocstot # int(355.25*20) 
# define number of topics
ntopics = 25
# Seed for random initialization (setting the seed allows the results to be reproduced)
random_state = 100
# define number of documents to be iterated through for each update.
update_every = 1
# define max number of iterations on a single document
iterations = 2000
# define number of chunk processed before evaluating again perplexity
eval_every = 5 
# define number of passes/epoch on the whole corpus of documents
passes = 16
# define parameter for document-topic distribution
alpha = 1./ntopics
# define parameter for topic-word distribution
eta = 1./ntopics
# define a number between [0.5, 1] to weight what percentage of the 
# previous lambda value is forgotten when each new document is examined
decay = 0.5
# define number of documents processed at the same step 
chunksize = 1024*16
# List of the parameters
paramlist  = [('datatype',datatype), ('ndocs',ndocstot), ('ntopics',ntopics), ('random_state',random_state), 
            ('update_every',update_every), ('iterations',iterations),
            ('eval_every',eval_every), ('passes',passes), ('alpha',alpha),
            ('eta',eta), ('decay',decay), ('chunksize',chunksize),
            ('nwthreshold',0.1)]
# string to format the directory name and results filenames
strformat = f'{datatype}-144ndocs-{ntopics:03d}top-{tm.strftime("%Y%m%dd-%H%M%Sh", tm.localtime())}'
# path where directory will be created
pathfig = "."
# create directory to store the results
os.mkdir(pathfig + '/' + strformat)
log.info("NEW-EXECUTION")
log.debug(strformat)
logtot.info("NEW-EXECUTION - %s" % (strformat))

# %% Gensim LDA
pdt_gensim_tot = np.zeros((368184, ntopics))
duration = 0
#corpus_paths = [corpus_dir + name for name in sorted(os.listdir(corpus_dir))]
corpus_paths = ["toy_corpus"]
for corpus_name in corpus_paths:
    with open(corpus_name, "rb") as f:
        flcorpus = pickle.load(f)
    temp_start = tm.time()
    # if 0, then we are starting lda from beginning
    if corpus_paths.index(corpus_name)==0:
        log.debug( "Starting LDA algorithm --- %s" % (corpus_name.strip('corpus/')) )
        # TRAIN LDA model with first part of dataset
        lda = models.ldamodel.LdaModel(flcorpus, num_topics=ntopics, random_state=random_state, 
                        update_every=update_every, iterations=iterations, eval_every=eval_every, 
                        passes=passes, alpha=alpha, eta=eta, 
                        decay=decay, chunksize=chunksize)
    # if not, then we already have a model that we just need to update
    else:
        log.debug( "Starting LDA UPDATE --- %s" % (corpus_name.strip('corpus/')) )
        # RETRAIN LDA model
        lda.update(flcorpus)
    temp_duration = tm.time() - temp_start
    log.debug( "--- batch #%s: %s seconds ---" % (corpus_paths.index(corpus_name),temp_duration) )
    duration = duration + temp_duration

# %% Extract document-topic distribution    
for corpus_name in corpus_paths:
    with open(corpus_name, "rb") as f:
        flcorpus = pickle.load(f)
    ndocs = flcorpus.shape[0]
    ndocstot = ndocstot + ndocs
    pdt_gensim = get_doc_topic(flcorpus, lda)
    pdt_gensim_tot[(ndocstot-ndocs):ndocstot,:] = pdt_gensim
del flcorpus

# %% Save model
paramlist[1] = ('ndocs', ndocstot)        
# print training duration
total_duration = tm.time() - tot_start_time
log.debug( "--- TOT LDA duration: %s seconds ---" % (duration) )
log.debug("--- TOT algorithm duration: %s seconds ---" % (total_duration))
# save results
save_model(lda, paramlist, duration, pdt_gensim_tot, pathfig + '/' + strformat)
log.debug("model saved")
logtot.info("Model saved - %s \n---TOT algorithm duration: %s seconds ---" 
            % (strformat, total_duration))


# load model (if needed)
# lda = models.ldamodel.LdaModel.load(modelFilename)

#  get_topics() for topic-word distributions