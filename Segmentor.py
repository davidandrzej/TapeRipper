"""
TapeRipper - A GUI program for dead simple tape-to-CD transfers.

David Andrzejewski
david.andrzej@gmail.com


Functions for segmenting the recorded audio sample into one or more
'tracks' (actual song/voice, plus some silence 'padding')

For multi-track recordings (a music tape of multiple songs with pauses
inbetween), segment into separate music tracks by looking for
contiguous regions of non-slicence interspersed with continugous
regions of near silence.

For single-track recordings (spoken word, or multiple songs for which
multi-track mode fails to find the correct segmentation), just 
trim the intro / outro silence.
"""
import numpy as NP

def segmentVoice(data):
    """
    Given sample amplitudes, simply define a single track with
    beginning / ending silence clipped off
    
    Return one-element List containing subsample indices for this track    
    """
    D = len(data)

    # Apply Gaussian smoothing kernel    
    W = 7
    kernel = NP.array([NP.exp(-(i-(W-1)/2)**2 / (W/2)) for i in range(W)])
    kernel = kernel / kernel.sum()
    cdata = NP.convolve(data,kernel,'same')

    # Find FIRST subsample w/ amplitude > max / 2, then take <padding> 
    # subsamples before that (or first subsample)
    pad = 6
    start = max(0,NP.where(cdata > 0.5 * cdata.max())[0].min() - pad)

    # Find LAST subsample w/ amplitude > max / 2, then take <padding> 
    # subsamples after that (or last subsample)
    finish = min(D-1,NP.where(cdata > 0.5 * cdata.max())[0].max() + pad)

    return [range(start,finish+1)]

def segmentTracks(data):
    """ 
    Given sample amplitudes, segment into individual tracks (split on
    gaps, enforce that all contiguous sequences be >= minrun)

    To do this, we define a 2-state HMM with hand-coded parameters
    and then simply compute the Viterbi path for the observed data

    Return List containing subsample indices for each track
    """
    D = len(data)

    # Pre-process data with Gaussian smoothing kernel
    W = 7
    kernel = NP.array([NP.exp(-(i-(W-1)/2)**2 / (W/2)) for i in range(W)])
    kernel = kernel / kernel.sum()
    cdata = NP.convolve(data,kernel,'same')

    # Initial distribution over states
    # (start recording before music begins)
    initprob = NP.array([0.999,0.001])

    # State transition parameters 
    # (prefer contiguous track/gap regions)
    transition = NP.array([[0.99,0.01],[0.01,0.99]])

    # Mean and std dev for each state's Gaussian emission model
    # state 0 = gap (near silence)
    # state 1 = track (use dataset mean)
    stateparam = [(0,data.std()), 
                  (data.mean(),data.std())]

    # Get state sequence of Viterbi decoding
    # (state assignments for each subsample)
    assign = viterbiDecoding(data,initprob,transition,stateparam)

    # Split each pair of tracks in the middle of the separating gap
    # (List of subsample indices for each track)
    return splitTracks(assign)

def splitTracks(assign,padding=6):
    """
    Partition all samples among tracks, padding each
    track with <padding> gap-assigned sample(s)

    Return List of track subsample index Lists where each track
    List contains subsample indices assigned to that track
    """
    tracks = []    
    # Detect contiguous same-label runs 
    # (run starts and lengths)
    (runStarts,runLengths) = detectRuns(assign)
    endIndex = runStarts[-1] + runLengths[-1] - 1
    # If we only have a single run, 
    # assume it is a single track
    if(len(runStarts) == 0):
        return [range(runStarts[0],
                      runStarts[0] + runLengths[0])]
    # Otherwise, is the 1st run track or gap?
    if(assign[runStarts[0]] == 1):
        # Track
        curTrack = range(runStarts[0],
                         min(runStarts[0] + runLengths[0] + padding, endIndex))
        curRun = 2
    else:
        # Gap
        curTrack = range(max(runStarts[0] + runLengths[0] - padding, 0),
                         min(runStarts[1] + runLengths[1] + padding, endIndex))
        curRun = 3
    tracks.append(curTrack)
    # Find rest of tracks
    while(curRun < len(runStarts)):
        # Take half the previous gap, the track,
        # then half the next gap (if it exists)
        curTrack = range(
            max(runStarts[curRun-1] + runLengths[curRun-1] - padding, 0),
            runStarts[curRun] + runLengths[curRun])
        if(curRun < len(runStarts) - 1):
            curTrack += range(runStarts[curRun+1],
                              min(runStarts[curRun+1] + padding, endIndex))
        tracks.append(curTrack)
        # Jump by 2 to next track (we know label pattern *must* alternate)
        curRun += 2
    return tracks

def detectRuns(assign):
    """ 
    Detect contiguous runs of each label
    
    Return indices and lengths of runs    
    """
    runStarts = [0] # 1st label begins the 1st run
    for i in range(len(assign)-1):
        if(assign[i] != assign[i+1]):
            runStarts.append(i+1)
    runLengths = []
    for gs in runStarts:
        tmp = 0
        while(gs+tmp < len(assign) and
              assign[gs+tmp] == assign[gs]):              
            tmp += 1
        runLengths.append(tmp)
    return (runStarts,runLengths)

def getMonoAmpSamples(sound,subrate):
    """
    Take the maximum absolute amplitude over each stereo
    channel, subsampling every subrate samples
    """
    nsamp = sound.info()[0] / subrate
    (lefts,rights) = (NP.zeros((nsamp,)),NP.zeros((nsamp,)))
    for i in range(nsamp):
        (lefts[i],rights[i]) = [int(val) for val 
                                in sound.sample(i*subrate).split()]
    lefts = NP.abs(lefts)
    rights = NP.abs(rights)
    return NP.maximum(lefts,rights)

def viterbiDecoding(data, initprob, transition, stateparam):
    """
    Given HMM parameters, return most probable 
    hidden state sequence (the viterbi decoding)

    data -- observed data (dim T)
    initprob -- initial distribution over states
    transition -- state transition matrix (dim S x S)
    stateparam -- state likelihood parameters (mu,sigma)

    Return value is NP array of hidden state assignments
    """       
    (T,S) = (len(data),len(stateparam))

    # Log-likes for each state-observation pair (dim S x T)
    loglikes = NP.zeros((S,T))
    for (si,(mu,sigma)) in enumerate(stateparam):
        loglikes[si,:] = logGauss(data,mu,sigma).T

    # Log transition probabilities (dim S x S)
    logtransition = NP.log(transition)

    # maxll_it = max loglike of all length-t seqs 
    # which are in state i at time t (dim S x T)
    maxll = NP.zeros((S,T))

    # trackback pointers (dim S x T)
    # (for maxll_it, which state did we 'come from' at time t-1?)
    trackback =  NP.zeros((S,T), NP.int)

    # Init from state 0
    maxll[:,0] = NP.log(initprob) + loglikes[:,0]

    # Calc maxll over t=1...T
    for t in range(1,T):
        # loglikes of being in each prev state, transitioning
        # into this one, and generating the observed data
        prevtrans =  NP.reshape(maxll[:,t-1],(S,1)) + logtransition
        # For each column (state at time t), keep trackback 
        # pointer of argmax over rows (states at time t-1) 
        trackback[:,t] = prevtrans.argmax(axis=0)
        # Use max of prevstate + transitions to calc max 
        # loglike of being in each state at time t
        maxll[:,t] = prevtrans.max(axis=0) + loglikes[:,t]

    # Start from the end and follow trackback pointers
    # back to the beginning to get the most probable 
    # hidden state sequence  (ie, the viterbi decoding/parse)
    return followTrackback(NP.argmax(maxll[:,-1]),trackback)

def followTrackback(finalstate,trackback):
    """ 
    Follow trackback matrix from most likely final 
    state to construct most likely state sequence
    """
    # Build up in reverse chronological order
    stateseq =  [finalstate]
    for backstep in range(2,trackback.shape[1] + 1):
        stateseq.append(trackback[stateseq[-1], -backstep])
    # Return result in chronological order
    stateseq.reverse()
    return stateseq

def logGauss(X,mu,sigma):
    """ Gaussian log-likelihood of x | mu,sigma """
    ll = -0.5 * NP.log(2 * NP.pi * sigma**2)
    ll -= NP.power(X-mu,2) / (2 * sigma**2)
    return ll
