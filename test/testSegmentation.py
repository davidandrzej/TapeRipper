import unittest
import cPickle as CP
import os, os.path
import sys

sys.path.append(os.path.split(os.path.abspath(os.getcwd()))[0])
import Segmentor
import ToyData

class TestSegmentation(unittest.TestCase):
    
    def setUp(self):
        """ (no setup necessary) """
        pass

    def testToyData(self):
        """ Test track splitting on synthetic data """
        # Generate synthetic data
        testruns = [('gap',20),
                    ('track',50), # Track 1
                    ('gap',10),
                    ('track',30),('gap',3),('track',20), # Track 2
                    ('gap',15),
                    ('track',70), # Track 3
                    ('gap',10),('track',1),('gap',10)]                
        (data, labels) = (ToyData.generateData(testruns), 
                          ToyData.generateLabels(testruns))
        # Segment tracks 
        assign = Segmentor.segmentTracks(data)
        # There should be 3 tracks found 
        # (fake gap and fake track should be ignored)
        self.assert_(len(assign) == 3)

    def testSplitTracks(self):
        """ 
        Test correct track-splitting on real (albeit contrived) data
        """
        tracksubsamples = CP.load(open('dummyTrack.p'))
        assign = Segmentor.segmentTracks(tracksubsamples)
        truth = CP.load(open('trackCorrect.p'))
        self.assert_(all([a == t for (a,t) in 
                          zip(assign,truth)]))                          

    def testTrimTrack(self):
        """ 
        Test correct trimming on real (albeit contrived) data
        (assume a single track - remove leading/trailing silences)
        """
        voicesubsamples = CP.load(open('dummyVoice.p'))
        voicetrack = Segmentor.segmentVoice(voicesubsamples)[0]
        self.assert_(min(voicetrack) == 45 and max(voicetrack) == 104)
    
# Run the unit tests!
if __name__ == '__main__':
    unittest.main()
