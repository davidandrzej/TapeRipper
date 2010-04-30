"""
TapeRipper - A GUI program for dead simple tape-to-CD transfers.

David Andrzejewski
david.andrzej@gmail.com


USAGE

Voice recording mode: resulting CD will have a single track
(cannot navigate by skipping individual songs)

Music recording mode: program will attempt to automatically 
find track breaks to create a CD with multiple tracks

1) Connect tape player to mic input and put blank CD in drive
2) Press the RECORD button
3) Press play on your cassette tape player
4) When tape is done, press STOP
5) When tracks are done, press BURN

If any problems are encountered, simply QUIT and start over

To test mic
-RECORD some of your tape as described above
-Hit PLAY to listen to the recording (use to test quality/volume/etc)
"""
import os,os.path
import re
import cPickle as CP

import numpy as NP
from Tkinter import *
import tkSnack

import Segmentor

datadir = 'data'

class TapeRipperApp:

    def __init__(self, master):
        # Initialize state to 0
        self.state = 0
        # Current track being recorded
        self.recsound = None
        # Side one (for double-sided tapes)
        self.sideone = None
        # Tk master frame
        self.frame = Frame(master)#,bg='white')
        self.frame.pack(padx=5,pady=5)        
        # Display 'burning CD' image
        img_fn = os.path.join('data','tape2cd.gif')
        burnimg = PhotoImage(file=img_fn)
        self.image = Label(self.frame,image=burnimg,justify='left',
                           borderwidth=0)
        self.image.imgref = burnimg
        self.image.grid(row=0,column=0,columnspan=3)
        # Instructions
        instructions = '\nWelcome!\n\n'
        instructions += '1) Press the RECORD button\n'
        instructions += '2) Press play on your cassette tape player\n'
        instructions += '3) When tape is done, press STOP\n'
        instructions += '4) When tracks are done, press BURN\n'
        self.instruct = Label(self.frame, text=instructions, justify='left')
        self.instruct.grid(row=1,column=0,columnspan=3)
        # Voice vs music radio button
        self.tracksplit = IntVar()
        # Voice recording, 1 side
        self.voice = Radiobutton(self.frame, text="Voice recording\n(do not split)", 
                                 variable=self.tracksplit, value=0)
        self.voice.grid(row=2,column=0)
        # (for 2-sided tape)
        # self.fliptape = Button(self.frame, text="FLIP\nTAPE",
        #                        command=self.doTapeFlip)
        # self.fliptape.grid(row=2,column=1)
        # Music recording (try to identify track breaks)
        self.music = Radiobutton(self.frame, text="Music recording\n(split tracks)",
                                 variable=self.tracksplit, value=2)
        self.music.grid(row=2,column=2)
        # Status messages
        self.status = Label(self.frame, text="")
        self.status.grid(row=3,column=0,columnspan=3)
        # RECORD / STOP / BURN button
        self.record_sound = Button(self.frame, text="RECORD", bg="green",
                                   command=self.recordSound)
        self.record_sound.grid(row=4,column=0)
        # TEST button (play back recorded sound)
        self.play_sound = Button(self.frame, text="TEST", 
                                 command=self.playSound)
        self.play_sound.grid(row=4,column=1)
        # QUIT button
        self.button = Button(self.frame, text="QUIT", 
                             command=self.cleanupQuit)
        self.button.grid(row=4,column=2)    
        # Automagically determine CD burner device ID
        scanbus = os.popen('cdrecord -scanbus')
        for line in scanbus:
            # If 'RW' appears in device description, 
            # assume this device can burn CDs
            if('RW' in line):
                m = re.search('(\d,\d,\d)',line)
                if(m == None):
                    notfounderr = 'ERROR - CD BURNER NOT FOUND'
                    self.status.config(text=notfounderr)
                else:
                    self.dev = m.group(1)
                    break

    def doTapeFlip(self):
        """ For 2-sided tape, save the first side and resume... """
        self.status.config(text='Flip tape, then play...')
        # Save side one
        self.recsound.stop()
        (sampfreq,subrate) = (44100,20000)
        samps = Segmentor.getMonoAmpSamples(self.recsound,
                                            subrate)            
        trackAssign = Segmentor.segmentVoice(samps)
        (tstart,tend) = (min(trackAssign[0])*subrate,
                         max(trackAssign[0])*subrate)
        self.sideone = tkSnack.Sound()
        self.sideone.copy(self.recsound,start=tstart,end=tend)
        # Restart recording
        self.recsound = tkSnack.Sound()
        self.recsound.configure(channels="Stereo")
        self.recsound.configure(frequency=44100)
        self.recsound.record()

    def cleanupQuit(self):
        """ Clear currently saved tracks and quit """
        tracks = [fn for fn in os.listdir(datadir)
                  if re.match('track\d+\.wav',fn)]        
        for track in tracks:
            os.remove(os.path.join(datadir,track))
        self.frame.quit()

    def recordSound(self):
        """ Start/stop recording from the mic input """        
        if(self.state == 0):
            #
            # Start recording
            #
            self.recsound = tkSnack.Sound()
            self.recsound.configure(channels="Stereo")
            self.recsound.configure(frequency=44100)
            self.recsound.record()
            self.status.config(text='Recording...')
            self.record_sound.config(text="STOP",bg="red")
            self.state = 1
        elif(self.state == 1):
            #
            # Stop recording, write tracks out to disk for burning
            # 
            self.recsound.stop()
            # Wave sampling freq (Hz)
            sampfreq = 44100 
            # Convert to mono (take max amplitude over channels) and subsample
            subrate = 20000
            samps = Segmentor.getMonoAmpSamples(self.recsound,
                                                subrate)            
            # Process subsampled audio
            self.status.config(text='Processing audio...')
            if(self.tracksplit.get() == 0):
                # VOICE: just find single track
                trackAssign = Segmentor.segmentVoice(samps)                
            elif(self.tracksplit.get() == 2):
                # MUSIC: segment recording into individual tracks
                trackAssign = Segmentor.segmentTracks(samps)
            # Write these tracks out to disk                     
            if(self.sideone != None):
                # We have a previous side-one track, write it out
                fn = os.path.join(datadir,'track%d.wav' % 0)
                self.sideone.write(fn)
                # Then write out the current track
                (tstart,tend) = (min(trackAssign[0])*subrate,
                                 max(trackAssign[0])*subrate)
                newTrack = tkSnack.Sound()
                newTrack.copy(self.recsound,start=tstart,end=tend)
                fn = os.path.join(datadir,'track%d.wav' % 1)
                newTrack.write(fn)
            else:                
                # No side-one, just write these tracks out to disk
                for (tracknum,ta) in enumerate(trackAssign):
                    (tstart,tend) = (min(ta)*subrate,max(ta)*subrate)
                    newTrack = tkSnack.Sound()
                    newTrack.copy(self.recsound,start=tstart,end=tend)
                    fn = os.path.join(datadir,'track%d.wav' % tracknum)
                    newTrack.write(fn)
            self.status.config(text='Ready to burn!')            
            self.recsound = None
            self.record_sound.config(text="BURN",bg='red')
            self.state = 2
        elif(self.state == 2):
            #
            # Burn tracks from disk to a blank CD 
            #
            tracks = [fn for fn in os.listdir(datadir)
                      if re.match('track\d+\.wav',fn)]
            def sortfn(a,b):
                aval = int(re.match('track(\d+)\.wav',a).group(1))
                bval = int(re.match('track(\d+)\.wav',b).group(1))
                return aval-bval
            tracks.sort(sortfn)        
            # Construct cdrecord command for CD burning
            cmd = 'cdrecord fs=4096k -v -useinfo speed=1 '
            cmd += '-dao -eject -pad -audio '
            cmd += '-dev=%s ' % (self.dev)
            for track in tracks:
                cmd += '\"%s\" ' % os.path.join(datadir,track)
            self.status.config(text='Burning CD...')
            os.system(cmd)
            self.status.config(text='Done - CD completed')
            # Now that we're done, cleanup by deleting temp track files
            for track in tracks:
                os.remove(os.path.join(datadir,track))
            # Reset state
            self.state = 0

    def playSound(self):
        """ Play the first track to test the recording setup """
        if(self.recsound == None):
            nosoundmsg = 'RECORD some audio first, then try to PLAY'
            self.status.config(text=nosoundmsg)
        else:
            playsoundmsg = 'Recording OK?  If not, QUIT and adjust volume.'
            self.status.config(text=playsoundmsg)
            self.recsound.play()

# Init stuff for launching the app
root = Tk()
root.title('Tape Ripper')
tkSnack.initializeSnack(root)
app = TapeRipperApp(root)
root.mainloop()
