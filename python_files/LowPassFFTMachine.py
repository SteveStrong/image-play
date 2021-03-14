import time
import FrameWrapper as fw
import os
import cvui
import numpy as np

PROJECT_DIR = os.getcwd()
OTHER_DIR = os.path.join(PROJECT_DIR, 'other_source_data')
WINDOW_NAME = "FFTMachine (Low Pass Filter w/ Gaussian Blur)"

valueTrack=[0.07]
sigmaXTrack=[11]
sigmaYTrack=[11]
Once=False

class LowPassFFTMachine():

    def __init__(self):
        self._live = True
        self._files= [f for f in os.listdir(OTHER_DIR) if os.path.isfile(os.path.join(OTHER_DIR, f)) and f.endswith( (".jpg", ".png") )]
        self._frame = np.zeros((750, 670), np.uint8)
        self._lastValue = 0
        self._sourceFile = '60-S1.png'
        self._source = fw.Frame.loadGrayscale( os.path.join(OTHER_DIR,self._sourceFile ))
        self._fft    = None
        self._magnatude = None
        self._lpf = None
        self._result = None
        self._filterDFT = fw.DFTFilter()
        self._filterShift = fw.FFTShiftFilter()
        self._filterIntensity = fw.IntensityFilter(20)
        self._filterLog = fw.LogFilter()
        self._filterMagnatude = fw.MagnatudeFilter()
        self._filterUnshift = fw.IFFTShiftFilter()
        self._filterIDFT = fw.IDFTFilter()
        self._filterLPF = fw.LowPassFilter(self._source._buffer.shape, valueTrack[0], (sigmaXTrack[0],sigmaYTrack[0]), fw.Border.ISOLATED)
        self._filterAbs = fw.AbsFilter()

    def clear(self):
        self._frame[:] = 51
    
    def processFFT(self):
        self._filterLPF.update( valueTrack[0], (sigmaXTrack[0],sigmaYTrack[0] ) )
        self._fft = self._source.filter( self._filterDFT ).filter(self._filterShift)
        self._lpf = self._fft.copy().filter( self._filterLPF )
        self._magnatude = self._lpf.copy().filter( self._filterAbs ).filter( self._filterLog ).filter( self._filterMagnatude )
        #self._magnatude.plot( 'Magnatude Spectrum', cmap='gray').show( 'Spectrum' )
        self._magnatude.show( 'Spectrum')
        
        self._result = self._lpf.copy().filter( self._filterUnshift ).filter( self._filterIDFT ).filter( self._filterMagnatude )
        self._result.plot( 'IDFT', cmap='gray' ).show( 'idft' )

    def drawUI(self):
        rVal = False
        self.clear()
        cvui.text(self._frame, 15, 10, f"File: {self._sourceFile}", 0.4, 0xffffff)

        cvui.text(self._frame, 15, 35, "Size:")
        vChange = cvui.trackbar(self._frame, 80, 35, 500, valueTrack, 0.0001, 0.2, 1, '%.4Lf' )
        cvui.text(self._frame, 15, 80, "Sigma X:")
        sxChange = cvui.trackbar(self._frame, 80, 80, 500, sigmaXTrack, 1, 99, 2, '%d', cvui.TRACKBAR_DISCRETE, 2 )
        cvui.text(self._frame, 15, 125, "Sigma Y:")
        syChange = cvui.trackbar(self._frame, 80, 125, 500, sigmaYTrack, 1, 99, 2, '%d', cvui.TRACKBAR_DISCRETE, 2 )

        if vChange:
            #print( f"size: {valueTrack}" )
            pass
        if sxChange or syChange:
            #print( f"sigmaXY: ( {sigmaXTrack}, {sigmaYTrack}" )
            pass

        #cvui.image(self._frame, 1170, 170, self._result.filter(self._filterAbs )._buffer )
        #cvui.image(self._frame, 585, 170, self._fft.filter(self._filterAbs )._buffer )
        cvui.image(self._frame, 15, 180, self._source._buffer )

        cvui.update()
        cvui.imshow( WINDOW_NAME, self._frame)

        return ( vChange or sxChange or syChange )  
    
    def start(self):
        cvui.init(WINDOW_NAME)
        
        self.processFFT()
        
        while self._live:
            #draw the user interface
            if self.drawUI():
                self.processFFT()
            
            #wait for quit
            if fw.waitKey(1) & 0xFF == ord('q'):
                self._live = False

            time.sleep(0.02)

if __name__ == "__main__":
    machine = LowPassFFTMachine()
    machine.start()