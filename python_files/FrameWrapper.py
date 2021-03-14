from __future__ import annotations
import numpy as np
from numpy.fft import fftshift, ifftshift, fft, ifft
#from scipy.fft import fft2, ifft2
import cv2
import time
import os
import io
import pandas as pd
import threading
import queue
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

#Border Constants
class Border():
    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    REFLECT_101 = cv2.BORDER_REFLECT_101
    WRAP = cv2.BORDER_WRAP
    DEFAULT = cv2.BORDER_DEFAULT
    ISOLATED = cv2.BORDER_ISOLATED

#Color Constants
class Color():
    GRY_BLACK = 0
    GRY_WHITE = 255
    CLR_BLACK = [0, 0, 0]
    CLR_WHITE = [255, 255, 255]

class StopWatch():
    def reset(self):
        self._start = 0
        self._stop = 0

    def start(self):
        self.reset()
        self._start = time.time()

    def stop(self) -> float:
        self._stop = time.time()
        return self.elapsed()

    def elapsed(self) -> float:
        rVal = 0.0
        mark = self._stop

        if mark == 0:
            mark = time.time()

        return mark - self._start


class FrameError(BaseException):
    def __init__(self, message: str = "An unexpected FrameWrapper error occurred"):
        super().__init__(message)


class Frame():
    '''
    A single frame of the video stream
    '''
    def __init__(self, buffer: np.ndarray = None, index: int = 0):
        self._buffer = buffer
        self._index = index
        self._sideline = None

    def update(self, index: int, buffer: np.ndarray):
        self._index = index
        self._buffer = buffer

    def index(self) -> int:
        return self._index

    def copy(self) -> Frame:
        rVal = Frame()
        rVal.update(self._index, self._buffer.copy())
        return rVal

    def show(self, title: str = 'Untitled', buffer: np.ndarray = None) -> Frame:
        if buffer is None:
            buffer = self._buffer
        cv2.imshow(title, buffer)
        return self
    
    def plot(self,title:str,cmap='gray'):
        fig = plt.figure()
        plt.imshow(self._buffer, cmap = cmap)
        plt.axis('off')
        ax=fig.add_subplot(1,1,1)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.tight_layout(pad=-1.0, h_pad=-1.0, w_pad=0, rect=None)
        fig.canvas.draw()
        img = np.fromstring( fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return Frame( np.asarray(img), self.index() )

    def filter(self, filter: Filter) -> Frame:
        if self._sideline is None:
            self._sideline = Frame()

        self._sideline.update(self._index, filter.process(self._buffer))

        return self._sideline

    def filterWith(self, filter: Filter, mask: Frame):
        if self._sideline is None:
            self._sideline = Frame()

        self._sideline.update(self._index, filter.process(
            self._buffer, mask._buffer))

        return self._sideline

    def add(self, other: Frame, position) -> Frame:
        h1, w1, c1 = self._buffer.shape
        h2, w2, c2 = other._buffer.shape
        x, y = position

        #print(f"{h1},{w1},{c1}  -/-  {h2}/{w2}/{c2}   -/-  {y},{x}")

        #l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

        if x >= 0 and y >= 0 and x + w2 < w1 and y + h2 < h1:
            self._buffer[y:y+h2, x:x+w2] = other._buffer
        else:
            pass
            #print("out of bounds")

        return self
    
    def save( self, filename : str ):
        cv2.imwrite( filename, self._buffer )

    def histogram(self):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self._buffer)

        # Find the median value for the primary blackbody
        hist = cv2.calcHist([self._buffer], [0], None, [max_val + 1 - min_val], [min_val, max_val + 1])
        pixel_threshold = 255
        median_index = (np.sum(hist) + 1) / 2
        current_index = 0

        for i in range(len(hist)):
            current_index += hist[i]
            if current_index > median_index:
                pixel_threshold = min_val + i
                break
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xs = np.arange(256)
        ax.plot(xs, cv2.calcHist([self._buffer], [0], None, [256], [0, 256]).ravel(), color = (0,0,1,1), label="Fullscreen Histogram", lw = 2)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize = 12)
        ax.tick_params(axis='y', left = False, labelleft = False)
        fig.tight_layout()
        ax = fig.add_subplot(111)
        xs = np.arange(256)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize = 12)
        ax.tick_params(axis='y', left = False, labelleft = False)
        fig.canvas.draw()
        img = np.fromstring( fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        return Frame( np.asarray(img), self.index() )

    @staticmethod
    def load(filePath: str, flag: int = cv2.IMREAD_UNCHANGED) -> Frame:
        rVal = Frame()
        rVal.update(0, cv2.imread(filePath, flag))
        return rVal

    @staticmethod
    def loadColor(filePath: str) -> Frame:
        return Frame.load(filePath, cv2.IMREAD_COLOR)

    @staticmethod
    def loadGrayscale(filePath: str) -> Frame:
        return Frame.load(filePath, cv2.IMREAD_GRAYSCALE)


class Primitive():
    '''static toolbox for making mask shapes'''
    @staticmethod
    def circle( shape, size: float, origin=(0, 0), filled=True, thickness=1, invert=False, channels=None) -> np.ndarray:
        '''
        Draw a primitive circle in a new image buffer using the given arguments

            shape: tuple of ints
                The shape of the array (512,640,3), for example

            size: float
                The size of the circle relative to the size of the image (0<x<1)

            origin: tuple of ints
                Where the circle is centered around, offset from the center of the image (x,y)

            filled: boolean
                If true, the circle will be filled

            thickness: int
                if not filled allows you to specify the thickness of the circle in pixels

        Returns: ndarray
             an ndarray containing the resulting image
        '''
        if len( shape ) == 2:
            height, width = shape
            channels = 1 if channels is None else channels
            if channels > 1:
                shape = ( height, width, channels )
        elif len(shape) == 3:
            height, width, channels = shape
        else:
            raise Exception( 'Invalid shape provided to Primitive.circle()' )

        min = height if height < width else width
        radius = int(min * size)
        thickness = -1 if filled else thickness
        value = 1

        if invert:
            value = 0
            buffer = np.ones(shape, np.uint8)
        else:
            buffer = np.zeros(shape, np.uint8)

        oWidth, oHeight = origin
        cHeight = int(height * 0.5  + oHeight)
        cWidth = int(width * 0.5 + oWidth)

        return cv2.circle(buffer, (cWidth,cHeight), radius, value, thickness)

    @staticmethod
    def donut( shape, size: float, ringWidth:float, origin=(0, 0), invert=False ) -> np.ndarray:
        if len( shape ) == 2:
            height, width = shape
        elif len(shape) == 3:
            height, width, color = shape
        else:
            raise Exception( 'Invalid shape provided to Primitive.circle()' )

        thickness = int((height if height < width else width ) * ringWidth)

        return Primitive.circle(shape, size, origin=origin, filled=False, thickness=thickness, invert=invert)


class Filter():
    '''
    Interface for filter instance
    '''

    def __init__(self):
        self._it = None

    def setIterator(self, iterator: FrameIterator):
        self._it = iterator

    def process(self, buffer: np.ndarray) -> np.ndarray:
        pass

class ColumnFilter(Filter):
    def __init__(self, width=1 ):
        self._width = 1
        self._mask = None
        self._history = []
        self._hpf = None

    def calibrate(self, buffer: np.ndarray ):
        if self._hpf is None:
            self._hpf = HighPassFilter( buffer.shape, 0.1, (11,11), Border.ISOLATED )
        
        #high pass filter
        img_dft = fftshift(cv2.dft(np.float32(buffer), flags=cv2.DFT_COMPLEX_OUTPUT))
        #print( f"shape: {img_dft.shape}" )
        img_hpf = self._hpf.process( img_dft )
        img_disp = 20 * np.log( cv2.magnitude( img_hpf[:,:,0], img_hpf[:,:,1] ) )
        img_idft = cv2.idft( np.fft.ifftshift( img_hpf ) )
        #img_idft = cv2.idft( np.fft.ifftshift( img_dft ) )
        buffer = cv2.magnitude( img_idft[:,:,0], img_idft[:,:,1] )
        plt.subplot(2,2,1)
        plt.imshow( img_disp, cmap='gray' )
        plt.subplot(2,2,2)
        plt.imshow( buffer, cmap='gray' )

        #rotate image so oclumns become rows
        rotated = cv2.rotate( np.float32(buffer), cv2.ROTATE_90_CLOCKWISE )

        #frame = Frame( rotated, 0 )
        #frame.plot( title="test", cmap='gray').show( 'rotated')
        
        #1D DFT
        dft = cv2.dft( rotated, flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT )
        mag = 20 * np.log( cv2.magnitude( dft[:,:,0], dft[:,:,1]) )
        inc = 0

        mag2 = cv2.rotate( np.float32(mag), cv2.ROTATE_90_COUNTERCLOCKWISE )

        plt.subplot(2,2,3)
        plt.imshow( mag2, cmap='gray' )
        x=[]
        y=[]

        for column in mag:
            sumColumn = 0
            
            for row in column:
                sumColumn = sumColumn + row

            x.append( inc )
            y.append( sumColumn )
            inc = inc + 1

        plt.subplot(2,2,4)
        plt.plot( x, y)
        plt.show()

    def detectNoise( self, buffer: np.ndarray ) -> bool:
        return True

    def process(self, buffer : np.ndarray ):
        if self.detectNoise:
            self.calibrate(buffer)

        #return self._mask * buffer
        return buffer

class LowPassFilter(Filter):
    def __init__(self, shape, size:float, sigmaXY, borderType:int=Border.DEFAULT ):
        self._make = None
        self._shape = shape
        self._borderType = borderType
        self.update( size, sigmaXY )
    
    def update(self, size: float, sigmaXY ):
        #print( f" {size}  {sigmaXY}")
        x,y = sigmaXY
        x = int(x)
        y = int(y)

        if x%2 == 0:
            x = x + 1
        if y%2 == 0:
            y = y + 1

        circle = Primitive.circle(self._shape, size, invert=True, channels=2)

        self._mask = cv2.GaussianBlur( np.float32( circle ), (x,y), self._borderType )

    def showMask(self):
        cv2.imshow( 'LPF Mask', np.float32(self._mask) )
    
    def process(self, buffer:np.ndarray) -> np.ndarray:
        return buffer * self._mask
        #return buffer

class MidPassFilter(Filter):
    def __init__(self, shape, size:float, width:float, sigmaXY, borderType : int = Border.DEFAULT ):
        self._mask = None
        self._shape = shape
        self._sigmaXY = sigmaXY
        self._borderType = borderType

        self.update( size, width )

    def update( self, size, width ):
        self._mask = cv2.GaussianBlur(np.float32(Primitive.donut( self._shape, size, width, invert=True )),self._sigmaXY, self._borderType)

    def showMask(self):
        cv2.imshow( 'MPF Mask', np.float32(self._mask) )
    
    def process(self, buffer:np.ndarray) -> np.ndarray:
        return buffer * self._mask

class HighPassFilter(Filter):
    
    def __init__(self, shape, size:float, sigmaXY, borderType:int = Border.DEFAULT ):
        self._mask = None
        self._shape = shape
        self._sigmaXY = sigmaXY
        self._borderType = borderType

        self.update( size )
    
    def update(self, size: float ):
        self._mask = cv2.GaussianBlur(np.float32( Primitive.circle( self._shape, size, channels=2 )), self._sigmaXY, self._borderType)

    def showMask(self):
        cv2.imshow( 'HPF Mask', np.float32(self._mask) )
    
    def process(self, buffer:np.ndarray) -> np.ndarray:
        return buffer * self._mask

class BilateralFilter(Filter):
    '''
    Bilateral filter blur
    '''

    def __init__(self, diameter: int, sigmaColor: float, sigmaSpace: float):
        self._diameter = diameter
        self._sigmaColor = sigmaColor
        self._sigmaSpace = sigmaSpace

    def process(self, buffer: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(buffer, self._diameter, self._sigmaColor, self._sigmaSpace)

class SubtractionFilter(Filter):
    '''
    A subtraction filter
    '''
    def process(self, buffer: np.ndarray, last: np.ndarray) -> np.ndarray:
        return last - buffer


class MinusFilter(Filter):
    def __init__(self, value):
        self._value = value

    def process(self,buffer:np.ndarray) -> np.ndarray:
        return buffer - self._value

class BitwiseAndFilter(Filter):
    def process(self, buffer: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return buffer & mask

class BitwiseNotFilter(Filter):
    def process(self, buffer: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return buffer &~ mask

class BitwiseOrFilter(Filter):
    def process(self, buffer: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return buffer | mask

class BitwiseXorFilter(Filter):
    def process(self, buffer: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return buffer ^ mask

class TemporalFilter(Filter):
    def __init__(self, filter : Filter ):
        self._last = None
        self._filter = filter

    def process(self, buffer: np.ndaray):
        if self._last is None:
            self._last = buffer 
        
        rVal = self._filter.process( buffer, self._last )
        
        self._last = buffer

        return rVal

class DFTFilter(Filter):
    '''
    A 2-dimensional Fast Fourier Transform (FFT) Filter
    '''

    def __init__(self, s=None, axes=None, norm: str = None, overwrite_x: bool = False, workers: int = None):
        '''
        A FFT filter that uses the numpy.fft.fft2 function to perform 2-dimensional
        FFT processing.

        x: array_like
            Input array, can be complex (internally provided)

        s: sequence of ints, optional
            Shape (length of each transformed axis) of the output (s[0] refers
            to axis 0, s[1] to axis 1, etc.). This corresponds to n for
            fft(x, n). Along each axis, if the given shape is smaller than
            that of the input, the input is cropped. If it is larger, the
            input is padded with zeros. if s is not given, the shape of the
            input along the axes specified by axes is used.

        axes: sequence of ints, optional
            Axes over which to compute the FFT. If not given, the last two
            axes sare used.

        norm: {“backward”, “ortho”, “forward”}, optional

        overwrite_x: bool, optional
            If True, the contents of x can be destroyed; the default is False.
            See fft for more details.

        workers: int, optional
            Maximum number of workers to use for parallel computation. If
            negative, the value wraps around from os.cpu_count(). See fft for
            more details.

        plan: object, optional
            This argument is reserved for passing in a precomputed plan
            provided by downstream FFT vendors. It is currently not used in
            SciPy.

    Returns

        out: complex ndarray
            The truncated or zero-padded input, transformed along the axes
            indicated by axes, or the last two axes if axes is not given.
        '''
        self._s = s
        self._axes = axes
        self._norm = norm
        self._overwrite_x = overwrite_x
        self._workers = workers

    def process(self, buffer: np.ndarray) -> np.ndarray:
        #print( "dft" )
        return cv2.dft( np.float32( buffer ), flags=cv2.DFT_COMPLEX_OUTPUT)

class IDFTFilter(Filter):
    '''
    A 2-dimensional Fast Fourier Transform (FFT) Filter
    '''
    def __init__(self, s=None, axes=None, norm: str = None, overwrite_x: bool = False, workers: int = None):
        '''
        A FFT filter that uses the numpy.fft.fft2 function to perform 2-dimensional
        FFT processing.

        s: sequence of ints, optional
            Shape (length of each transformed axis) of the output (s[0] refers
                   to axis 0, s[1] to axis 1, etc.).
            This corresponds to n for fft(x, n). Along each axis, if the given
            shape is smaller than that of the input, the input is cropped. If
            it is larger, the input is padded with zeros. if s is not given,
            the shape of the input along the axes specified by axes is used.
        axes: sequence of ints, optional
            Axes over which to compute the FFT. If not given, the last two
            axes are used. A repeated index in axes means the transform over
            that axis is performed multiple times. A one-element sequence
            means that a one-dimensional FFT is performed.
        norm: {“backward”, “ortho”, “forward”}, optional
            Normalization mode (see numpy.fft). Default is “backward”.
            Indicates which direction of the forward/backward pair of
            transforms is scaled and with what normalization factor.
        magnatude: float
            Maganafies the intensity of the FFT result
        '''
        self._s = s
        self._axes = axes
        self._norm = norm
        self._overwrite_x = overwrite_x
        self._workers = workers

    def process(self, buffer: np.ndarray) -> np.ndarray:
        #return ifft2(buffer)
        #return ifft2(buffer, s=self._s, axes=self._axes, norm=self._norm ) #, overwrite_x=self._overwrite_x, workers=self._workers)
        #print( "idft" )
        return cv2.idft( buffer )

class FFTShiftFilter(Filter):
    '''
    Shift the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes listed (defaults to all).
    Note that y[0] is the Nyquist component only if len(x) is even.
    '''

    def __init__(self, axes=None):
        '''
        Parameters
            axes: int or shape tuple, optional
                Axes over which to shift. Default is None, which shifts all
                axes.
        '''
        self._axes = axes

    def process(self, buffer: np.ndarray) -> np.ndarray:
        #print( "shift" )
        return fftshift(buffer, self._axes)


class IFFTShiftFilter(Filter):
    '''
    The inverse of fftshift. Although identical for even-length x, the
    functions differ by one sample for odd-length x.
    '''

    def __init__(self, axes=None):
        '''
        Parameters

        axes: int or shape tuple, optional
              Axes over which to calculate. Defaults to None, which shifts all
              axes.
        '''
        self._axes = axes

    def process(self, buffer: np.ndarray) -> np.ndarray:
        #print( "unshift" )
        return ifftshift(buffer, self._axes)


class AbsFilter(Filter):
    def process(self, buffer: np.ndarray) -> np.ndarray:
        #print("abs")
        return np.abs(buffer)


class LogFilter(Filter):
    def process(self, buffer: np.ndarray) -> np.ndarray:
        #print( "log" )
        return np.log(buffer)

class MagnatudeFilter(Filter):
    def process(self, buffer: np.ndarray) -> np.ndarray:
        return cv2.magnitude( buffer[:,:,0], buffer[:,:,1])

class IntensityFilter(Filter):
    def __init__(self, factor=20):
        self._factor = factor

    def process(self, buffer: np.ndarray) -> np.ndarray:
        #print( "inten" )
        return buffer * self._factor

class FrameIterator():
    '''
    Class that handles the frame iteration of a video stream
    '''

    def __init__(self, source: FrameSource):
        self._watch = StopWatch()
        self._sources = []
        self._frames = []
        self._count = []

        self.addSource(source)

    def __iter__(self) -> FrameIterator:
        self._watch.start()

        # rewind streams
        for source in self._sources:
            source._stream.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # reset frame indices
        for frame in self._frames:
            frame.update(0, None)

        return self

    def __next__(self) -> Frame:
        anySuccess = False
        inc = 0
        for i, source in enumerate(self._sources):
            if source.match():
                while self.fps(i) > source.getFrameRate():
                    time.sleep(0.005)
                    inc += 1
                    if inc > 10:
                        break

            if self._frames[i].index() + 1 < self._count[i]:
                success, buffer = source.grab()
                if success:
                    if not anySuccess:
                        anySuccess = True
                    self._frames[i].update(self._frames[i].index() + 1, buffer)

        if not anySuccess:
            self._watch.stop()
            raise StopIteration

        return tuple(self._frames)

    def addSource(self, source: FrameSource) -> FrameIterator:
        self._sources.append(source)
        self._frames.append(Frame())
        self._count.append(source.getFrameCount())
        return self

    def elapsed(self) -> float:
        return self._watch.elapsed()

    def fps(self, sourceIdx: int = 0) -> float:
        rVal = 0
        elapsed = self.elapsed()

        if elapsed > 0.0:
            rVal = self._frames[sourceIdx].index() / elapsed

        return rVal

    def getSources(self):
        return self._sources

    def getSource(self, index: int = 0):
        return self._sources[index]


class FrameSource():
    '''
    Class that wrappers a OpenCV VideoCapture instance
    '''

    def __init__(self, locator: str | int, matchFPS: bool = False):
        self._stream = cv2.VideoCapture(locator)
        self._iterator = FrameIterator(self)
        self._match = matchFPS
        self._rate = -1
        self._dim = None

    def __del__(self):
        if self._stream is not None:
            self.finish()

    def getDimensions(self):
        if self._dim is None:
            self._dim = (int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return self._dim

    def getFrameRate(self) -> float:
        if(self._rate == -1):
            self._rate = float(self._stream.get(cv2.CAP_PROP_FPS))
        return self._rate

    def getFrameCount(self) -> int:
        return self._stream.get(cv2.CAP_PROP_FRAME_COUNT)

    def iterator(self) -> FrameIterator:
        return self._iterator

    def grab(self):
        return self._stream.read()

    def finish(self):
        self._stream.release()
        self._stream = None

    def match(self, flag: None | bool = None) -> bool:
        if type(flag) is bool:
            self._match = flag
        return self._match

class FrameWriter():
    def __init__(self, filename: string, fourcc: int, fps: int, frameSize):
        self._stream = cv2.VideoWriter(filename, fourcc, fps, frameSize)

    def __del__(self):
        if self._stream is not None:
            self.finish()

    def write(self, frame: Frame):
        self._stream.write(frame._buffer)

    def finish(self):
        self._stream.release()
        self._stream = None

    @staticmethod
    def makeFromSource(fileName: str, fourcc: int, source: FrameSource):
        return FrameWriter(fileName, fourcc, source.getFrameRate(), source.getDimensions())

    @staticmethod
    def fourcc(c1: str, c2: str, c3: str, c4: str) -> int:
        return cv2.VideoWriter_fourcc(c1, c2, c3, c4)

class RawReader():
    HEADER_LENGTH=2048
    FILE_DATA_BEGIN='FILE_DATA_BEGIN'
    FILE_DATE_TIME='FILE_DATE_TIME'
    FILE_NUMERICAL_FORMAT='FILE_NUMERICAL_FORMAT'
    FILE_BYTE_ORDERING='FILE_BYTE_ORDERING'
    FRAME_HEADER_SIZE='FRAME_HEADER_SIZE'
    META_HEADER_SIZE='META_HEADER_SIZE'
    CAMERA_FRAME_RATE='CAMERA_FRAME_RATE'
    CAMERA_BIT_DEPTH='CAMERA_BIT_DEPTH'
    IMAGE_HEIGHT='IMAGE_HEIGHT'
    IMAGE_WIDTH='IMAGE_WIDTH'
    IMAGE_ADJUST_RESOLUTION='IMAGE_ADJUST_RESOLUTION'
    IMAGE_MAX_PIXEL_VALUE='IMAGE_MAX_PIXEL_VALUE'
    IMAGE_MIN_PIXEL_VALUE='IMAGE_MIN_PIXEL_VALUE'
    POLARIMETER_SP_NUC='POLARIMETER_SP_NUC'
    POLARIMETER_TP_NUC='POLARIMETER_TP_NUC'

    NORM_TC=1
    NORM_S1=2

    def __init__(self, streams, normalize):
        self._streams = streams
        self._filePath = None
        self._file = None
        self._index = 0
        self._fps = None
        self._fpsNow = 0
        self._frameSize = None
        self._header = {}
        self._shape = None
        self._version = 0
        self._matchFPS = True
        self._pixelSize=4
        self._numberType=None
        self._watch = StopWatch()
        self._frameHeaderSize = 0
        self._frames = []
        self._fileSize = 0
        self._stats = ()
        self._normArgs = ( 0.7, -0.10, 1.0, np.dtype(np.float32) )

        for i in range(0,streams):
            self._frames.append( Frame() )

    def __del__(self):
        if self._file is not None:
            self._file.close()
    
    def __iter__(self):
        if self._file is None:
            raise Exception( 'RawReader.__iter__(): No file has been opened' )
        else:
            self._file.seek( self.HEADER_LENGTH, 0 ) #SEEK_SET
            self._index = 0
            self._watch.start()
        return self
    
    def __next__(self):
        rVal = []
        inc = 0

        #update internal fps tracker
        mark = self._watch.elapsed()

        if self._file is None:
            raise StopIteration
        else:
            if self._matchFPS:
                self._fpsNow = 0 if mark == 0 else self._index / mark
                while self._fpsNow > self._fps:
                    time.sleep(0.005)
                    inc += 1
                    if inc > 10:
                        break
                    mark = self._watch.elapsed()
                    self._fpsNow = 0 if mark == 0 else self._index / mark

            if self._frameHeaderSize > 0:
                self._file.seek( self._frameHeaderSize, 1 ) #SEEK_CUR, seek to start of frame data

            for i in range(0, self._streams ):
                buffer = self._file.read( self._frameSize ) #read the frame

                if buffer == -1:
                    raise StopIteration
                elif len(buffer) < self._frameSize:
                    #print( "Off by: " + str(len( buffer )))
                    raise StopIteration
                else:
                    if buffer == -1:
                        raise StopIteration
                    elif len(buffer) < self._frameSize:
                        #print( "Off by: " + str(len( buffer )))
                        raise StopIteration
                    else:
                        if self._pixelSize == 4:
                            self._frames[i].update( self._index, self.normalizeTC( np.frombuffer( buffer, dtype=np.float32 ).reshape( self._shape ) ) )
                        else:
                            raise Exception( f"RawReader::__next__(): unknown support pixel number value ( {self._pixelSize} )" )

                    self._index += 1

        return tuple(self._frames)

    def normalizeS1( self, buffer : np.ndarray ) -> np.ndarray:
        ceiling, floor, factor, dtype = self._normArgs
        min, spread = self._stats
        return (((( buffer - min ) * 100) % 1 - 0.1) * factor).astype(dtype)

    def normalizeTC( self, buffer : np.ndarray ) -> np.ndarray:
        ceiling, floor, factor, dtype = self._normArgs
        min, spread = self._stats
        return (((( buffer - min ) * spread) + floor ) * factor).astype(dtype)

    def buildDictionary(self, infoBuf ):
        fields = infoBuf.split("\r\n")
        for field in fields:
            keyValue = field.split("=")

            if len( keyValue ) == 2:
                key,value = keyValue
                
                if any(v.isdigit() for v in value):
                    if value.find('.') != -1:
                        try:
                            value = float( value )
                        except ValueError:
                            value = value
                    else:
                        try:
                            value = int(value)
                        except ValueError:
                            value = value
                    self._header[key] = value
                else:
                    self._header[key]=value

    def readHeader(self):
        self._file.seek(0,0)
        infoBuf = self._file.read( self.HEADER_LENGTH ).decode( 'utf-8' )

        if infoBuf == -1 or len(infoBuf) < self.HEADER_LENGTH:
            raise Exception( 'RawReader.readHeader(): invalid header detected (2)')

        self.buildDictionary( infoBuf )

        #dimensions        
        height = self.getValue( self.IMAGE_HEIGHT )
        width  = self.getValue( self.IMAGE_WIDTH )

        #Parameterize Reader
        self._leader = self.getValue(self.FILE_DATA_BEGIN)
        #self._pixelSize = int(self.getValue(self.CAMERA_BIT_DEPTH) / 8 ) + 1
        self._pixelSize = 4
        self._frameSize = height * width * self._pixelSize
        self._shape = (height,width)
        self._numberType = self.getValue( self.FILE_NUMERICAL_FORMAT )
        self._frameHeaderSize = self.getValue( self.FRAME_HEADER_SIZE )
        self._fps = int( self.getValue( self.CAMERA_FRAME_RATE ) / self._streams)

        #get file size and number of frames
        self._file.seek(0,2) #SEEK_END
        self._fileSize = self._file.tell()
        self._frameCount = ( self._fileSize - 2048 ) / (self._frameHeaderSize + ( self._frameSize * self._streams ))

    def open(self, filePath, matchFPS : bool=True ) -> RawReader:
        if not os.path.isfile( filePath ):
            raise Exception( f"RawReader.open(): no such file {filePath}" )

        self._file = open( filePath, 'rb' )

        #read the header
        self.readHeader()

        #move file pointer to first frame
        self._file.seek(self._leader,0)

        #save preference and file location
        self._filePath = filePath
        self._matchFPS = matchFPS

        self.calcStats()

    def calcStats(self):
        ceiling, floor, factor, dtype = self._normArgs
        count = 0
        min = None
        max = None
        while count < self._frameCount:
            if self._frameHeaderSize > 0:
                self._file.seek( self._frameHeaderSize, 1 ) #SEEK_CUR, seek to start of frame data
            buffer = np.frombuffer( self._file.read( self._frameSize ), dtype=np.float32 ).reshape( self._shape )
            amin = np.amin( buffer )
            amax = np.amax( buffer )
            min = amin if min is None or amin < min else min
            max = amax if max is None or amax > max else max
            count = count + 1

        range = max-min
        self._stats = ( min, ceiling / range )

        #debug 
        #print( f"min: {min}  max: {max}  range: {range}")

        #rewind
        self._file.seek(self._leader,0)

    def getValue(self, key : str ):
        rVal = None
        if key in self._header:
            rVal = self._header[ key ]
        return rVal

    def close(self):
        if self._file is not None:
            self._file.close()
            self._header = None
        else:
            raise Exception( "RawReader.close(): file is not open" )

class PipelineWorker(threading.Thread):
    SLEEP=0.02
    def __init__(self, tid:int, pipeline ):
        threading.Thread.__init__(self)
        self._pipeline = pipeline

    def run(self):
        workQueue = self._pipeline.getWorkQueue()
        doneQueue = self._pipeline.getDoneQueue()

        while self._pipeline.live():
            newWork = None
            doneWork = None

            if not workQueue.empty():
                buffer, filterIndex, callback, data = workQueue.get()
                filter = self._pipeline._filters[ filterIndex ]

                #print( f"[3] Filter: {filter.__class__.__name__} file: {data}")

                buffer = filter.process( buffer )

                if filterIndex + 1 < len( self._pipeline._filters ):
                    newWork = ( buffer, filterIndex+1, callback, data )
                else:
                    doneWork = ( buffer, callback, data )

                workQueue.task_done()

            if newWork is not None:
                workQueue.put( newWork )
            if doneWork is not None:
                doneQueue.put( doneWork )
                
            time.sleep(PipelineWorker.SLEEP)

class FilterPipeline():
    def __init__(self, workers=6):
        self._work = queue.Queue(0)
        self._done = queue.Queue(0)
        self._filters=[]
        self._workers=[]
        self._live=True
        self._frame = Frame()

        for i in range(0,workers):
            worker = PipelineWorker( i+1, self )
            self._workers.append( worker )
            worker.start()

    def addFilter( self, filter : Filter ) -> FilterPipeline:
        self._filters.append( filter )

    def filter( self, frame : fw.Frame, callback ):
        self._work.put( ( frame, 0, callback ) )

    def hasWork(self) -> bool:
        return not self._work.empty()

    def addWork(self, frame : Frame, callback, data=None ):
        #print( f"adding... {data}" )
        self._work.put( (frame._buffer, 0, callback, data ))

    def live(self):
        return self._live

    def join(self):
        self._live = False

        for thread in self._workers:
            thread.join()

    def getWorkQueue(self):
        return self._work
    
    def getDoneQueue(self):
        return self._done

    def poll(self):
        rVal = not ( self._done.empty() and self._work.empty() )
        
        if not self._done.empty():
            (buffer, callback, data ) = self._done.get()
            self._frame.update( 0, buffer)
            callback( self._frame, data )
            self._done.task_done()

        return rVal

class Export():
    
    @staticmethod
    def saveSlide( frame : Frame, data ):
        if type(data) is str:
            print( f"Saving .... {data}" )
            frame = frame.plot( data, cmap='gray' )
            frame.save( data )

    @staticmethod
    def showSlide( frame : Frame, data ):
        pass
        #frame._buffer = np.uint8( frame._buffer )
        #frame.plot('move').show( 'movie' )

    @staticmethod
    def slideshow_special( iterator, folderPath="./slideshow", prefix="frame" ):
        for frame, in iterator:
            padded = f"{frame.index()}".rjust(5, '0');
            frame.save( f"{folderPath}/{prefix}-{padded}.tif" ) 
            print(padded)


    @staticmethod
    def slideshow( iterator, folderPath="./slideshow", pipeline : FilterPipeline=None ):
        for frame, in iterator:
            file = f"{folderPath}/frame-{frame.index()}.tif"
            frame._buffer = np.float32( frame._buffer )
            if pipeline is None:
                Export.saveSlide( frame, file ) 
            else:
                pipeline.addWork( frame, Export.saveSlide, file )

def closeWindows():
    cv2.destroyAllWindows()

def waitKey(delayMS: int):
    return cv2.waitKey(delayMS)
