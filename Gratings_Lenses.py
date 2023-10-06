import matplotlib.pyplot as plt
import numpy as np

def init_holo() :
	return np.zeros( ( px_x, px_y, len( spots ) ) )

def complex_holo( holo ) : 
 return np.exp( 1j * holo )

# Constants
um = 1e-6
pi = np.pi


##### Set up system paramters #####

# SLM dimensions in pixels
px_x = 512
px_y = 512

wave = 1064e-9      # incident wavelength
f    = 200e-3       # focal length of fourier lens
M    = 3 / 250      # lateral magnification of imaging system
M_z  = 1.5 * M ** 2 # axial magnification of imaging system


##### Hologram algorithm constants #####
const_grating = 2 * pi * M * um / ( wave * f )
const_lens    = pi * M_z * um / ( f ** 2 * wave )


##### spots[ x, y, z, vortex, intensity ] #####
spots = [ [ .1, .1, 0, 0, 1 ], [ -.1, .1, 0, 0, 1 ] ]


##### init holograms #####
holo_grating = init_holo()
holo_lens    = init_holo()
holo_vortex  = init_holo()

holo_sum     = init_holo()
holo_mod     = init_holo() + 1j


##### calculate holograms #####
for x in range( px_x ) : 
 for y in range( px_y ) : 
  for s in range( len( spots ) ) :
   holo_grating[ y, x, s ]	= ( spots[ s ][ 0 ] * x + spots[ s ][ 1 ] * y ) * const_grating % ( 2 * pi )
   holo_lens[ y, x, s ]		= spots[ s ][ 2 ] * ( x ** 2 + y ** 2 ) * const_lens % ( 2 * pi )
   #holo_vortex[ y, x, s ] = np.cos( spots[ s ][ 3 ] * ( np.arctan( y / ( x + 1 ) ) - x ) )

# sum holograms for single spot with all spot params applied
for s in range( len( spots ) ) :
 holo_sum[ :, :, s ] = ( holo_grating[ :, :, s ] + holo_lens[ :, :, s ] + holo_vortex[ :, :, s ] ) % ( 2 * pi )

# combine holograms for multiple spots
for s in range( len( spots ) ) :
 holo_mod[ :, :, s ] = spots[ s ][ 4 ] * complex_holo( holo_sum[ :, :, s ] )

holo_spot_sum = np.sum( holo_mod, axis = 2 )

holo_abs = abs( holo_spot_sum )
holo_abs = holo_abs / np.amax( holo_abs )

holo_comb = np.arctan2( holo_spot_sum.imag, holo_spot_sum.real )

holo_comb = holo_comb * holo_abs


# Calculate Fourier transform of grating
ft = np.fft.ifftshift( holo_comb )
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)
ft = abs( ft )


##### plot #####
#plt.subplot(1, 2, 1)
plt.imshow( holo_abs, interpolation = 'none', cmap='gray' ), plt.colorbar()
#plt.subplot(1, 2, 2)
plt.imshow( ft, interpolation = 'none', cmap='gray' ), plt.colorbar()
plt.show()

