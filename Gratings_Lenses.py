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
cen_x = 0.5
cen_y = 0.5

wave = 1064e-9      # incident wavelength
f    = 200e-3       # focal length of fourier lens
M    = 3 / 250      # lateral magnification of imaging system
M_z  = 1.5 * M ** 2 # axial magnification of imaging system


##### Hologram algorithm constants #####
const_grating = 2 * pi * M * um / ( wave * f )
const_lens    = pi * M_z * um / ( f ** 2 * wave )


##### spots[ x, y, z, vortex, intensity ] #####
d = 0.4
spots = [ [ d, d, 0, 0, 1 ], [ -d, d, 0, 0, 1 ] ]


##### init holograms #####
holo_grating = init_holo()
holo_lens    = init_holo()
holo_vortex  = init_holo()
holo_sum     = init_holo()
holo_mod     = init_holo() + 1j


##### calculate holograms #####
for p in range( px_x ) :
 x = p - ( px_x * cen_x )
 for n in range( px_y ) :
  y = n - ( px_y * cen_y )
  for s in range( len( spots ) ) :
   holo_grating[ n, p, s ]	= ( spots[ s ][ 0 ] * x + spots[ s ][ 1 ] * y ) * const_grating % ( 2 * pi )
   holo_lens[ n, p, s ]		= spots[ s ][ 2 ] * ( x ** 2 + y ** 2 ) * const_lens % ( 2 * pi )
   holo_vortex[ n, p, s ]  = - spots[ s ][ 3 ] * np.arctan2( y , x ) % ( 2 * pi )

# sum holograms for single spot with all spot params applied
for s in range( len( spots ) ) :
 holo_sum[ :, :, s ] = ( holo_grating[ :, :, s ] + holo_lens[ :, :, s ] + holo_vortex[ :, :, s ] ) 

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

print( np.sum( ft))
##### plot #####
plt.subplot(2, 2, 1)
plt.imshow( holo_abs, interpolation = 'none', cmap='gray' )#, plt.colorbar()
plt.subplot(2, 2, 2)
plt.imshow( ft, interpolation = 'none', cmap='gray' )#, plt.colorbar()
plt.xlim([206, 306])
plt.ylim([306, 206])
plt.subplot(2,2,3)
plt.plot( ft[245,:])
plt.xlim([206, 306])
plt.show()
 