import os
import numpy as np
import cv2
import requests
import sys

from PIL import Image
from io import BytesIO
from matplotlib import pyplot

#
# manTraNet_root = './'
# manTraNet_srcDir = os.path.join( manTraNet_root, 'src' )
# sys.path.insert( 0, manTraNet_srcDir )
# manTraNet_modelDir = os.path.join( './pretrained_weights' )
#
# manTraNet_dataDir = os.path.join( manTraNet_root, 'data' )
# sample_file = os.path.join( manTraNet_dataDir, 'samplePairs.csv' )
# assert os.path.isfile( sample_file ), "ERROR: can NOT find sample data, check `manTraNet_root`"
# with open( sample_file ) as IN :
#     sample_pairs = [line.strip().split(',') for line in IN.readlines() ]
# L = len(sample_pairs)


# def get_a_random_pair() :
#     idx = np.random.randint(0,L)
#     return ( os.path.join( manTraNet_dataDir, this ) for this in sample_pairs[idx] )


import modelCore
manTraNet = modelCore.load_pretrain_model_by_index( 4, './mantra_model')

from datetime import datetime
def read_rgb_image( image_file ) :
    rgb = cv2.imread( image_file, 1 )[...,::-1]
    return rgb

def decode_an_image_array( rgb, manTraNet, dn=1 ) :
    x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )[:,::dn,::dn]
    t0 = datetime.now()
    y = manTraNet.predict(x)[0,...,0]
    t1 = datetime.now()
    return y, t1-t0

def decode_an_image_file( image_file, manTraNet, dn=1 ) :
    rgb = read_rgb_image( image_file )
    mask, ptime = decode_an_image_array( rgb, manTraNet, dn )
    return rgb[::dn,::dn], mask, ptime.total_seconds()

def get_image_from_url(url, xrange=None, yrange=None, dn=1) :
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    if img.shape[-1] > 3 :
        img = img[...,:3]
    ori = np.array(img)
    if xrange is not None :
        img = img[:,xrange[0]:xrange[1]]
    if yrange is not None :
        img = img[yrange[0]:yrange[1]]
    mask, ptime =  decode_an_image_array( img, manTraNet, dn )
    ptime = ptime.total_seconds()
    # show results
    pyplot.figure( figsize=(15,5) )
    pyplot.title('Original Image')
    pyplot.subplot(131)
    pyplot.imshow( img )
    pyplot.title('Forged Image (FotoCops)')
    pyplot.subplot(132)
    pyplot.imshow( mask, cmap='gray' )
    pyplot.title('Predicted Mask (FotoCops)')
    pyplot.subplot(133)
    pyplot.imshow( np.round(np.expand_dims(mask,axis=-1) * img[::dn,::dn]).astype('uint8'), cmap='jet' )
    pyplot.title('Highlighted Forged Regions')
    # pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( url, rgb.shape, ptime ) )
    pyplot.show()

def get_image():
    img = Image.open('static/uploads/test.jpg')
    img = np.array(img)
    if img.shape[-1] > 3:
        img = img[..., :3]
    mask, ptime = decode_an_image_array(img, manTraNet, 1)
    pyplot.imshow(mask, cmap='gray')
    pyplot.title('Masked Forged Region')
    pyplot.savefig('static/uploads/saved.jpg')

# for k in range(1) :
#     # get a sample
#     forged_file, original_file = get_a_random_pair()
#     # load the original image just for reference
#     ori = read_rgb_image( original_file )
#     # manipulation detection using ManTraNet
#     rgb, mask, ptime = decode_an_image_file( forged_file, manTraNet )
#     # show results
#     pyplot.figure( figsize=(15,5) )
#     pyplot.subplot(131)
#     pyplot.imshow( ori )
#     pyplot.title('Original Image')
#     pyplot.subplot(132)
#     pyplot.imshow( rgb )
#     pyplot.title('Forged Image (FotoCops)')
#     pyplot.subplot(133)
#     pyplot.imshow( mask, cmap='gray' )
#     pyplot.title('Predicted Mask (FotoCops)')
#     pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( os.path.basename( forged_file ), rgb.shape, ptime ) )
#     pyplot.show()


# get_image_from_url('https://i.imgur.com/2gS6lgL.png')
get_image()
