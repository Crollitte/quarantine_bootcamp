import matplotlib.image as mpimg
import sys

from tensorflow.python.keras.metrics import Metric


import solaris as sol
from solaris.data import data_dir
import os
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union
import matplotlib as plt
import json

import gdal
import osr
import numpy as np

path_to_spacenet_utils = '/mnt/Data/Rasters/spacenet_utilities/python/'
sys.path.extend([path_to_spacenet_utils])

from spaceNetUtilities import geoTools as gT


##############################################################################    

def geojson_to_pixel_arr2(raster_file, geojson_file, pixel_ints=True,
                       verbose=False):
    '''
    Tranform geojson file into array of points in pixel (and latlon) coords
    pixel_ints = 1 sets pixel coords as integers
    '''
    
    # load geojson file
    with open(geojson_file) as f:
        geojson_data = json.load(f)

    # load raster file and get geo transforms
    src_raster = gdal.Open(raster_file)
    targetsr = osr.SpatialReference()
    targetsr.ImportFromWkt(src_raster.GetProjectionRef())
        
    geom_transform = src_raster.GetGeoTransform()
    
    # get latlon coords
    latlons = []
    types = []
    for feature in geojson_data['features']:
        coords_tmp = feature['geometry']['coordinates'][0]
        type_tmp = feature['geometry']['type']
        if verbose: 
            print("features:", feature.keys())
            print ("geometry:features:", feature['geometry'].keys())

            #print ("feature['geometry']['coordinates'][0]", z
        latlons.append(coords_tmp)
        types.append(type_tmp)
        #print feature['geometry']['type']
    
    # convert latlons to pixel coords
    pixel_coords = []
    latlon_coords = []
    for i, (poly_type, poly0) in enumerate(zip(types, latlons)):
        
        if poly_type.upper() == 'MULTIPOLYGON':
            #print ("oops, multipolygon"
            for poly in poly0:
                poly=np.array(poly)
                if verbose:
                    print ("poly.shape:", poly.shape)
                    
                # account for nested arrays
                if len(poly.shape) == 3 and poly.shape[0] == 1:
                    poly = poly[0]
                    
                poly_list_pix = []
                poly_list_latlon = []
                if verbose: 
                    print ("poly", poly)
                for coord in poly:
                    if verbose: 
                        print ("coord:", coord)
                    lon, lat, z = coord 
                    px, py = gT.latlon2pixel(lat, lon, input_raster=src_raster,
                                         targetsr=targetsr, 
                                         geom_transform=geom_transform)
                    poly_list_pix.append([px, py])
                    if verbose:
                        print ("px, py", px, py)
                    poly_list_latlon.append([lat, lon])
                
                if pixel_ints:
                    ptmp = np.rint(poly_list_pix).astype(int)
                else:
                    ptmp = poly_list_pix
                pixel_coords.append(ptmp)
                latlon_coords.append(poly_list_latlon)            

        elif poly_type.upper() == 'POLYGON':
            poly=np.array(poly0)
            if verbose:
                print ("poly.shape:", poly.shape)
                
            # account for nested arrays
            if len(poly.shape) == 3 and poly.shape[0] == 1:
                poly = poly[0]
                
            poly_list_pix = []
            poly_list_latlon = []
            if verbose: 
                print ("poly", poly)
            for coord in poly:
                if verbose: 
                    print ("coord:", coord)
                lon, lat, z = coord 
                px, py = gT.latlon2pixel(lat, lon, input_raster=src_raster,
                                     targetsr=targetsr, 
                                     geom_transform=geom_transform)
                poly_list_pix.append([px, py])
                if verbose:
                    print ("px, py", px, py)
                poly_list_latlon.append([lat, lon])
            
            if pixel_ints:
                ptmp = np.rint(poly_list_pix).astype(int)
            else:
                ptmp = poly_list_pix
            pixel_coords.append(ptmp)
            latlon_coords.append(poly_list_latlon)
            
        else:
            print ("Unknown shape type in coords_arr_from_geojson()")
            return
            
    return pixel_coords, latlon_coords


def prepare_data():
  

    data=list() 
    pixelsData=list() 
    maskTest=range(1388,6940)
    maskTrain=range(0,1387)

    ######### mano, isso ta demorando infinitos, whyyyyy??
    # cara, acho q deve ter alguma func do python pra baixar a pasta inteira de uma vez, não é possivel
    # se pa o pandas faz isso.. imagino que provavelmente ele vai ser compatível com o solaris

    for i in range(1,6940):
        print('Writing Raster Number: '+str(i))
        path=str('/mnt/Data/Rasters/processedBuildingLabels/3band/3band_AOI_1_RIO_img'+str(i)+'.tif')
        tempData=mpimg.imread(path)
        data.append(tempData)
        #imgplot = plt.imshow(data[i])
        path2=str('/mnt/Data/Rasters/processedBuildingLabels/vectordata/geojson/geojson/Geo_AOI_1_RIO_img'+str(i)+'.geojson')
        #TempPixelsData=geojson_to_pixel_arr2(path,path2)
        #pixelsData.append(TempPixelsData)

    ### Labels

    y_data_footprint=[]
    for i in range(1,6940):
      print('Writing vector Number: '+str(i))
      path='/mnt/Data/Rasters/processedBuildingLabels/3band/'
      file1='3band_AOI_1_RIO_img'+str(i)+'.tif'

      path2=str('/mnt/Data/Rasters/processedBuildingLabels/vectordata/geojson/geojson/')


      file2 = 'Geo_AOI_1_RIO_img'+str(i)+'.geojson'
    
      fp_mask = sol.vector.mask.footprint_mask(df=os.path.join(path2, file2),
                                          reference_im=os.path.join(path,file1))
      print(fp_mask.shape)
        
      if (np.sum(fp_mask)==0.0):
            fp_mask=np.zeros((438,406), dtype=float)
      y_data_footprint.append(fp_mask)




    y_data_road=[]
    for i in range(1,6940):
      print('Writing vector Number: '+str(i))
      path='/mnt/Data/Rasters/processedBuildingLabels/3band/'
      file1='3band_AOI_1_RIO_img'+str(i)+'.tif'

      path2=str('/mnt/Data/Rasters/processedBuildingLabels/vectordata/geojson/geojson/')


      file2 = 'Geo_AOI_1_RIO_img'+str(i)+'.geojson'

      fp_mask = sol.vector.mask.footprint_mask(df=os.path.join(path2, file2),
                                          reference_im=os.path.join(path,file1))
      print(fp_mask.shape)
        
      if (np.sum(fp_mask)==0.0):
            fp_mask=np.zeros((438,406), dtype=float)
            
      y_data_road.append(fp_mask)


    y_data_instances=[]

    for i in range(1,6940):
      print('Writing vector Number: '+str(i))
      path='/mnt/Data/Rasters/processedBuildingLabels/3band/'
      file1='3band_AOI_1_RIO_img'+str(i)+'.tif'

      path2=str('/mnt/Data/Rasters/processedBuildingLabels/vectordata/geojson/geojson/')


      file2 = 'Geo_AOI_1_RIO_img'+str(i)+'.geojson'
    
      fp_mask = sol.vector.mask.footprint_mask(df=os.path.join(path2, file2),
                                          reference_im=os.path.join(path,file1))
      print(fp_mask.shape)


      if (np.sum(fp_mask)==0.0):
            fp_mask=np.zeros((438,406), dtype=float)

      print(fp_mask.shape)
      y_data_instances.append(fp_mask)
     # plot_truth_coords(data[i],pixelsData[i])


    train_inputs = data[4000:6000]
    train_labels = y_data_footprint[4000:6000]
    test_inputs = data[3500:3800]
    test_labels = y_data_footprint[3500:3800]

    return train_inputs, train_labels, test_inputs, test_labels