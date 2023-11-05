# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:09:57 2018

@author: Debora Gil, Guillermo Torres

Quality Measures of an automatic segmentation computed from
a mask of the object (ground truth) 
Two types of measures are implemented:
    1. Volumetric (dice, voe, relvoldiff) compute differences and 
    similarities between the two volumes. They are similar to precision and
    recall.
    2. Distance-base (AvDist, MxDist) compare volume surfaces 
    in terms of distance between segmentation and ground truth.
    Average distances, AvDist, is equivalent to Euclidean distance between
    volumes, while Maximum distance, MxDist, is the infinite norm and detects
    puntual deviations between surfaces

References: 
    1. T. Heimann et al, Comparison and Evaluation of Methods for
Liver Segmentation From CT Datasets, IEEE Trans Med Imag, 28(8),2009
"""
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as bwdist

def DICE(Seg,GT):
    """
    Computes dice index between segmenation Seg and 
    ground truth mask GT:
        dice=2 ||Seg \intersect GT||/||Seg \union GT||
    for || . || indicating the volume
     
    INPUT:
    1> Seg: Binary ndarray of segmentation
    2> GT:  Binary ndarray of true object 
    """
    dice=np.sum(Seg[np.nonzero(GT)])*2.0 / (np.sum(Seg) + np.sum(GT))
    return dice

def VOE(Seg,GT):
    """
    Computes volume overlap error (voe) between segmenation Seg and 
    ground truth mask GT:
        voe=1-2 ||Seg \intersect GT||/||Seg \union GT||
    for || . || indicating the volume
    
    INPUT:
        1> Seg: Binary ndarray of segmentation
        2> GT:  Binary ndarray of true object 
    """
    voe=(1-2*np.sum(Seg*GT)) / (np.sum(Seg) + np.sum(GT))
    return voe

def RelVolDiff(Seg,GT):
    """
    Computes relative volume difference between segmenation Seg and 
    ground truth mask GT:
        RelVolDiff= ||Seg - GT||/||Seg||
    for || . || indicating the volume
    
    INPUT:
        1> Seg: Binary ndarray of segmentation
        2> GT:  Binary ndarray of true object 
    """
    RelVolDiff=(np.sum(Seg)-np.sum(GT))/np.sum(Seg)
    return RelVolDiff

def DistScores(Seg,GT):
    
    # Distances to Segmented Volume
    DistSegInt=bwdist(Seg)
    DistSegExt=bwdist(1-Seg)
    DistSeg=np.maximum(DistSegInt,DistSegExt)
    # Distances to GT Volume
    DistGTInt=bwdist(GT)
    DistGTExt=bwdist(1-GT)
    DistGT=np.maximum(DistGTInt,DistGTExt)
    
    # Boundary points
    # OBS: This way is more accurate than using 
    # pyhton function to compute isosurface:
    # vertices, _,_,_ = 
    # measure.marching_cubes_lewiner(Seg, level=0.9)
    # index=vertices.astype(int)
    # i=index[0];j=index[1];k=index[2]
    # idxs=np.ravel_multi_index((i,j,k),Seg.shape)
     
    BorderSeg=((DistSegInt<1)+(DistSegInt>1))==0;
    BorderGT=((DistGTInt<1)+(DistGTInt>1))==0;
    
    DistAll= np.concatenate((DistSeg[BorderGT],DistGT[BorderSeg]),axis=0)
    
    DistAvg=np.mean(DistAll)
    DistMx=np.max(DistAll)
    
    return DistAvg,DistMx