# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:59:33 2023

@author: tandeitnik

Description: As explained in the user guide of the DLP4500 .45 WXGA DMD,
because of the diamond disposition of its mirrors , images displayed on it suffer
from conversion and scaling artifacts. More precisely, the final imaged is streched
by a factor of 2 along the horizontal direction. Squares becomes rectangles, cicles
becomes ellipses and so forth. Therefore, the user should prepare the image with
this in mind, as squeezing it by a factor of 2 along the horizontal direction.

This scripts emulates how the image looks like when displayed on the surface of
the DMD. Given an image, it calculates the diamond grid of the DMD, converts the
image to black and white, and for each pixel of the image fills the corresponding
mirror on the diamond grid.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image 

def makeGrid(row,col,side = 1):
    """
    Parameters
    ----------
    row : int
        Number of rows of the grid.
    col : int
        Number of columns of the grid.
    side : float, optional
        Size of the principal axis of the diamond. The default is 1.

    Returns
    -------
    grid : list of arrays
        List for which each elenent contains the points of each row of the grid. 
        Each row is coded as an array with two rows, where the first (second) row 
        stores the x (y) coordinate of the points.

    """
    grid = []
    
    points = np.zeros([2,4*col+1])
    points[0,:] = np.concatenate((np.linspace(0,side*col,2*col+1), np.flip(np.linspace(0,side*col-side/2,2*col))))
    points[1,1:2*col+1:2] = side/2
    points[1,2*col+1::2] = -side/2
    
    grid.append(points)
    
    for i in range(1,row):
        
        pointsTemp = np.copy(points)
        pointsTemp[0,:] -= side/2*(i%2)
        pointsTemp[1,:] -= side*0.5*i
        grid.append(pointsTemp)
        
    return grid

def printGrid(grid, linewidth = 1, color = 'black'):
    """
    Parameters
    ----------
    grid : list of arrays
        Grid given by the makeGrid() function.
    linewidth : float, optional
        Line width of the plot. The default is 1.
    color : string, optional
        Color of lines. The default is 'black'.

    Returns
    -------
    Plots the grid and returns nothing.
    """
    
    for i in range(len(grid)):
        
        plt.plot(grid[i][0,:],grid[i][1,:], color = color, linewidth = linewidth)


def drawPixel(grid,i,j,col):
    """
    Parameters
    ----------
    grid : list of arrays
        Grid given by the makeGrid() function.
    i : int
        Row index of the pixel to be painted black.
    j : int
        Column index of the pixel to be painted black.
    col : int
        Number of columns of the picture.

    Returns
    -------
    Prints the assigned pixel, i.e., colors the diamond correspondig to the assigned
    pixel on the grid and returns nothing.

    """
    
    plt.fill_between(grid[i][0,2*j:2*j+2], grid[i][1,2*j:2*j+2], np.flip(grid[i][1,4*col-2*j-1:4*col-2*j+1]), color='black',
    				alpha=1)
    plt.fill_between(grid[i][0,2*j+1:2*j+3], grid[i][1,2*j+1:2*j+3], np.flip(grid[i][1,4*col-2*j-2:4*col-2*j]), color='black',
    				alpha=1)
    
def drawDmdImage(fileName, saveName = 'dmdImage.png',dpi = 300, linewidth = 0):
    """
    Parameters
    ----------
    fileName : string
        Name of the file to be converted into a DMD image. Should contain the 
        file path if the file is not in the active folder.
    saveName : string
        Name of the file to be saved. The default is 'dmdImage.png'.
    dpi : int, optional
        Dots per inch of the final image. The default is 300.
    linewidth : int, optional
        Line width of the plot. If the image is big, the recommended is to leave
        it at 0. The default is 0.

    Returns
    -------
    Saves the DMD image and returns nothing.
    """
    image = Image.open(fileName) # open colour image
    image = image.convert('1') # convert image to black and white
    image = np.array(image)
    row,col = np.shape(image)
    grid = makeGrid(row,col) #make the diamond grid
    plt.ioff() #disable plot
    
    for i in tqdm(range(row)):
        
        printGrid(grid, linewidth)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.gray()
        
        for j in range(col):
            
            if image[i,j] == 0:
                
                drawPixel(grid,i,j,col)
        
        plt.savefig(saveName, bbox_inches='tight',dpi=dpi)
        imTemp = Image.open(saveName)
        imTemp = imTemp.convert('1') # convert image to black and white
        
        if i == 0:
            dmdImage = np.array(imTemp)
        else:
            dmdImage = dmdImage&np.array(imTemp)

        plt.close()
    
    im = Image.fromarray(dmdImage)
    im.save(saveName)
    plt.ion() #re-enable plotting
