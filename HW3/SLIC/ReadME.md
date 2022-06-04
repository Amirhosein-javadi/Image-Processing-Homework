# **SLIC**

This algorithm generates superpixels by clustering pixels based on their color similarity and proximity in the image plane.  

This is done in the five-dimensional **labxy** space, where **lab** is the pixel color vector in CIELAB color space and **xy** is the pixel position. 

We need to normalize the spatial distances in order to use the Euclidean distance in this 5D space because the maximum possible distance between two colors in the CIELAB space is limited whereas the spatial distance in the xy plane depends on the image size.

𝑑<sub>𝑙𝑎𝑏</sub> = (𝐿<sub>𝑘</sub> − 𝐿<sub>𝑛</sub>)<sup>2</sup> + (𝑎<sub>𝑘</sub> − 𝑎<sub>𝑛</sub>)<sup>2</sup> + (𝑏<sub>𝑘</sub> − 𝑏<sub>𝑛</sub>)<sup>2</sup>

𝑑<sub>𝑥𝑦</sub> = (𝑥<sub>𝑘</sub> − 𝑥<sub>𝑛</sub>)<sup>2</sup> + (𝑦<sub>𝑘</sub> − 𝑦<sub>𝑛</sub>)<sup>2</sup>

d = 𝑑<sub>𝑥𝑦</sub> + c 𝑑<sub>𝑙𝑎𝑏</sub> 

Input Image:

<img src="slic.jpg" width="400" height="300">

K = 64:

<img src="Result1.jpg" width="400" height="300">


K = 256:

<img src="Result2.jpg" width="400" height="300">


K = 1024:

<img src="Result3.jpg" width="400" height="300">


K = 2048:

<img src="Result4.jpg" width="400" height="300">
