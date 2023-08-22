# Anaglyph 

Repository under construction! 

Feed the 3D-coordinates of your data manifold to produce videos that pop out of your screen.

## How does it work?

Our eyes usuallly see the same objects with slightly different angles. Anaglyph images use this to produce a virtual image that appears to be closer to us than it really is. For this, two images (usually in cyan and red) that have two different projections are overlapped and the viewer will see the objects popping out of it when wearing this type of glasses:
<img src="https://github.com/reginasar/anaglyph/assets/50836927/5fa9f30d-230f-4388-ab29-2786de2390c0" alt="glasses" width="200"/>

The colours in each side of the glasses will prevent seeing part of the image -the part that the other eye does see-.


<IMG SRC="3Danim.gif">.

## This repository

This repository has two main classes. One is used to define the trajectory of the camera that will record the 3D-video. The other one is used to create the particles object, which has a method to record the video (using as input a trajectory). 

