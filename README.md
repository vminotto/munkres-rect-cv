munkres-rect-cv
===============

Inplementation of the Munkres assignment method (also know as the Hungarian Algorithm). This implementation handles 
rectangular problems (non-square matrix). OpenCV is used for manipulating data, handling either float or double-typed
costs. The current version was developed and tested using OpenCV 2.49, but lower and
higher versions should compile with no problem. The project needs to be linked to the opencv_coreVER.lib (where VER
is the the version suffix).

Examples on how to use the algorithm are found in main.cpp.

This code is based on a Matlab implementation, by Yi Cao, that can be found in the link below.
http://www.mathworks.com/matlabcentral/fileexchange/20652-hungarian-algorithm-for-linear-assignment-problems--v2-3-

As is, the code may still receive some optimizations, which might be necesasary if very large assignment problems are being
solved. For small problems, about 15x15, the runtime is lower than 1ms.
