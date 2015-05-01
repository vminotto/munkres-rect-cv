munkres-rect-cv
===============

Munkres assignment method (also know as the Hungarian Algorithm). This implementation handles both square and
rectangular problems (non-square cost matrices). OpenCV is used for manipulating data, and either float or double-typed
costs are allowed. The current implementation was developed and tested using OpenCV 2.49, but lower and
higher versions of the library should compile with no problem. The project needs to be linked to the opencv_coreVER.lib (where VER is the the version suffix).

Examples on how to use the algorithm are found in main.cpp.

This code is based on a Matlab implementation, by Yi Cao, that can be found in the link below.
http://www.mathworks.com/matlabcentral/fileexchange/20652-hungarian-algorithm-for-linear-assignment-problems--v2-3-

As is, the code may still receive some optimizations, which might be necesasary if very large assignment problems are being
solved. For small problems, about 15x15, the runtime is lower than 0.1ms.

At your own risk, feel free to use this code to whatever purposes you wish.
