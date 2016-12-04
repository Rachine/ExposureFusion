# ExposureFusion
Exposure Fusion Technique
This code implements the Exposure Fusion, a Low-Dynamic-Range technique. It blends multi-exposure sequence of photo into a high-quality image, and is guided by measurement as Contrast, Saturation and Well-exposedness.

You can find the Research paper where the algorithm comes from [here](https://github.com/Rachine/ExposureFusion/blob/master/exposure_fusion.pdf) [1]

This code used OpenCV 3 and python 2.7 and the installation is based on this [Tutorial](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/).

Here some multi-exposure sequence of photo used:

<img src="/image_set/jpeg/t_0_1" width="240"> <img src="/image_set/jpeg/t_5_1" width="240"> <img src="/image_set/jpeg/t_9_1" width="240">

Here the first result with the Naive implemtation:

<img src="/result_jpeg_naive.png" width="500">


# References
[1]: Exposure Fusion: A Simple and Practical Alternative to High Dynamic Range Photography
Mertens, T.; Kautz, J.; Van Reeth, F.
