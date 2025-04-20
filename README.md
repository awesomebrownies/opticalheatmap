# Edge and Motion Detection with OpenCV
An image analyzation system that uses multi-threading to compute in real-time.

![34683b32b10494012aa97415a60dc631245d3972_download_filename_a49470921c142dd7ab15_Screenshot_from_2025-04-18_21-52-19](https://github.com/user-attachments/assets/68da2dbd-41f4-46aa-a624-aac83a3c510f)

## Featuring three layers of data
### Edge Detection
Gaussian blur noise reduction, and canny edge detection with dynamic threshold based on average intensity.
### Background Removal
Using MOG2, the background is removed and therefore lights up new objects in the frame of reference. This algorithm updates over time to adapt to light conditions or changes in scenery.
### Image subtraction
Finds differences between multiple frames of the live feed, and colors from **yellow** to **red**, depending on intensity of the change. The "positive" motion is masked onto the background removed image.

[![Edge and Motion Detection Youtube Video](https://img.youtube.com/vi/9QLFio-HLRw/maxresdefault.jpg)](https://www.youtube.com/watch?v=9QLFio-HLRw)
