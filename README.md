# VGGFace-Ear Dataset Annotations

This annotations were made over the existing VGGFace2 dataset (www.robots.ox.ac.uk/~vgg/data/)

Annotations are saved in CSV files, there is one file per set. The train set gathers 600 subject and the test set, 60 subjects.

Annotations are as follows: 

1.  folder 	    &emsp string 	Class name given by the VGGFace dataset
2.  file_img 	  &emsp string 	Image or sample name given by the VGGFace datase
3.  num_object 	int 	  Order number of ear detected in the image using the Mask-RCNN
4.  score 	    float 	Detection score given by the Mask-RCNN
5.  y 		      int 	  Y Upper-left value of the bounding box given by the Mask-RCNN
6.  x 		      int 	  X Upper-left value of the bounding box given by the Mask-RCNN
7.  y2 		      int 	  Y Lower-right value of the bounding box given by the Mask-RCNN
8.  x 		      int 	  X Lower-right value of the bounding box given by the Mask-RCNN
9.  height 	    int 	  Calculated height given bounding box coordinates
10. width 	    int 	  Calculated width given bounding box coordinates
11. ratio 	    float 	Calculated aspect ratio given height and width
12. mask_per 	  float 	Calculated percentage of pixels of the mask over the total number of pixels in the image
13. ear_nonear 	float 	Classification score from the ears and non-ears classifier


To download the annotations, please check the [Release Agreement](https://docs.google.com/document/d/1bkIcBDEIUh2I14i5uck2jiembiUQqRoh/edit?usp=sharing&ouid=105859654314927776258&rtpof=true&sd=true)

