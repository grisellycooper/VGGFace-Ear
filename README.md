# VGGFace-Ear Dataset Annotations

This annotations were made over the existing VGGFace2 dataset (www.robots.ox.ac.uk/~vgg/data/)

Annotations are saved in CSV files, there is one file per set. The train set gathers 600 subject and the test set, 60 subjects.

Annotations are as follows: 

|N°  | Field        | Type    | Description
|----| ------------ | --------| ---------------------------------------------------------------------------------------------
| 1  | folder 	    | string 	| Class name given by the VGGFace dataset
| 2  | file_img 	  | string 	| Image or sample name given by the VGGFace datase
| 3  | num_object 	| int 	  | Order number of ear detected in the image using the Mask-RCNN
| 4  | score 	      | float 	| Detection score given by the Mask-RCNN
| 5  | y 		        | int 	  | Y Upper-left value of the bounding box given by the Mask-RCNN
| 6  | x 		        | int 	  | X Upper-left value of the bounding box given by the Mask-RCNN
| 7  | y2 		      | int 	  | Y Lower-right value of the bounding box given by the Mask-RCNN
| 8  | x 		        | int 	  | X Lower-right value of the bounding box given by the Mask-RCNN
| 9  | height 	    | int 	  | Calculated height given bounding box coordinates
| 10 | width 	      | int 	  | Calculated width given bounding box coordinates
| 11 | ratio 	      | float 	| Calculated aspect ratio given height and width
| 12 | mask_per 	  | float 	| Calculated percentage of pixels of the mask over the total number of pixels in the image
| 13 | ear_nonear 	| float 	| Classification score from the ears and non-ears classifier


To download the annotations, please check the [Release Agreement](https://docs.google.com/document/d/1bkIcBDEIUh2I14i5uck2jiembiUQqRoh/edit?usp=sharing&ouid=105859654314927776258&rtpof=true&sd=true)


## Citation
Please use this bibtex to cite this work:

```
@Article{vggfaceear_dataset_2022,
  AUTHOR = {Ramos-Cooper, Solange and Gomez-Nieto, Erick and Camara-Chavez, Guillermo},
  TITLE = {VGGFace-Ear: An Extended Dataset for Unconstrained Ear Recognition},
  JOURNAL = {Sensors},
  VOLUME = {22},
  YEAR = {2022},
  NUMBER = {5},
  ARTICLE-NUMBER = {1752},
  URL = {https://www.mdpi.com/1424-8220/22/5/1752},
  PubMedID = {35270896},
  ISSN = {1424-8220},
  DOI = {10.3390/s22051752}
}

```
