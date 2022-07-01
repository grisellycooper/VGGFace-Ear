# BECAUSE MEMORY' ISSUES ---> TO CHECK!
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	# Restrict TensorFlow to only allocate 4GB of memory on the first GPU
	try:
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')

		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  	
	except RuntimeError as e:
    	# Memory growth must be set before GPUs have been initialized
		print(e)


import os
from basemodels import VGG
from commons import functions, distances, metrics

ROOT_DIR = os.path.abspath("")

def load_model(model_name, layer):
	basemodels = {
		'VGG16': VGG.loadVGG16, 
		'VGG16Face': VGG.loadVGG16Face,
		'VGG16Ear': VGG.loadVGG16Ear,
		'VGG16Fusion': VGG.loadVGG16Fusion,				
		'VGGFace': VGG.loadVGGFace,	## **
		'OpenFace2': VGG.loadOpenFace2,
		'VGGFaceFace': VGG.loadVGGFace_Face,	
		'VGGFaceEar': VGG.loadVGGFace_Ear,		
		'VGGFaceFusion': VGG.loadVGGFace_Fusion,		
		'VGGFace_EDA1':VGG.loadVGGFace_EDA_first,
		'VGGFace_EDA2':VGG.loadVGGFace_EDA_second,
		'VGGFace_EDA3':VGG.loadVGGFace_EDA_third,
		'VGGEar': VGG.loadVGGEar, 
		'VGGFaceEar_MS': VGG.loadVGGFaceEar_MS,
		'VGG16Ear2': VGG.loadVGG16_Ear,
		'VGGFaceEar2':VGG.loadVGGFace_Ear_2,
		'VGGFaceEar160':VGG.loadVGGFace_Ear_160,
		'VGGFaceEar224_2':VGG.loadVGGFace_Ear_2nd,
		'VGGEar':VGG.loadVGGFace_Ear_3rd, ## VGGEar    ex VGGFaceEar224_3
		'VGGFaceEar224_3_uerc':VGG.loadVGGFace_Ear_3rd_uerc,
		'VGGFaceEar224_3_earvn':VGG.loadVGGFace_Ear_3rd_earvn,
		'VGG16Ear_CI':VGG.loadVGG16_Ear_CustomInput,
		'OpenFace':VGG.loadOpenFace, ## This is DeepFace
		'VGGFaceEar_TwoStream':VGG.loadVGGFaceEar_TwoStream, ## 60c
		'VGGFaceEar_TwoStream_2':VGG.loadVGGFaceEar_TwoStream_2, ## 300c
		'VGGFaceEar_TL':VGG.loadVGGFaceEar_tripleloss,
		'VGG_ucsm':VGG.loadVGGUcsm,
	}
	VGG.loadModel(layer)
	model = basemodels.get(model_name)

	if model:
		model = model()
		return model
	else:
		raise ValueError('**Invalid name: {}'.format(model_name))


def verify(img1_path = '', img2_path = '', model_name = 'VGGFace'):
	
	# get descriptor according to model_name
	descriptor = load_model(model_name=model_name)

	# get image data
	img_data1 = functions.preprocess_image(img1_path)
	img_data2 = functions.preprocess_image(img2_path)
	
	# get feature vectors 
	img1_fv = descriptor.predict(img_data1)[0]
	img2_fv = descriptor.predict(img_data2)[0]

	#print("----> Feature vectors", img1_fv.shape, " - ", img2_fv.shape)
	# get distances
	cosine = distances.getCosineDistance(img1_fv, img2_fv)
	euclidean = distances.getEuclideanDistance(img1_fv, img2_fv)

	print("---> Cosine: {} \n Euclidean: {}".format(cosine, euclidean))

	return 

def verify_batch(files_paths, model_name, layer, distances, load_predictions, load_matrix_distances, save_csvfiles, verbose):

	### ---> check if 'files_path' exists	
	if len(files_paths) == 0:
		raise ValueError('** E: Files path should be given.')

	for filepath in files_paths:
		if os.path.exists(filepath):
			directories = next(os.walk(filepath))[1]
			print('{} directories found.'.format(len(directories)))
		else:
			raise ValueError('** {} was not found!'.format(filepath))		

	### ---> get descriptor according to 'model_name'
	descriptor = load_model(model_name=model_name, layer=layer)
	descriptor.summary()

	### ---> set distances
	#distances = ['chisquared', 'cosine', 'euclidean']
	
	results = metrics.getRankMetric(files_paths = files_paths
		, descriptor = descriptor
		, distances = distances
		, load_predictions = load_predictions
		, load_matrix_distances = load_matrix_distances
		, csvfiles = save_csvfiles
		, pverbose = verbose)


def verify_batch_scorefusion(files_paths, models_name, layer, distances, fusion, annot_csv, load_predictions, save_csvfiles, verbose):

	### ---> check if 'files_path' exists	
	if len(files_paths) == 0:
		raise ValueError('**Files path should be given.')

	for filepath in files_paths:
		if os.path.exists(filepath):
			directories = next(os.walk(filepath))[1]
			print('{} directories found.'.format(len(directories)))
		else:
			raise ValueError('**Files path was not found: {}'.format(filepath))		

	### ---> get descriptor according to 'model_name'
	descriptors = []
	for model in models_name:
		descriptor = load_model(model_name=model, layer=layer)
		descriptor.summary()
		descriptors.append(descriptor)

	### ---> set distances	
	#distances = ['cosine']

	results = metrics.getRankMetric_ScoreFusion(files_paths = files_paths
		, descriptors = descriptors
		, distances = distances
		, fusion = fusion
		, annot_csv = annot_csv
		, load_predictions = load_predictions
		, csvfiles = save_csvfiles
		, pverbose = verbose)


def main():
	#img1_path = os.path.join(ROOT_DIR, "tmp", "01_01.jpg")
	#img2_path = os.path.join(ROOT_DIR, "tmp", "01_02.jpg")
    
	#print(img1_path, img2_path)
	#verify(img1_path=img1_path, img2_path=img2_path, model_name='VGGFace')

    #G:/MCS/Labs/Datasets/EarDomainAdap_VGGFace/test_EDA/VGGFace_25class_ear
    #G:/MCS/Labs/Datasets/EarDomainAdap_VGGFace/test_EDA/UERC_180Class_testset_preprocessed
	
    #G:/MCS/Labs/Datasets/EarDomainAdap_VGGFace/train_EDA/WPUT
    #G:/MCS/Labs/Datasets/EarDomainAdap_VGGFace/train_EDA/WPUT_75x75
    #G:/MCS/Labs/Datasets/EarDomainAdap_VGGFace/train_EDA/VGGFaceEar_train

    #G:/MCS/Labs/Datasets/UERC/uerctrain

    ##---> MultiBiometric Dataset
    #G:/MCS/Labs/Datasets/VGGFaceEar/MS_25_class_test/face_full
    #G:/MCS/Labs/Datasets/VGGFaceEar/MS_25_class_test/ear_full

    #G:/MCS/Labs/Datasets/AWE_MS_35class/AWE_ear
    #G:/MCS/Labs/Datasets/AWE_MS_35class/AWE_face


    ### CLEI Experiments

    # VGGFaceEar 
    # G:/MCS/Labs/Datasets/VGGFace/VGGFace_new_test_set_25_preprocessed
    # G:/MCS/Labs/Datasets/VGGFace/VGGFace_new_train_set_50-350/val_test

	#paths = ["G:/MCS/Labs/Datasets/VGGFaceEar/MS_25_class_test/face_full/", "G:/MCS/Labs/Datasets/VGGFaceEar/MS_25_class_test/ear_full/"]
	

	#"G:/MCS/Labs/Datasets/VGGFaceEar/101class_test/ear"

	### AMI
	# G:/MCS/Labs/Datasets/AMI/ami_preprocessed
	# G:/MCS/Labs/Datasets/AMI/ami_preprocessed_CI

	# P:/MCS/AllDatasets/WPUT_not_repeated_preprocessed_250x250
	
	# G:/MCS/Labs/Datasets/UERC/uerctrain_preprocessed
	# G:/MCS/Labs/Datasets/UERC/UERC_180Class_testset_preprocessed
	
	# G:/MCS/Labs/Datasets/VGGFace/VGGFace_new_train_set_50-350/val_test
	# G:/MCS/Labs/Datasets/VGGFace/Test_150x10/VGGFace_new_test_1_preprocessed
	# G:/MCS/Labs/Datasets/VGGFace/VGGFace_new_test_set_25

	#### Multistream Test 1
	# P:/MCS/Datasets/VGGFaceEar/101class_test/ear
	# P:/MCS/Datasets/VGGFaceEar/101class_test/face
	

	## VGGFace-Ear test set 36 classes x 20 samples 
	# G:/MCS/Labs/Datasets/VGGFace2/test_36c_preprocessed
	# G:/MCS/Labs/Datasets/VGGFace2/test_36c_square


	### UERC test set
	# G:/MCS/Labs/Datasets/UERC/UERC_180Class_testset_preprocessed
	# G:/MCS/Labs/Datasets/UERC/UERC_180Class_testset_preprocessedCI

	## UERC val set
	# G:/MCS/Labs/Datasets/UERC/EDA_uerctrain_preprocessed/val

	### EarVN1.0 test set
	# G:/MCS/Labs/Datasets/EarVN10/test_58c_preprocessed
	# G:/MCS/Labs/Datasets/EarVN10/test_58c_preprocessed_CI

	### EarVN1.0 val set
	# G:/MCS/Labs/Datasets/EarVN10/train_106c_preprocessed/val

	### VGGFace-Ear Test Set  ## Sensors
	# G:/MCS/Labs/Datasets/VGGFace2/sensors_test/test_60c_150/preprocessed
	# G:/MCS/Labs/Datasets/VGGFace2/sensors_test/test_60c_150/preprocessed_CI

	### VGGFace-Ear Test Set 2 ## New Sensor	
	# M:/MyEarDataset/VGGFace_Detected_2/2_test_60_options/2_preprocessed	
	# M:/MyEarDataset/VGGFace_Detected_2/2_test_60_options/2_preprocessed_CI


	### VGGFace-Ear Val Set
	# G:/MCS/Labs/Datasets/VGGFace2/sensors_test/train_600c/val
	# G:/MCS/Labs/Datasets/VGGFace2/sensors_test/train600c_3_cis/val
	
	
	### UCSM Dataset
	# M:/UCSM_dataset/output_detect_filtered_preprocessed
	# M:/UCSM_dataset/output_detect_filtered_preprocessed_CI

	# G:/MCS/Labs/Datasets/UCSM/Test/1/test
	# G:/MCS/Labs/Datasets/UCSM/Test/2/test
	# G:/MCS/Labs/Datasets/UCSM/Test/3/test


	## VGGFace-Ear MB
	## M:/VGGFace-EarMB/face_selected/test2 ## Face
	## M:/VGGFace-EarMB/ear_selected/test2  ## Ear


	
	# VGGFaceEar super short 15c
	## M:/VGGFace-EarMB/ear_selected_short/train_supershort/test

	# Supershorts
	## M:/VGGFace-EarMB/face_supershort ## Preprocessed
	## M:/VGGFace-EarMB/face_supershort/normal
	
	## UERC MB 
	## G:/MCS/Labs/Datasets/Multibiometrics/UERC/Test/selected_test/face/test
	## G:/MCS/Labs/Datasets/Multibiometrics/UERC/Test/selected_test/ear/test

	#paths = ["M:/VGGFace-EarMB/face_selected/test2", "M:/VGGFace-EarMB/ear_selected/test2"]
	#paths = ["M:/VGGFace-EarMB/ear_selected/test2"]

	#paths = ["G:/MCS/Labs/Datasets/Multibiometrics/UERC/Test/selected_test/face/test", "G:/MCS/Labs/Datasets/Multibiometrics/UERC/Test/selected_test/ear/test"]
	
	#paths = ["G:/MCS/Labs/Datasets/Multibiometrics/UERC/Test/selected_test/ear/test"]

	'''verify_batch(files_paths = paths  ## Data must be preprocessed
								, model_name = 'VGGFaceEar_TwoStream_2'
								, layer = 2 	## 2-> 2622 last fc layer// 4-> 4096_1 // 6-> 4096_2 
								, distances = ['cosine'] 	## ['chisquared', 'cosine', 'euclidean']
								, load_predictions = None
								#, load_predictions = "G:/MCS/Labs/Test/VGGFaceEar/vggfaceear/csvfiles/fusion_unimodal/face/1_uerc_test_2622_predictions.csv"
								#, load_matrix_distances = "G:/MCS/Labs/Test/VGGFaceEar/vggfaceear/csvfiles/fusion_unimodal/ear/1_cosine_distances_matrix.csv"
								, load_matrix_distances = None
								, save_csvfiles = True
								, verbose = False)'''
							
	##---> Score-level Fusion

	paths = ["M:/VGGFace-EarMB/face_selected/test2", "M:/VGGFace-EarMB/ear_selected/test2"]
							
	annot_mb_path = "M:/VGGFace-EarMB/annot_files/vggface-ear_mb_annotations_test_8999.csv"

	predictions_paths = ["G:/MCS/Labs/Test/VGGFaceEar/vggfaceear/csvfiles/fusion_unimodal/vggfaceearmb/face/3_2622_predictions.csv","G:/MCS/Labs/Test/VGGFaceEar/vggfaceear/csvfiles/fusion_unimodal/vggfaceearmb/ear/2_2622_predictions.csv"]

	verify_batch_scorefusion(files_paths = paths
						, models_name = ['VGGFace', 'VGGEar']
						, layer = 2 	## 2-> 2622 last fc layer// 4-> 4096_1 // 6-> 4096_2
						, distances = ['cosine']  ## ['chisquared', 'cosine', 'euclidean']
						, fusion = 'wsum' 	## 'avg', 'max', 'wsum'
						, annot_csv = annot_mb_path
						, load_predictions = predictions_paths ## 1:face 2:ear
						#, load_predictions = None ## 1:face 2:ear
						, save_csvfiles = True
						, verbose = False)
											

	return

if __name__ == "__main__":
    main()