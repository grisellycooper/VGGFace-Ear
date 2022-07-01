import os
import csv
import time
import numpy as np
import pandas as pd

from commons import functions
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, auc

ROOT_DIR = os.path.abspath("")

def getPredictions(files_paths, descriptor, load_predictions = None):

	tic = time.time()
	print("----> Getting predictions...")
	
	### ---> Get pre saved predictions	
	if load_predictions != None:
		features_data = pd.read_csv(load_predictions)

		if len(features_data.index) > 0:
			print(f"** I: Presaved predictions were loaded: {len(features_data.index)} rows")
			return features_data
		else:
			raise ValueError("** Couldn't load pre saved predictions!")

	### ---> Get descriptor's input size
	target_size = (descriptor.layers[0].input_shape[0][1], descriptor.layers[0].input_shape[0][2])
	print(f"** target_size: {target_size}")

	### ---> Read files in folders
	#TEST_DIR = files_paths[0]
	TEST_DIR = files_paths
	features_list = []
	dirnames = next(os.walk(TEST_DIR))[1]
	dirnames = sorted(dirnames)
	for classname in tqdm(dirnames):
		FILE_DIR = os.path.join(TEST_DIR, classname)		
		files = next(os.walk(FILE_DIR))[2]
		files = sorted(files)
		for filename in files:
	            ### ---> Check file's format
	            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
	            	print("---> Not image format file: ", filename)
	            	continue
	            
	            ### ----> Save information of class and filename
	            image_class = np.array([str(classname)])
	            image_filename = np.array([str(filename)])

	            ### ---> Preprocess image
	            img_data = functions.preprocess_image(os.path.join(FILE_DIR, filename), target_size)

	            ### ---> Predict ans save	            
	            image_vector = np.array(descriptor.predict(img_data)) 

	            features_list.append(np.concatenate((image_filename, image_class, np.squeeze(image_vector)), axis=0))  
	            
	columns = []
	columns.append('filename')	
	columns.append('class')
	for i in range(1, len(np.squeeze(image_vector))+1):
		columns.append("v"+ str(i))

	features_data = pd.DataFrame(features_list, columns = columns)

	### ---> Save CSV file
	if save_csvfiles:
		csvFilesFolder = os.path.join(ROOT_DIR, "csvfiles")
		if not os.path.exists(csvFilesFolder):
			os.mkdir(csvFilesFolder)

		### ---> Build CSV file		
		csvfilename = os.path.join(csvFilesFolder, "predictions.csv")
		features_data.to_csv(csvfilename, encoding='utf-8', index=False)

	### ---> Print time taken
	toc = time.time()	
	print("---> getPredictions() run in: {}".format(time.strftime("%H:%M:%S", time.gmtime(toc-tic))))

	return features_data

def getPredictionsMS(files_paths, descriptor):
	
	first_set_dirs = next(os.walk(files_paths[0]))[1]	
	second_set_dirs = next(os.walk(files_paths[1]))[1]	

	first_set_dirs = sorted(first_set_dirs)
	second_set_dirs = sorted(first_set_dirs)

	if first_set_dirs != second_set_dirs:		
		raise ValueError('** Paths do not contain same number of classes.')		

	tic = time.time()
	print("----> Getting predictions MS...")
	### ---> Get descriptor's input size
	target_size_first = descriptor.inputs[0].shape[1:-1]
	target_size_second = descriptor.inputs[1].shape[1:-1]

	### ---> Read files in folders
	features_list = []
	dirnames = next(os.walk(files_paths[0]))[1]
	dirnames = sorted(dirnames)
	for classname in tqdm(dirnames):
		
		### ---> Files for the first Stream
		FILE_DIR_FIRST = os.path.join(files_paths[0], classname)		
		files_first = next(os.walk(FILE_DIR_FIRST))[2]
		files_first = sorted(files_first)

		### ---> Files for the second Stream
		FILE_DIR_SECOND = os.path.join(files_paths[1], classname)		
		files_second = next(os.walk(FILE_DIR_SECOND))[2]
		files_second = sorted(files_second)

		if (len(files_first) != len(files_second)):
			raise ValueError('** Directories of {} class do not contain the same number of images!')

		for idx, filename in enumerate(files_first):
	            ### ---> Check file's format
	            if not files_first[idx].lower().endswith(('.png', '.jpg', '.jpeg')):
	            	print("---> Not image format file: ", files_first[idx])
	            	continue
	            
	            if not files_second[idx].lower().endswith(('.png', '.jpg', '.jpeg')):
	            	print("---> Not image format file: ", files_second[idx])
	            	continue
	            
	            ### ----> Save information of class and filename
	            image_class = np.array([str(classname)])
	            image_filename = np.array([str(filename)])

	            ### ---> Preprocess image
	            img_data1 = functions.preprocess_image(os.path.join(FILE_DIR_FIRST, files_first[idx]), target_size_first)
	            img_data2 = functions.preprocess_image(os.path.join(FILE_DIR_SECOND, files_second[idx]), target_size_second)

	            ### ---> Predict ans save	            
	            image_vector = np.array(descriptor.predict([img_data1, img_data2]))            	                                              
	            features_list.append(np.concatenate((image_filename, image_class, image_vector[0]), axis=0))  
	
	columns = []
	columns.append('filename')
	columns.append('class')
	for i in range(1, len(image_vector[0])+1):
		columns.append("v"+ str(i))

	features_data = pd.DataFrame(features_list, columns = columns)

	### ---> Save CSV file
	if save_csvfiles:
		csvFilesFolder = os.path.join(ROOT_DIR, "csvfiles")
		if not os.path.exists(csvFilesFolder):
			os.mkdir(csvFilesFolder)

		### ---> Build CSV file		
		csvfilename = os.path.join(csvFilesFolder, "multistream_predictions.csv")
		features_data.to_csv(csvfilename, encoding='utf-8', index=False)

	### ---> Print time taken
	toc = time.time()	
	print("---> getPredictionsMS() run in: {}".format(time.strftime("%H:%M:%S", time.gmtime(toc-tic))))

	return features_data

def getMatchingMatrix(data, distances_, load_matrix_distances = None):

	tic = time.time()
	print("----> Calculating distances...")
	
	dataFrames = []

	### ---> Get pre saved distances
	## ToDo: This only works for one dsitance_csv file, but we can send many distances
	## ToDo: We can not read the file already saved, because duplicate values are not allowed as column names, SOLUTION: Manage better cols and indexes names in dataframes
	if load_matrix_distances != None:
		cols_data = pd.read_csv(load_matrix_distances, index_col = 0, nrows=0)
		new_cols = [x.split(".")[0] for x in cols_data]  ## avoid 0001 0001.1 0001.2 0001.3
		
		distances_data = pd.read_csv(load_matrix_distances, index_col = 0, names=new_cols, header=None)

		if len(distances_data.index) > 0:
			print(f"** I: Presaved distance matrix was loaded: {len(distances_data.index)} rows")
			dataFrames.append(distances_data)
			return dataFrames
		else:
			raise ValueError("** Couldn't load pre saved distance matrix!")

	### ---> Calculate distances 
	for distance in distances_:
		#Xdf = data.drop('class', axis=1)
		Xdf = data.drop(['class','filename'], axis=1)
		ydf = data['class']
		ydf2 = data['filename']
		X = Xdf.values.astype(float)
		y = ydf.values
		y2 = ydf2.values
		s = (len(X), len(X))
		y_pred = np.zeros(s)
		print('rows: {} -  columns: {}'.format(len(X), len(X[0])))
		
		### --> Get distances matrix in y_pred
		print(f"... using {distance} distance...")
		for i in tqdm(range(len(X))):
			for j in range (i, len(X)):
				dis = functions.getDistance(distance, X[i], X[j])
				y_pred[i][j] = dis
				y_pred[j][i] = dis
		
		### ---> Normalize distances  
		### --> Values are already normalized when getting the cosine distance (between 0-2)
		## Min-Max Normalization, just in case we use a different distances than cosine
		y_pred = y_pred - np.min(y_pred) / np.max(y_pred) - np.min(y_pred) 

		### ---> Build DataFrame
		#-> column names saves class labels, while indexes save sample names, nontheles column and row refers to the same sample
		df = pd.DataFrame(y_pred, columns=y, index=y2)

		### ---> Save CSV file distances
		if save_csvfiles:
			csvFilesFolder = os.path.join(ROOT_DIR, 'csvfiles')
			if not os.path.exists(csvFilesFolder):
				os.mkdir(csvFilesFolder)
			
			### ---> Build CSV file
			distancesfilename = os.path.join(csvFilesFolder, distance + '_distances_matrix.csv')
			df.to_csv(distancesfilename, encoding='utf-8', index=True)
		
		### ---> Add to DataFrames list
		dataFrames.append(df)

	### ---> Print time taken
	toc = time.time()
	print('----> getMatchingMatrix() run in: {}'.format(time.strftime("%H:%M:%S", time.gmtime(toc-tic))))	

	return dataFrames

def foo_ws_score(a):
	if a== 1: b = 0.65 ## trust most on face
	if a== 2: b = 0.5 ## trust on both
	if a== 3: b = 0.35 ## trust on ear
	return b

def getScoreFusion(df_distances_face, df_distances_ear, distances, fusion, annot_csv):
	
	tic = time.time()
	print("----> Getting score-level fusion... ", fusion)
	dataFrames = []
	
	for i in range(len(distances)):
		Xdf1 = df_distances_face[i].values 
		Xdf2 = df_distances_ear[i].values  
		y1 = np.array((df_distances_face[i].columns), dtype='str')
		y2 = np.array((df_distances_ear[i].columns), dtype='str')
		y3 = np.array((df_distances_face[i].index), dtype='str')

		comparison = y1 == y2
		if not comparison.all():
			print("W: Input Data does not have the same labels!")
			return

		### --> Working with similarity instead of distance
		#Xdf1 = 1 - (Xdf1/np.max(Xdf1))
		#Xdf2 = 1 - (Xdf2/np.max(Xdf2))
		if fusion == "max":
			# Max Rule Fusion
			print("** Using MAX score fusion")
			Xconcat = np.concatenate([np.expand_dims(Xdf1, axis = 0),np.expand_dims(Xdf2, axis = 0)])			
			max_indexes = Xconcat.argmin(axis = 0) ## MIN distance means MAX similarity
			#print(max_indexes)
			final_list = []
			for indi, row in enumerate(max_indexes):
				for indj, maxindx in enumerate(row):
					final_list.append(Xconcat[maxindx][indi][indj])
			Xfinal = np.array(final_list)			
			Xfinal = Xfinal.reshape(Xdf1.shape)
		elif fusion == "avg":
			# Average Rule Fusion
			print("** Using AVG score fusion")
			Xfinal = (Xdf1 * 0.5) + (Xdf2 * 0.5) 						
		elif fusion == "wsum":
			# Weighted Sum Rule Fusion
			print("** Using WEIGHTED SUM score fusion")
			
			##--> Get weight for weighted sum fusion
			assert annot_csv, "**E: For WEIGHTED SUM fusion, an annotation file is needed!"
			df = pd.read_csv(annot_csv)
			dff = df[df['folder'].isin(y1) & df['file'].isin(y3)]
			#print(y1)
			#print(y3)
			if len(dff.index) != len(y1): ## ToDo By now we are trusting that all data match every sample with every row 
				if verbose: print("** E: Weighted SUM score level fusion, not all data found on vggface-ear_mb_annotations.csv")
				print(f"** data rows: {len(dff.index)} - label rows: {len(y1)}")
				#Xfinal = (Xdf1 * 0.5) + (Xdf2 * 0.5) ## weighted sum					
			arr_pos = df['face_pos'].values
			if verbose: print("len(arr_pos): {} - {}".format(len(arr_pos), arr_pos))
			weights_face = np.array([foo_ws_score(x) for x in arr_pos])
			weights_ear = 1 - weights_face
			if verbose: print("len(weights_ear): {} - {}".format(len(weights_face), weights_face))
			Xfinal = (Xdf1 * weights_face) + (Xdf2 * weights_ear)
		else:
			print("---> Invalid fusion: {}. Available fusions are: {}".format(fusion, ['avg', 'max', 'wsum']))

		### --> Working again with distances
		#Xfinal = 1 - Xfinal 
		df_fused = pd.DataFrame(Xfinal, columns=y1, index=y3)
		dataFrames.append(df_fused)

		### ---> Save CSV file distances
		if save_csvfiles:
			csvFilesFolder = os.path.join(ROOT_DIR, 'csvfiles')
			if not os.path.exists(csvFilesFolder):
				os.mkdir(csvFilesFolder)

			### ---> Build CSV file		
			fused_distancesfilename = os.path.join(csvFilesFolder, distances[i] + '_distances_matrix_fused.csv')
			df_fused.to_csv(fused_distancesfilename, encoding='utf-8', index=True)

	toc = time.time()
	print("---> getScoreFusion() run in: {}".format(time.strftime("%H:%M:%S", time.gmtime(toc-tic))))

	return dataFrames

def getRankAccuracy(df_distances, s_distances, rank = 5,):
	'''
        ** Calculates Rank metric from a table of distances ** 
        Inputs:
            -> df_distances: DataFrame of all-vs-all samples distances
            -> s_distances: list of strings with distances names ('chisquared', 'cosine', 'euclidean', etc)
            -> rank: int The values of RankX will be return
        Outputs:
            -> return rankX values
            -> csvfiles: if 'save_csvfiles' global variable is True, files will be saved in /csvfile folder.
                - matches_table.csv: it gives for every sample, a list of match samples oredered by distance
                - true_matches_table.csv: it gives the position value of true match coincidence (samples from the same class) in the match table 
                - confusion_matrix.csv: it gives what it says =D
                - CMC.csv: it saves info of the cumulative match characteristics metric until the position where all samples in the gallery match the probe    
    '''
	
	tic = time.time()
	print("----> Getting rank accuracy...")
	
	for i, df in enumerate(df_distances):
		### ---> Get X and y values from distance matrix
		X = df.values
		y = np.array((df.columns), dtype=str)
		labels = y.reshape(len(y),1)
		imgs = np.array((df.index), dtype=str)
		imgs = imgs.reshape(len(imgs), 1)

		x = int(np.floor(len(y)*0.1))
		if verbose: print(f'Labels len: {len(labels)} \n --> {labels.T[:x]}')
		if verbose: print(f'Imgs indexes (names) {len(imgs)} \n --> {imgs.T[:x]}')

		### ---> Create a list that fuses labels and imgs names
		labelimg = []
		for ix, label in enumerate(labels):
			r = np.concatenate([labels[ix],['/'], imgs[ix]])
			labelimg.append(''.join(r))
		labelimg = np.array(labelimg)
		labelimg = labelimg.reshape(len(labelimg), 1)    
		if verbose: print(f'Labelimg list: {len(labelimg)} \n --> {labelimg.T[:x]}')
        
        ### ---> Lists for csv files and Ranks
		max_correct_match_position = 0
		matches_table = []
		true_matches_table = []
		ytrue_ypred = []
		cmc_list = []
		rankbins = np.zeros(len(labels))
        
        ### ---> Check every sample as a probe imgs
		for idx, row in enumerate(X):
			if verbose: print(f'Checking: {imgs[idx]}')
			distances = row    ## row contains distances
			distances = distances.reshape(len(row), 1)

			## --> Create a table of distances, labels, imgs names and labelimgs and order it by distances
			newrow = np.concatenate((distances, labels, imgs, labelimg), axis=1)
			newrow = newrow[newrow[:,0].argsort()]

			## --> Get and save matches of every probe
			row_imgsmatch = np.concatenate((labelimg[idx], newrow[:,3].T[1:]), axis=0)
			if verbose: print(f'Matches ordered by  distances: {row_imgsmatch[0]} --> {row_imgsmatch[1:]}')
			matches_table.append(row_imgsmatch[:100]) ## only 100-first matches 

			## --> Get and save positions from matches that belong to the same class (same label)
			coincidence = np.where(newrow[:,1] == y[idx])[0]
			if verbose: print(f'True matches at: {coincidence}')
			true_matches_table.append(coincidence[:100])

			## --> Save values for Ranks --> First coincidence belong to the same sample, so second value is take it
			rankbins[coincidence[1]] += 1
			## --> Saving max rank value ever  --> To lately set the length of CMC list
			if coincidence[1] > max_correct_match_position: max_correct_match_position = coincidence[1]

			## --> Save predicted classes for Confusion Matrix
			ytrue_ypred.append([y[idx], newrow[:,1][1]])

		if verbose: print("-----------")
		#rankbins = rankbins/len(y) ## Get porcentage among the total number of samples		
		final_ranks = np.cumsum(rankbins[:max_correct_match_position+1])
		cmc_list.append(rankbins[:max_correct_match_position+1])
		cmc_list.append(final_ranks)
		cmc_list.append(final_ranks/len(y))
		if verbose: print(f'Max correct match position is: {max_correct_match_position}')
		if verbose: print(f'CMC: \n --> {final_ranks/len(y)}')

		### ---> Build Confusion Matrix
		ytrue_ypred = np.array(ytrue_ypred)
		if verbose: print(f'Ytrue - Ypred: \n {ytrue_ypred}')
		cm = confusion_matrix(ytrue_ypred[:,0], ytrue_ypred[:,1], labels=np.unique(y, return_counts=False))
		if verbose: print(f"Confusion Matrix: shape {cm.shape} \n --> {cm}")

		### ---> Save CSV files
		if save_csvfiles:
			print("Saving CSV files")
			csvFilesFolder = os.path.join(ROOT_DIR, 'csvfiles')
			if not os.path.exists(csvFilesFolder):
				os.mkdir(csvFilesFolder)

			mtFileName = os.path.join(csvFilesFolder, s_distances[i]+'_matches_table.csv')
			df = pd.DataFrame(matches_table)
			df.to_csv(mtFileName, encoding='utf-8', index=False)

			tmtFileName = os.path.join(csvFilesFolder, s_distances[i]+'_true_matches_table.csv')
			df2 = pd.DataFrame(true_matches_table, index=labelimg)
			df2.to_csv(tmtFileName, encoding='utf-8', index=True)

			cmFileName = os.path.join(csvFilesFolder, s_distances[i]+'_confusion_matrix.csv')
			dfcm = pd.DataFrame(cm, columns=np.unique(y, return_counts=False), index=np.unique(y, return_counts=False))
			dfcm.to_csv(cmFileName, encoding='utf-8', )

			cmcFileNamer = os.path.join(csvFilesFolder, s_distances[i]+'_CMC.csv')
			#dfcmc = pd.DataFrame(np.expand_dims(final_ranks, axis=0))
			dfcmc = pd.DataFrame(cmc_list, index=range(len(cmc_list)))
			dfcmc.to_csv(cmcFileNamer, encoding='utf-8', index=False)

		print(f'Distance: {s_distances[i]} - Ranks: {final_ranks[1:rank+1]/len(y)}')
	
	toc = time.time()
	print("---> getRankAccuracy() run in: {}".format(time.strftime("%H:%M:%S", time.gmtime(toc-tic))))

	return

def getRankAccuracy2(df_distances, s_distances, rank = 5):
	'''
        ** Calculates Rank metric from a table of distances ** 
        Inputs:
            -> df_distances: DataFrame of all-vs-all samples distances
            -> s_distances: list of strings with distances names ('chisquared', 'cosine', 'euclidean', etc)
            -> rank: int The values of RankX will be return            
        Outputs:
            -> return rankX values
            -> csvfiles: if 'save_csvfiles' global variable is True, files will be saved in /csvfile folder.
                - matches_table.csv: it gives for every sample, a list of match samples oredered by distance
                - true_matches_table.csv: it gives the position value of true match coincidence (samples from the same class) in the match table 
                - confusion_matrix.csv: it gives what it says =D
                - CMC.csv: it saves info of the cumulative match characteristics metric until the position where all samples in the gallery match the probe    
    '''
	
	tic = time.time()
	print("----> Getting rank accuracy...")
	
	for i, df in enumerate(df_distances):
		### ---> Get X and y values from distance matrix
		X = df.values
		y = np.array((df.columns), dtype=str)

		labels = y.reshape(len(y),1)
		unique_labels = np.unique(y)
		imgs = np.array((df.index), dtype=str)
		imgs = imgs.reshape(len(imgs), 1)

		print(f"Data Size ({len(X)},{len(X[0])})")
				
		x = int(np.floor(len(y)*0.1))
		if verbose: print(f'Labels len: {len(labels)} \n --> {labels.T[:x]}')
		if verbose: print(f'Imgs indexes (names) {len(imgs)} \n --> {imgs.T[:x]}')

		### ---> Create a list that fuses labels and imgs names
		labelimg = []
		for ix, label in enumerate(labels):
			r = np.concatenate([labels[ix],['/'], imgs[ix]])
			labelimg.append(''.join(r))
		labelimg = np.array(labelimg)
		labelimg = labelimg.reshape(len(labelimg), 1)    
		if verbose: print(f'Labelimg list: {len(labelimg)} \n --> {labelimg.T[:x]}')
        
        ### ---> Lists for csv files and Ranks		
		matches_table = []
		true_matches_table = []
		ytrue_ypred = []
		cmc_list = []		
		new_rankbins = np.zeros(len(unique_labels)+1)
        
        ### ---> Check every sample as a probe imgs
		for idx, row in enumerate(X):
			distance_class = []
			if verbose: print(f'Checking: {imgs[idx]}')
			distances = row    ## row contains distances
			distances = distances.reshape(len(row), 1)
			
			## --> Create a table of distances, labels, imgs names and labelimgs and order it by distances
			newrow = np.concatenate((distances, labels, imgs, labelimg), axis=1)
			newrow = newrow[newrow[:,0].argsort()]
			
			## --> Get and save matches of every probe
			row_imgsmatch = np.concatenate((labelimg[idx], newrow[:,3].T[1:]), axis=0)
			if verbose: print(f'Matches ordered by  distances: {row_imgsmatch[0]} --> {row_imgsmatch[1:]}')
			matches_table.append(row_imgsmatch[:100]) ## only 100-first matches 

			## --> Get and save positions from matches that belong to the same class (same label)
			coincidence = np.where(newrow[:,1] == y[idx])[0]
			if verbose: print(f'True matches at: {coincidence}')
			true_matches_table.append(coincidence[:100])
			
			## --> Get new ranks
			for l in unique_labels:
				coincidence_label = np.where(newrow[1:,1] == l)[0] #First element of newrow is the same sample                
				distance_class.append((l, coincidence_label[0]))

			## --> Sort class according to distance positions
			distance_class.sort(key=lambda x:x[1])
			dclass = [dc[0] for dc in distance_class] 

			## --> Save values for Ranks... starting at 1
			new_coincidence = dclass.index(y[idx])
			new_rankbins[new_coincidence+1] +=1
			
			## --> Save predicted classes for Confusion Matrix
			ytrue_ypred.append([y[idx], newrow[:,1][1]])

		if verbose: print("-----------")
		if verbose: print(f'Rankbins: {new_rankbins}')		
		final_ranks = np.cumsum(new_rankbins)
		cmc_list.append(new_rankbins)
		cmc_list.append(final_ranks)
		cmc_list.append(final_ranks/len(y))				
		if verbose: print(f'CMC: \n --> {final_ranks/len(y)}')

		### ---> Build Confusion Matrix
		ytrue_ypred = np.array(ytrue_ypred)
		if verbose: print(f'Ytrue - Ypred: \n {ytrue_ypred}')
		cm = confusion_matrix(ytrue_ypred[:,0], ytrue_ypred[:,1], labels=np.unique(y, return_counts=False))
		if verbose: print(f"Confusion Matrix: shape {cm.shape} \n --> {cm}")

		### ---> Save CSV files
		if save_csvfiles:
			print("----> Saving CSV files...")
			csvFilesFolder = os.path.join(ROOT_DIR, 'csvfiles')
			if not os.path.exists(csvFilesFolder):
				os.mkdir(csvFilesFolder)

			mtFileName = os.path.join(csvFilesFolder, s_distances[i]+'_matches_table.csv')
			df = pd.DataFrame(matches_table)
			df.to_csv(mtFileName, encoding='utf-8', index=False)

			tmtFileName = os.path.join(csvFilesFolder, s_distances[i]+'_true_matches_table.csv')
			df2 = pd.DataFrame(true_matches_table, index=labelimg)
			df2.to_csv(tmtFileName, encoding='utf-8', index=True)

			cmFileName = os.path.join(csvFilesFolder, s_distances[i]+'_confusion_matrix.csv')
			dfcm = pd.DataFrame(cm, columns=np.unique(y, return_counts=False), index=np.unique(y, return_counts=False))
			dfcm.to_csv(cmFileName, encoding='utf-8', )

			cmcFileNamer = os.path.join(csvFilesFolder, s_distances[i]+'_CMC.csv')
			#dfcmc = pd.DataFrame(np.expand_dims(final_ranks, axis=0))
			dfcmc = pd.DataFrame(cmc_list, index=range(len(cmc_list)))
			dfcmc.to_csv(cmcFileNamer, encoding='utf-8', index=False)

		print(f'Distance: {s_distances[i]} - Ranks: {final_ranks[1:rank+1]/len(y)}')
		print(f'AUC: {auc(range(1,len(final_ranks)),(final_ranks[1:]/len(y))/len(unique_labels))}')

	toc = time.time()
	print("---> getRankAccuracy() run in: {}".format(time.strftime("%H:%M:%S", time.gmtime(toc-tic))))

	return

def getRankMetric(files_paths, descriptor, distances, load_predictions=None, load_matrix_distances=None, csvfiles=False, pverbose=False):
	
	### ---> Help printing files of each step
	global save_csvfiles
	save_csvfiles = csvfiles
	global verbose
	verbose = pverbose
	
	if len(files_paths) == 1:
		features = getPredictions(files_paths[0], descriptor, load_predictions)
	else:
		features = getPredictionsMS(files_paths, descriptor)

	df_distances = getMatchingMatrix(features, distances, load_matrix_distances)
	ranks = getRankAccuracy2(df_distances,distances)
	
	return

def getRankMetric_ScoreFusion(files_paths, descriptors, distances,  fusion="max", annot_csv=None, load_predictions=None, csvfiles=False, pverbose=False):
	
	### ---> Help printing files of each step
	global save_csvfiles
	save_csvfiles = csvfiles
	global verbose
	verbose = pverbose

	if not len(files_paths) == 2 and len(descriptors) == 2:
		raise ValueError('**Two paths and two descriptors are needed for fusion.')	

	if load_predictions != None:
		face_features = getPredictions(files_paths[0], descriptors[0], load_predictions[0])
		ear_features = getPredictions(files_paths[1], descriptors[1], load_predictions[1])
	else:
		face_features = getPredictions(files_paths[0], descriptors[0])
		ear_features = getPredictions(files_paths[1], descriptors[1])

	df_distances_face = getMatchingMatrix(face_features, distances)
	df_distances_ear = getMatchingMatrix(ear_features, distances)

	df_fused = getScoreFusion(df_distances_face, df_distances_ear, distances, fusion, annot_csv)

	ranks = getRankAccuracy2(df_fused, distances)
		
	return