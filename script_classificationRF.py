import numpy as np
import gdal
import pandas as pd
#from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
import time
import pickle
import os
import sys

   ##developed by Daniel Moraes
   ##moraesd90@gmail.com







#set time function to get track of time
def the_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    return current_time

#function to get values (Using Local Varibles increases speed)
#stacks multiple arrays into one array then reshapes result
def getvalues(array,row,col,bands):
    print(the_time(),'----- Stacking')
    array2 = np.stack(array, axis=2)
    print(the_time(),'----- Reshaping')
    arrayfinal = np.reshape(array2, [row*col,bands])
    
    return arrayfinal


#function to open block according to blocksize
def read_block(ds,b,col,block_size):
    
    array = ds.ReadAsArray(0,b,col,block_size)

    return array

#function to create a GeoTIFF file with the given data
def createGeotiff(outRaster, c1, c2, geo_transform, projection, nodata):

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = c1.shape
    rasterDS = driver.Create(outRaster, cols, rows, 2, gdal.GDT_Int32)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    rasterDS.GetRasterBand(1).SetNoDataValue(nodata) 
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(c1)
    rasterDS.GetRasterBand(2).SetNoDataValue(nodata) 
    band = rasterDS.GetRasterBand(2)
    band.WriteArray(c2)
    rasterDS = None


#create probabilities raster
def createGeotiff2(outRaster2, data, geo_transform, projection, rows, cols, nodata):
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    #rows, cols = data.shape
    rasterDS = driver.Create(outRaster2, cols, rows, 1, gdal.GDT_Float32)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    rasterDS.GetRasterBand(1).SetNoDataValue(nodata) 


    band = rasterDS.GetRasterBand(1)
    #band.WriteArray(data[30-i].reshape((rows,cols)))
    band.WriteArray(data.reshape((rows,cols)))
    

##    band1 = rasterDS.GetRasterBand(1)
##    band1.WriteArray(final[0].reshape((rows,cols)))
##
##  
##    band2 = rasterDS.GetRasterBand(2)
##    band2.WriteArray(final[1].reshape((rows,cols)))
##
##    band3 = rasterDS.GetRasterBand(3)
##    band3.WriteArray(final[2].reshape((rows,cols)))
    rasterDS = None
    



def getColumns():
    
    df = pd.read_csv(csv_file, sep=',',nrows=1)
    columns = df.columns
    del df
    return columns


def runScript():
    
   #set log filename and save printouts to file 
   log = open('log_'+os.path.basename(__file__)[:-3]+'_'+time.strftime("%Y-%m-%d_%H%M", time.localtime())+'.txt','w')
   class Unbuffered:

      def __init__(self, stream):

          self.stream = stream

      def write(self, data):

          self.stream.write(data)
          self.stream.flush()
          log.write(data)    # Write the data of stdout here to a text file as well
          log.flush() 
   stdout_original = sys.stdout 
   sys.stdout=Unbuffered(sys.stdout)


   print('----------')
   print(time.strftime("%H:%M:%S", time.localtime()),'--- STARTING SCRIPT ---')
   print(time.strftime("%H:%M:%S", time.localtime()),'---- Reading input data')
   tstart = time.time()


   ################SET YOUR INPUTS BELOW##################
   #######################################################


   #define original raster dataset path (eg. 'C:/Users/YourUser/Desktop/Classification/image.tif')
   #fn = 'C:/Users/Daniel/Desktop/NOVA IMS/Thesis/DGT/Sampling/testes_classificacao_nodata/teste_nodata.tif'
   #fn = 'D:/Mestrado/Sentinel 2 Composite/S2data_121.tif'
   #fn = 'E:/Mestrado/Sentinel 2/S2_estrato_4_v3.tif'
   #fn = 'Z:/Users/morae/Desktop/FTP/DGT_to_Daniel/S2data_121.tif'
   #define classified raster path 
   #outRaster = 'C:/Users/Daniel/Desktop/NOVA IMS/Thesis/DGT/Sampling/testes_classificacao_nodata/OUTPUT_estrato4_class.tif'
   #define number of blocks (must be greater than 1)
   #splitting dataset in blocks helps fitting data into memory when working with very large images
   #blocks = 11104 ## change block size here


   print(time.strftime("%H:%M:%S", time.localtime()),'---- Input data read')

   ############END OF SETTING INPUTS######################
   #######################################################







   #get raster dimensions
   ds = gdal.Open(fn, gdal.GA_ReadOnly)
   col = ds.RasterXSize
   row_total = ds.RasterYSize
   bands = int(ds.RasterCount)
   band = ds.GetRasterBand(1)
   no_data = band.GetNoDataValue()
   ds = None
   band = None
   block_size = int(row_total/blocks)
   if row_total - block_size*blocks >= block_size:
       print('!!!!!! Warning: first block might be too large !!!!!!')

   if blocks > row_total:
       print('!!!!!! Warning: Number of blocks cannot be larger than '+str(row_total))
       raise NameError('Change number of blocks - Number of blocks cannot be larger than '+str(row_total))

   
   #print information
   print(the_time(), '----------')
   print(the_time(), '--- Run information ---')
   print(the_time(), '---- Running with number of blocks =', blocks)
   print(the_time(), '---- Input raster:',os.path.basename(fn))
   print(the_time(), '---- Raster dimensions: x,y =', col,',',row_total)
   print(the_time(), '---- Raster bands:', bands)
   print(the_time(), '----------')


   #Set classifier
   print(the_time(),'---- Starting classification')
   #check if user wants to train classifier or import from pickle
   if clfMode == 1:
       print(the_time(),'---- Classification Mode: Train Classifier')
       print(the_time(),'---- Setting classifier parameters')
       #import training dataset
       #define training dataset (in csv format)
       #define training data path, separator and dtype
       #defining proper dtype contributes to reduce memory used (define dtype as: dtype = {'Band1':'int16', 'Band2':'int16'...} )where Bandx are columns of csv file
       #define csv separator (eg. for comma use sep=',')
       df = pd.read_csv(treino, sep=',')
       #define dataframe composed only by bands columns (eg df[['Band1', 'Band2', 'Band3'...]]) according to csv columns
       #data = df[['S2_1','S2_2','S2_3','S2_4','S2_5','S2_6','S2_7','S2_8','S2_9','S2_10','S2_11','S2_12','S2_13','S2_14','S2_15','S2_16','S2_17','S2_18','S2_19','S2_20','S2_21','S2_22','S2_23','S2_24','S2_25','S2_26','S2_27','S2_28','S2_29','S2_30','S2_31','S2_32','S2_33','S2_34','S2_35','S2_36','S2_37','S2_38','S2_39','S2_40','S2_41','S2_42','S2_43','S2_44','S2_45','S2_46','S2_47','S2_48','S2_49','S2_50','S2_51','S2_52','S2_53','S2_54','S2_55','S2_56','S2_57','S2_58','S2_59','S2_60','S2_61','S2_62','S2_63','S2_64','S2_65','S2_66','S2_67','S2_68','S2_69','S2_70','S2_71','S2_72','S2_73','S2_74','S2_75','S2_76','S2_77','S2_78','S2_79','S2_80','S2_81','S2_82','S2_83','S2_84','S2_85','S2_86','S2_87','S2_88','S2_89','S2_90','S2_91','S2_92','S2_93','S2_94','S2_95','S2_96','S2_97','S2_98','S2_99','S2_100','S2_101','S2_102','S2_103','S2_104','S2_105','S2_106','S2_107','S2_108','S2_109','S2_110','S2_111','S2_112','S2_113','S2_114','S2_115','S2_116','S2_117','S2_118','S2_119','S2_120','S2_121','S2_122','S2_123','S2_124','S2_125','S2_126','S2_127','S2_128','S2_129','S2_130','S2_131','S2_132','S2_133','S2_134','S2_135','S2_136','S2_137','S2_138','S2_139','S2_140','S2_141','S2_142','S2_143','S2_144','S2_145','S2_146','S2_147','S2_148','S2_149','S2_150','S2_151','S2_152','S2_153','S2_154','S2_155','S2_156','S2_157','S2_158','S2_159','S2_160','S2_161','S2_162','S2_163','S2_164','S2_165','S2_166','S2_167','S2_168','S2_169','S2_170','S2_171','S2_172','S2_173','S2_174','S2_175','S2_176','S2_177','S2_178','S2_179','S2_180','S2_181','S2_182','S2_183','S2_184','S2_185','S2_186','S2_187','S2_188','S2_189','S2_190','S2_191','S2_192','S2_193','S2_194','S2_195','S2_196','S2_197','S2_198','S2_199','S2_200','S2_201','S2_202','S2_203','S2_204','S2_205','S2_206','S2_207','S2_208','S2_209','S2_210','S2_211','S2_212','S2_213','S2_214','S2_215','S2_216','S2_217','S2_218','S2_219','S2_220','S2_221','S2_222','S2_223','S2_224','S2_225','S2_226','S2_227','S2_228','S2_229','S2_230','S2_231','S2_232','S2_233','S2_234','S2_235','S2_236','S2_237','S2_238','S2_239','S2_240','S2_241','S2_242','S2_243','S2_244','S2_245','S2_246','S2_247','S2_248','S2_249','S2_250','S2_251','S2_252','S2_253','S2_254','S2_255','S2_256','S2_257','S2_258','S2_259','S2_260','S2_261','S2_262','S2_263','S2_264','S2_265','S2_266','S2_267','S2_268','S2_269','S2_270','S2_271','S2_272','S2_273','S2_274','S2_275','S2_276','S2_277','S2_278','S2_279','S2_280','S2_281','S2_282','S2_283','S2_284','S2_285'
       #]]
       data = df[variables]
       #define labels (actual class values) according to csv label column
       #class values must be integer, not string (11111 -> correct; Built up -> incorrect)
       #label = df['COD']
       label = df[labelCod]
     
       #train classifier
       #initialize classifier
       clf1 = RandomForestClassifier()
       #set parameters
       if estimators_user:
           clf1.set_params(n_estimators=int(estimators_user))
       if maxfeatures_user:
           if max_features_user in ['auto','sqrt','log2']:
               clf1.set_params(max_features_user)
           if '.' in max_features_user:
               clf1.set_params(float(max_features_user))
           else:
               clf1.set_params(int(max_features_user))
       if njobs_user:
           clf1.set_params(n_jobs=int(njobs_user))
       if bootstrap_user:
           if bootstrap_user == 'True':
               clf1.set_params(bootstrap=True)
           if bootstrap_user == 'False':
               clf1.set_params(bootstrap=False)
       if criterion_user in ['gini','entropy']:
           clf1.set_params(criterion=criterion_user)
       if max_depth_user:
           clf1.set_params(max_depth=int(max_depth_user))
       if minsamplessplit_user:
           if '.' in minsamplessplit_user:
               clf1.set_params(min_samples_split=float(minsamplessplit_user))
           else:
               clf1.set_params(min_samples_split=int(minsamplessplit_user))
       if minsamplesleaf_user:
           if '.' in minsamplesleaf_user:
               clf1.set_params(min_samples_leaf=float(minsamplesleaf_user))
           else:
               clf1.set_params(min_samples_leaf=int(minsamplesleaf_user))
       if minweightleaf_user:
           clf1.set_params(min_weight_fraction_leaf=float(minweightleaf_user))
       if maxleafnodes_user:
           clf1.set_params(max_leaf_nodes=int(maxleafnodes_user))
       if minimpdecrease_user:
           clf1.set_params(min_impurity_decrease=float(minimpdecrease_user))
       if minimpsplit_user:
           clf1.set_params(min_impurity_split=float(minimpsplit_user))
       if oobscore_user:
           if oobscore_user == 'True':
               clf1.set_params(oob_score=True)
           if oobscore_user == 'False':
               clf1.set_params(oob_score=False)
       #if randomstate_user:
           #to be implemented
       if verbose_user:
           clf1.set_params(verbose=int(verbose_user))
       if warmstart_user:
           if warmstart_user == 'True':
               clf1.set_params(warm_start=True)
           if warmstart_user == 'False':
               clf1.set_params(warm_start=False)
       #if classweight_user:
           #to be implemented
       if ccpalpha_user:
           clf1.set_params(ccp_alpha=float(ccpalpha_user))
       if maxsamples_user:
           if '.' in maxsamples_user:
               clf1.set_params(max_samples=float(maxsamples_user))
           else:
               clf1.set_params(max_samples=int(maxsamples_user))
 
       #clf1 = RandomForestClassifier(n_estimators=300,bootstrap=False,criterion='entropy',n_jobs=-1) ## set classifier parameters. For more info read sklearn RandomForestClassifier documentation
       print(the_time(), '---- Classifier set with the following parameters:')
       print(the_time(), clf1.get_params())
       print(the_time(), '---- Starting training')
       clf1.fit(data,label)
       data = None
       label = None
       df = None
       print(the_time(),'---- Training finished')
   #alternatively, you can import the classifier from pickle
   else:
       print(the_time(),'---- Classification Mode: Import Classifier')
       #'C:/Users/Daniel/Desktop/NOVA IMS/Thesis/DGT/Sampling/testes_classificacao_nodata/clf_cortes2015-2018.pickle'
       pickle_in = open(clfPickle, "rb")
       clf1 = pickle.load(pickle_in)
       del pickle_in
       print(the_time(),'---- Successfully imported classifier')
        

   #get number of classes for each classifier
   n_classes = [clf1.n_classes_]


   print(the_time(), '--- Starting Reading Blocks ---')
   #opening blocks
   ds = gdal.Open(fn, gdal.GA_ReadOnly)
   i = 0
   rest = False
   value = 0
   if block_size*blocks < row_total:
       rest = True
       value = row_total-(block_size*blocks)


   while i < blocks:
       print(the_time(), '---- Reading Block', i+1)
       if rest == True:
           if i == 0:
               test = pd.DataFrame(getvalues(read_block(ds,0,col,(block_size+value)),block_size+value,col,bands), dtype='int16')
               test = test[(test!=no_data).all(axis=1)]
           else:
               test = pd.DataFrame(getvalues(read_block(ds,i*block_size+value,col,block_size),block_size,col,bands), dtype='int16')
               test = test[(test!=no_data).all(axis=1)]
               #test = test.append(block, ignore_index=True)
               #del block
               
       else:
           if i == 0:
               test = pd.DataFrame(getvalues(read_block(ds,0,col,block_size),block_size,col,bands), dtype='int16')
               test = test[(test!=no_data).all(axis=1)]
           else:
               test = pd.DataFrame(getvalues(read_block(ds,i*block_size,col,block_size),block_size,col,bands), dtype='int16')
               test = test[(test!=no_data).all(axis=1)]
               #test = test.append(block, ignore_index=True)
               #del block
       print(the_time(), '---- Finished Reading Block', i+1)
       print(the_time(), '----------')
       print(the_time(),'----- Prediction for block', i+1)

       if i == 0:
           if test.size!= 0:
               #y_pred1 = clf1.predict(test).astype('int32')
               pred_prob1 = clf1.predict_proba(test).astype('float32')
               y_pred1_c1 = clf1.classes_[np.argpartition(pred_prob1,-2)[:,-1]].astype('int32')
               y_pred1_c2 = clf1.classes_[np.argpartition(pred_prob1,-2)[:,-2]].astype('int32')
               pred_prob_sort = np.sort(pred_prob1)
               p1p2 = np.take(pred_prob_sort, int(n_classes[0])-1, axis=1) - np.take(pred_prob_sort, int(n_classes[0])-2, axis=1)
               del pred_prob1,pred_prob_sort
               

       else:
           if test.size!= 0:
               #y_pred1 = np.append(y_pred1,clf1.predict(test).astype('int32'))
               #pred_prob1 = np.append(pred_prob1,clf1.predict_proba(test).astype('float32'))
               pred_prob1 = clf1.predict_proba(test).astype('float32')
               y_pred1_c1 = np.append(y_pred1_c1,clf1.classes_[np.argpartition(pred_prob1,-2)[:,-1]].astype('int32'))
               y_pred1_c2 = np.append(y_pred1_c2,clf1.classes_[np.argpartition(pred_prob1,-2)[:,-2]].astype('int32'))
               pred_prob_sort = np.sort(pred_prob1)
               p1p2 = np.append(p1p2,np.take(pred_prob_sort, int(n_classes[0])-1, axis=1) - np.take(pred_prob_sort, int(n_classes[0])-2, axis=1))
               del pred_prob1,pred_prob_sort
           
       print(the_time(),'----- Prediction for block', i+1,'completed')

       del test

       
       i = i + 1
   del clf1

   ds = None
   print(the_time(),'---- Classification completed')

   print(the_time(), '----------')
   ##Set Environment to create classified raster
   #set input and output raster location
   inpRaster = fn
   rasterds = gdal.Open(inpRaster, gdal.GA_ReadOnly)
   band = rasterds.GetRasterBand(1)
   print(the_time(),'--- Getting raster dataset spatial reference ---')
   #Get spatial reference
   geo_transform = rasterds.GetGeoTransform()
   projection = rasterds.GetProjectionRef()
   rows = rasterds.RasterYSize
   cols = rasterds.RasterXSize
   print(the_time(),'--- Done')




   #set training data
   print(the_time(), '----------')




   
   print(the_time(), '----------')
   # Create a GeoTIFF file with the given data
   print(the_time(), '---- Creating raster of classification ----')
   time.sleep(2)

   #get non nodata indices
   
   flags = band.ReadAsArray()
   
   flags = flags.astype('int32')
   flags = flags.flatten()
   
   indices = np.where(flags != no_data)[0]
   
   flags[indices] = np.array(y_pred1_c1.flatten())
   
   del y_pred1_c1
   y_pred_c1_final = np.array(flags)
   
   flags[indices] = np.array(y_pred1_c2.flatten())
   
   del y_pred1_c2
   y_pred_c2_final = np.array(flags)
   

   c1 = y_pred_c1_final.reshape((rows,cols))
   c2 = y_pred_c2_final.reshape((rows,cols))
   
   createGeotiff(outRaster, c1, c2, geo_transform, projection, no_data)

   del c1,c2,y_pred_c1_final,y_pred_c2_final
   print(the_time(),'---- Raster Created')
   #predict class probabilities
   print(the_time(),'----------')


   print(the_time(),'---- Creating probabilities raster')
   #print(the_time(),'--- Splitting array')

   #pred_prob1 = np.split(pred_prob1,len(pred_prob1)/int(n_classes[0]))  ##change here the number of classes (31)


   #print(the_time(),'--- Done splitting array')
   #print(the_time(),'--- Stacking array')
   #pred_prob1 = np.sort(pred_prob1)
   #pred_prob1 = np.stack(pred_prob1,axis=1)
   flags = flags.astype('float32')
   #get highest probability
   flags[indices] = p1p2
   pred_prob = np.array(flags)
   del p1p2, flags, indices
   #get second-highest probability
   #flags[indices] = np.array(pred_prob1[int(n_classes[0])-2])

   #pred_prob.append(np.array(flags))
   #get third-highest probability
   #flags[indices] = np.array(pred_prob1[int(n_classes[0])-3])
   #pred_prob.append(np.array(flags))

   #print(the_time(),'--- Done stacking array')
   #del pred_prob


   #final = np.stack(final, axis=1)





   #outRaster2 = 'C:/Users/Daniel/Desktop/NOVA IMS/Thesis/DGT/Sampling/testes_classificacao_nodata/output_proba_estrato4.tif'


   createGeotiff2(outRaster2, pred_prob, geo_transform, projection,row_total,col, no_data)

   #createGeotiff2(outRaster2, pred_prob, geo_transform, projection,row_total,col)
   print(the_time(),'---- Raster created')
   print(the_time(),'---- Class output:',os.path.basename(outRaster))
   print(the_time(),'---- Probabilities output:',os.path.basename(outRaster2))
   print(the_time(),'---')
   print(the_time(),'---')
   tend = time.time()
   print(the_time(),'--- [[[CLASSIFICATION FINISHED. Execution took', round((tend-tstart)/60,2), 'minutes]]] ---')
   #finished = input('Press Enter to Exit')
   log.close()
   sys.stdout = stdout_original
   
   







