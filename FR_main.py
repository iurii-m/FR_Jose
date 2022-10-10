# -*- coding: utf-8 -*-
"""
Demo Face Recognition functionality.
Include 2 simple functions (extract_embedding, match_embeddings) for extracting 
face embeddings (deep representations) and matching of these embeddings.

Model - small MobileNet_V2
Performance of the model on LFW Dataset:
FMR         0.1000	 0.0100	  0.001  0.0001  1E-05	1E-06
FNMR        0.0013	 0.0060	  0.015  0.0540  0.054	0.054
threshold   0.3654	 0.4891	  0.571  0.6572  0.657	0.657


Performance on a custom ICAO-Compliant  benchmark
EER -  0.00189 at threshold ~ 0.75
FMR         0.1     0.01        0.001   0.0001      1e-05   1e-06
FNMR        0.0     7.4e-05     0.0053  0.03957     0.0965  0.2184
threshold   0.42    0.6694      0.7967  0.85627     0.8874  0.9161


For folder processing the default image extentions:  [".jpg",".bmp",".png"]


#Example of usage:

#Just Run Demo
    python FR_main.py

#Extract single image embedding
    python FR_main.py -m s_im
    python FR_main.py -m s_im -i {image_path} -ie {path_embedding}
  
#Extract embeddings from a directory with images
    python FR_main.py -m s_im
    python FR_main.py -m s_im -d {dataset_path} -de {embedding_of_dataset}

#match embeddings  
    python FR_main.py -m m_emb  
    python FR_main.py -m m_emb -e1 {embedding_1} -e2 {embedding_2}

"""

__author__ = "Iurii Medvedev"


import numpy as np
import argparse
import os
import cv2
import warnings
import glob, random
import tensorflow.keras
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import skimage.transform
from mtcnn.mtcnn import MTCNN
from fr_utils import image_crop_center, image_preprocessed, image_normalize
from alignment_insf import _read_image_2_cvMat, _align_faces, _align_face_insightface


# CPU only deployment setting  
os.environ['CUDA_VISIBLE_DEVICES']= "-1"


"""Default parameters definition"""

default_model_path = r"./models/frcv_model.tflite"
default_embedding_size = 128

default_im_size = (224,224,3)
default_sub_mean = [0.56863784, 0.43306758, 0.3709035]
default_cropp_percentage = 1.0
default_norm_val = 1.0

default_if_detect = True

default_threshold = 0.75


    
def extract_embedding(image, 
                      fr_model_tflite,
                      
                      # image preprocessing settings
                      sub_mean = default_sub_mean,
                      cropp_percentage = default_cropp_percentage,
                      im_size = default_im_size,
                      norm_val = default_norm_val,
                      
                      #face detection settings
                      if_detect = default_if_detect,
                      detector = None,
                      landmarks = []):
    """
    Extracts face embedding from the input face image.
    :param image: np.ndarray. Values - uint8 in the range [0,255]. 3 Channels with order - RGB!!! Be Carefull with OpenCV. 
    :param fr_model_tflite: tflite model.
    :param sub_mean: list of floats. order of channels (RGB). Default = default_sub_mean.
    :param cropp_percentage: float. Rate to crop the aligned face. Default = default_cropp_percentage.
    :param im_size: Tuple (width, height, channels). Input image size. Default = default_im_size.
    :param norm_val: float. Normalizing value. Default = default_norm_val.
    :param if_detect: bool. If the face detection with the detector is required. Default = default_if_detect.
    :param detector: The detector must be set in case of if_detect = True. Default = None.
    :param landmarks: list of lists or np.ndarray. Facial landmarks. 
                    The landmarks must be set as a 5-element list in case of if_detect = False. 
                    Order of the landmarks (viewer perspective): left_eye, right_eye,
                    nose, left_mouth_corner, right_mouth_corner. Default = [].
    :return: np.ndarray - normalized face embedding.
    """
   
    
    face_image = None
    
    if if_detect:
        if not detector:
            raise ValueError('Face detector is empty')
        
        #detection and alignment
        result, number_of_detected, main_idx, landmarks, _ = _align_faces(image, detector, (im_size[0],im_size[1]))
        
        #check the detection result
        if number_of_detected<1:
            raise ValueError('No faces detected on the input image')
        #choosing the main detected face
        face_image = (result[main_idx]).astype(np.float32)
        
    else:
        if len(landmarks)<5:
            raise ValueError('Input landmarks list is too short')
        #alignment
        face_image  = _align_face_insightface(image, landmarks, (im_size[0],im_size[1]))

    #debug the aligned image
    #cv2.imshow(str(random.randint(0,10000)), cv2.cvtColor(face_image.astype(np.uint8) , cv2.COLOR_RGB2BGR))
        
        
    #Normalizing to the range [0.0,1.0]
    face_image = face_image/255.0
    
    
    # get the input-output shapes and types
    input_details = fr_model_tflite.get_input_details()
    output_details = fr_model_tflite.get_output_details()
    # print('Input details', input_details, '\n', 'Output details', output_details)
     
    # image preprocessing   
    image1_pr = image_normalize(
                    skimage.transform.resize(
                        image_crop_center(
                            face_image, 
                            cropp_percentage),
                        im_size),
                    sub_mean = sub_mean, 
                    norm_val = norm_val)
    
    
    image1_pr =  image1_pr.astype(np.float32) 
    #print("input shape", image1_pr.shape)
    
    #expanding dimention to match the network input
    batch_image1_pr  = np.expand_dims(image1_pr , axis=0)
    fr_model_tflite.set_tensor(input_details[0]['index'], batch_image1_pr)
    
    # running inference
    fr_model_tflite.invoke()
    
    #getting the output
    output_data = fr_model_tflite.get_tensor(output_details[0]['index'])
    result1 = np.squeeze(output_data)

    
    #normalising the result
    n_result1 = result1 / np.linalg.norm(result1)    
    # print("result 1 shape", result1.shape,"n result 1 shape", n_result1.shape)  # result[0].shape in case of multiple output tensors
  
    return n_result1


def match_embeddings(embedding_1,
                     embedding_2,
                     verification_threshold = default_threshold):
    """
    Matches 2 feature embeddings.
    :param embedding_1: np.ndarray or list of numericals.  The first face feature embedding.
    :param embedding_2: np.ndarray or list of numericals.  The second face feature embedding.
    :return: Tuple (verification_decision, similarity) SIMILARITY RANGE IS [-1; 1]. Higher value means higher similarity.
    """
    

    try:
        np_embedding_1 = np.array(embedding_1)
        np_embedding_2 = np.array(embedding_2)
        
        if not (np_embedding_1.size == np_embedding_2.size):
            warnings.warn("Warning! Size of face embedings doesnt match!")                        
        
        if (np_embedding_1.size == 0) or (np_embedding_2.size == 0):
            warnings.warn("Warning! Size of some face embedings is zero!") 
            
    except:
        raise ValueError('Cant transform one of the face embeddings to np.array')
        
    n_np_embedding_1 = np_embedding_1 / np.linalg.norm(np_embedding_1)
    n_np_embedding_2 = np_embedding_2 / np.linalg.norm(np_embedding_2)
        
    # LEGAL SIMILARITY RANGE IS [-1.0; 1.0] 
    # Normalize if it is needed -> e.g to [0.0; 1.0] -> similarity = (similarity+1)/2. 
    # However this will lead to nececity of redifininition of the verification_threshold
    similarity = np.dot(n_np_embedding_1, n_np_embedding_2)
    
    verification_decision = similarity>verification_threshold
    
    return (verification_decision, similarity)




# Unit Test   
def unit_test_fr(path2images: str,
                        image_1_path: str,
                        image_2_path: str,
                        threshold: float,
                        if_detect: bool,
                        landmarks_1,
                        landmarks_2,
                        target_similarity: float):
    
    
    

    fr_mode_interpreter = tf.lite.Interpreter(default_model_path)
    fr_mode_interpreter.allocate_tensors()
       
    
    #Detector Instance
    detector = None
    if if_detect:
        detector = MTCNN()
    
    #Loading images wit to mat opencv
    image1 = _read_image_2_cvMat(path2images+image_1_path)
    image2 = _read_image_2_cvMat(path2images+image_2_path)
         
    
    #Show detected images
    # cv2.imshow(image_1_path, cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    # cv2.imshow(image_2_path, cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))

    embedding_1 = extract_embedding(image1, 
                          fr_model_tflite = fr_mode_interpreter,
                          sub_mean = default_sub_mean,
                          cropp_percentage = default_cropp_percentage,
                          im_size = default_im_size,
                          norm_val = default_norm_val,
                          if_detect = if_detect,
                          detector = detector,
                          landmarks = landmarks_1)
      
    embedding_2 = extract_embedding(image2, 
                          fr_model_tflite = fr_mode_interpreter,
                          sub_mean = default_sub_mean,
                          cropp_percentage = default_cropp_percentage,
                          im_size = default_im_size,
                          norm_val = default_norm_val,
                          if_detect = if_detect,
                          detector = detector,
                          landmarks = landmarks_2)
    
    print('Comparison of ', image_1_path , 'and' , image_2_path)
    
    print('embedding_1', embedding_1)
    print('embedding_2', embedding_2) 

    verification_decision, similarity = match_embeddings(embedding_1,
                                                         embedding_2,
                                                         default_threshold)
    
    print('Similarity of ', image_1_path , 'and' , image_2_path , 'is' , similarity,'. Target similarity is ', target_similarity,  'Verificaiton decision is ', verification_decision) 
    
    if (abs(similarity-target_similarity))<0.05:
        print('Similarity is close to the target - PASSED')
    else:
        print('Similarity is NOT close to the target - NOT PASSED')

##DEMOS:

def demo_full():

    print('Full demo mode executed')
    #image pathes image 1 and 2 are from the same identity
    dataset = "./test_images/"

    
    """Execute demo Tests"""

    """TEST WITH FACE DETECTION"""
    

    # Anjelina Jolie vs Anjelina Jolie
    # define image names, and target similarity
    image_path_1 = "Anjelina_Jolie_1.jpg"
    image_path_2 = "Anjelina_Jolie_2.jpg"
    target_similarity = 0.8830
    unit_test_fr(path2images = dataset,
                    image_1_path = image_path_1,
                    image_2_path = image_path_2,
                    threshold = default_threshold,
                    if_detect = True,
                    landmarks_1 = [],
                    landmarks_2 = [],
                    target_similarity = target_similarity)
    
    
    # Anjelina Jolie vs Scarlett
    # define image names, and target similarity
    image_path_1 = "Anjelina_Jolie_1.jpg"
    image_path_2 = "Scarlett_Johansson_2.jpg"
    target_similarity = 0.4932
    unit_test_fr(path2images = dataset,
                    image_1_path = image_path_1,
                    image_2_path = image_path_2,
                    threshold = default_threshold,
                    if_detect = True,
                    landmarks_1 = [],
                    landmarks_2 = [],
                    target_similarity = target_similarity)
   

    
    cv2.waitKey(1)


def demo_extract_single_embedding(image_path:str,
                                  save_embedding_path:str
                                  ):
    
    print("Extracting a single embedding", image_path)
    
    fr_mode_interpreter = tf.lite.Interpreter(default_model_path)
    fr_mode_interpreter.allocate_tensors()
       
    #Detector Instance
    detector = MTCNN()
    
    #Loading images wit to mat opencv
    image = _read_image_2_cvMat(image_path)         
    
    #Show detected images
    # cv2.imshow(image_1_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    embedding_1 = extract_embedding(image, 
                          fr_model_tflite = fr_mode_interpreter,
                          sub_mean = default_sub_mean,
                          cropp_percentage = default_cropp_percentage,
                          im_size = default_im_size,
                          norm_val = default_norm_val,
                          if_detect = True,
                          detector = detector,
                          landmarks = [])
    
    print("Saving Embedding")
    np.save(save_embedding_path, embedding_1)


def demo_extract_from_folder(path_to_input_images: str,
                             path_to_output_embeddings:str, 
                             im_exts = [".jpg",".bmp",".png"]
                             ):
    print("Extracting a embedding from a dataset")
    files = []
    for imext in im_exts:
        files = files+ [f for f in glob.glob(path_to_input_images+"/" + "*"  + imext)]
    
    for file in files:
        new_file = file.replace(os.sep, '/')
        new_file = path_to_output_embeddings+"/" + os.path.basename(new_file)
        for imext in im_exts:
            new_file = new_file.replace(imext, '.npy')
        demo_extract_single_embedding(file, new_file)
    
    
    pass

def demo_match_embeddings_demo(path_to_first_emb:str,
                               path_to_second_emb:str, 
                               ):
    
    print("Matching embeddings ", path_to_first_emb, "and", path_to_second_emb)
    
    embedding_1 = np.load(path_to_first_emb)
    
    embedding_2 = np.load(path_to_second_emb)
    
    verification_decision, similarity = match_embeddings(embedding_1,
                                                         embedding_2,
                                                         verification_threshold = default_threshold)
    print("verification_decision is ",verification_decision, " similarity is ", similarity)
    pass
    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--mode', 
                        default='demo', 
                        type=str,
                        choices=['demo', 's_im', 'im_dir','m_emb'],
                        help='Mode of programm. demo, single_image (s_im), image_directory(im_dir) or match_embeddings(m_emb)')

    parser.add_argument('-i','--image_path',
                        default="./test_images/Anjelina_Jolie_1.jpg",
                        type=str,
                        help='Path to the image for embedding extraction.')
    
    parser.add_argument('-ie','--image_embedding_path', 
                        default="./test_output/Anjelina_Jolie_1.npy",
                        type=str,
                        help='Path to the extracted embedding.')
    
    parser.add_argument('-d','--dataset_path', 
                        default='./test_images/dataset/', 
                        type=str,
                        help='Path to the dataset for embedding extraction.')
        
    parser.add_argument('-de','--dataset_embeddings_path', 
                        default="./test_output/dataset_embeddings/", 
                        type=str,
                        help='height of image (first layer y size)')
            
    parser.add_argument('-e1','--embedding_path_1', 
                        default="./test_embeddings/Anjelina_Jolie_1.npy", 
                        type=str,
                        help='First Embedding for matching')
        
    parser.add_argument('-e2','--embedding_path_2', 
                        default="./test_embeddings/Scarlett_Johansson_1.npy", 
                        type=str,
                        help='Second Embedding for matching')


    args = parser.parse_args()

    return args


    
#Testing Execution  
if __name__ == '__main__':
    
    args = parse_args()
    
    
    if (args.mode=="demo"):
        demo_full()
    elif(args.mode=="s_im"):
        demo_extract_single_embedding(image_path = args.image_path,
                                      save_embedding_path = args.image_embedding_path)
    elif(args.mode=="im_dir"):
        demo_extract_from_folder(path_to_input_images = args.dataset_path,
                                 path_to_output_embeddings = args.dataset_embeddings_path)
    elif(args.mode=="m_emb"):
        demo_match_embeddings_demo(path_to_first_emb = args.embedding_path_1,
                                   path_to_second_emb = args.embedding_path_2)
    else:
        print("Unknown Mode")
        pass
    
   