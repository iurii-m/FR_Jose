# Demo Face Recognition functionality.
Include 2 simple functions (extract_embedding, match_embeddings) for extracting 
face embeddings (deep representations) and matching of these embeddings.

Model - MobileNet_V2
## Performance of the model on LFW Dataset:
| FMR       | 0.1000 | 0.0100 | 0.001 | 0.0001 | 1E-05 | 1E-06 |
|-----------|--------|--------|-------|--------|-------|-------|
| FNMR      | 0.0013 | 0.0060 | 0.015 | 0.0540 | 0.054 | 0.054 |
| threshold | 0.3654 | 0.4891 | 0.571 | 0.6572 | 0.657 | 0.657 |

## Performance on a custom ICAO-Compliant  benchmark
EER -  0.00189 at threshold ~ 0.75

| FMR       | 0.1  | 0.01    | 0.001  | 0.0001  | 1e-05  | 1e-06  |
|-----------|------|---------|--------|---------|--------|--------|
| FNMR      | 0.0  | 7.4e-05 | 0.0053 | 0.03957 | 0.0965 | 0.2184 |
| Threshold | 0.42 | 0.6694  | 0.7967 | 0.85627 | 0.8874 | 0.9161 |


For folder processing the default image extentions:  [".jpg",".bmp",".png"]


# Example of usage:

##### Just Run Demo
    python FR_main.py


##### Extract single image embedding
    python FR_main.py -m s_im
    python FR_main.py -m s_im -i {image_path} -ie {path_embedding}
  

##### Extract embeddings from a directory with images
    python FR_main.py -m s_im
    python FR_main.py -m s_im -d {dataset_path} -de {embedding_of_dataset}


##### match embeddings  
    python FR_main.py -m m_emb  
    python FR_main.py -m m_emb -e1 {embedding_1} -e2 {embedding_2}