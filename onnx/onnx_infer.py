import cv2 
import numpy as np
import onnxruntime as ort 

class BrainUnetInference: 
    def __init__(self, onnx_path = "onnx/brain_unet.onnx"):
        self.onnx_path = onnx_path 
        self.ort_session = self.ort_session = ort.InferenceSession(self.onnx_path) 

    def sigmoid(self, x): 
        return 1 / (1 + np.exp(-x))
    
    def preprocessing(self, image_path : str): 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256,256)) / 255.0

        image = np.transpose(image, (2,1,0)) 
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image 
    
    def predict(self, image_path : str):
        image = self.preprocessing(image_path) 
        logits = self.ort_session.run(None, {'input': image})[0][0] 
        pred = self.sigmoid(logits)
        pred = np.where(pred < 0.5, 0, 1)
        return pred[0]
    

if __name__ == "__main__": 
    unet = BrainUnetInference()

    img = "onnx/TCGA_CS_4941_19960909_12.tif"
    pred = unet.predict(img) 
    print(pred.shape)