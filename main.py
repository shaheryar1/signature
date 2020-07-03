from fastapi import FastAPI,File, UploadFile
import numpy as np
import io
import base64
from PIL import Image
from  signature_extractor import extract,get_boxes
import base64


def encode(img):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return  new_image_string



app = FastAPI()

import cv2
@app.post("/extract_sign")
async def root(file: bytes = File(...)):
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")
        img = np.array(image)
        cv2.imwrite('a.png',img)
        img = cv2.imread('a.png',0)

        results={}
        signature=extract(img)
        boxes=get_boxes(signature)
        results["boxes"] = boxes
        results['extrcted_image'] =encode(signature)

        return results
    except Exception as e :
        return {"Error",str(e)}
