from fastapi import FastAPI,File, UploadFile
import numpy as np
import io
import base64
from PIL import Image
from  signature_extractor import extract
import base64


def encode(img):
    pil_img = Image.fromarray(img)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return  new_image_string



app = FastAPI()


@app.post("/extract_sign")
async def root(file: bytes = File(...)):
    try:
        image = Image.open(io.BytesIO(file)).convert("RGB")[0]
        img = np.array(image)

        results={}

        results['extrcted_image'] = extract(img)
        return results
    except Exception as e :
        return {"Error",e}
