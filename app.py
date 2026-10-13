import os
import io
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

app = FastAPI(title="Rehman V6 API")

upsampler = None

def get_upsampler():
    global upsampler
    if upsampler is None:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=None
        )
    return upsampler

@app.get("/")
def home():
    return {"status": "ok", "message": "Rehman V6 API is Live!"}

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        upsampler_instance = get_upsampler()
        output, _ = upsampler_instance.enhance(img, outscale=4)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(output_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
