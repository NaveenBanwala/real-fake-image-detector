import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

# --- Configuration ---
TARGET_SIZE = (224, 224) 
MODEL_NAME = "efficientnet_robust_latest (1).keras"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

ml_models = {}

def build_model_reconstruction():
    """Reconstructs the EXACT Colab architecture to bypass Keras 3 loading bugs."""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights=None, 
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        pooling='avg' 
    )
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("🔄 Reconstructing model architecture...")
        model = build_model_reconstruction()
        
        print(f"📥 Loading weights from {MODEL_NAME}...")
        model.load_weights(MODEL_PATH)
        
        ml_models["efficientnet"] = model
        print("✅ Model Reconstruction & Weight Loading Successful!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

def process_like_colab(file_bytes: bytes, file_ext: str):
    """Saves file temporarily to use Keras load_img, matching Colab perfectly."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
    try:
        temp_file.write(file_bytes)
        temp_file.close()

        # EXACT COLAB MATH
        img = keras_image.load_img(temp_file.name, target_size=TARGET_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    finally:
        os.unlink(temp_file.name)

@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Detector</title>
        <style>
            body { font-family: sans-serif; background: #f4f7f6; display: flex; justify-content: center; padding: 50px; }
            .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); width: 400px; text-align: center; }
            h2 { color: #2c3e50; }
            .upload-section { margin: 20px 0; border: 2px dashed #3498db; padding: 20px; border-radius: 8px; }
            button { background: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; width: 100%; font-weight: bold; }
            #result { margin-top: 20px; padding: 15px; border-radius: 6px; display: none; font-weight: bold; }
            .real { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .fake { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🛡️ Deepfake Detector</h2>
            <p>Upload an image to verify authenticity.</p>
            <div class="upload-section">
                <input type="file" id="fileInput" accept="image/*">
            </div>
            <button onclick="analyze()">Analyze Image</button>
            <div id="result"></div>
        </div>

        <script>
            async function analyze() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                if (!fileInput.files[0]) return alert("Select a file first!");

                resultDiv.style.display = 'block';
                resultDiv.innerHTML = "🧠 Analyzing layers...";
                resultDiv.className = "";

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    if (data.status === "success") {
                        resultDiv.innerHTML = `VERDICT: ${data.label}<br>Confidence: ${data.confidence}`;
                        resultDiv.className = data.label === "REAL" ? "real" : "fake";
                    } else {
                        resultDiv.innerHTML = "Error: " + data.detail;
                    }
                } catch (err) {
                    resultDiv.innerHTML = "Connection Error";
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if "efficientnet" not in ml_models:
        raise HTTPException(status_code=503, detail="Model failed to load.")

    try:
        contents = await file.read()
        _, ext = os.path.splitext(file.filename)
        if not ext:
            ext = ".jpg"
            
        processed_image = process_like_colab(contents, ext)
        
        model = ml_models["efficientnet"]
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        # Match Colab exactly
        verdict = "REAL" if prediction > 0.5 else "FAKE"
        confidence = prediction if prediction > 0.5 else (1 - prediction)

        return {
            "status": "success",
            "label": verdict,
            "confidence": f"{confidence:.2%}",
            "raw_score": float(prediction)
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# import io
# import os
# import base64
# import numpy as np
# import tensorflow as tf
# import matplotlib.cm as cm
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse
# from PIL import Image
# from contextlib import asynccontextmanager

# # --- Configuration ---
# # 1. Matching Colab exactly
# TARGET_SIZE = (224, 224) 
# MODEL_FILENAME = "efficientnet_robust_latest (1).keras" 
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# ml_models = {}

# def build_model_shell():
#     # 2. Rebuilding the shell to safely load weights and bypass the Keras bug
#     main_input = tf.keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    
#     base_model = tf.keras.applications.EfficientNetB0(
#         include_top=False, 
#         weights=None, 
#         input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
#         pooling='avg',
#         name="efficientnetb0"
#     )
    
#     x = base_model(main_input)
#     x = tf.keras.layers.Dense(256, activation='relu', name="fc_dense")(x)
#     output = tf.keras.layers.Dense(1, activation='sigmoid', name="fc_output")(x)
    
#     return tf.keras.Model(inputs=main_input, outputs=output)

# def get_gradcam_bulletproof(img_array, model):
#     # 3. Explicit forward pass - identical logic to your Colab fix
#     base_model = model.get_layer("efficientnetb0")
#     dense_layer = model.get_layer("fc_dense")
#     output_layer = model.get_layer("fc_output")

#     # Safe inner model
#     inner_model = tf.keras.Model(
#         inputs=base_model.input,
#         outputs=[base_model.get_layer('top_activation').output, base_model.output]
#     )

#     img_tensor = tf.cast(img_array, tf.float32)

#     with tf.GradientTape() as tape:
#         # Pass through the backbone
#         conv_outputs, x = inner_model(img_tensor, training=False)
        
#         # Pass through YOUR top layers manually
#         x = dense_layer(x, training=False)
#         predictions = output_layer(x, training=False)
#         loss = predictions[:, 0]

#     # Calculate exactly what pixels caused the verdict
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
#     return heatmap.numpy()

# def apply_heatmap(img_bytes, heatmap, alpha=0.4):
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     width, height = img.size
    
#     heatmap_255 = np.uint8(255 * heatmap)
    
#     try:
#         jet = cm.get_cmap("jet")
#     except AttributeError:
#         import matplotlib
#         jet = matplotlib.colormaps.get_cmap("jet")
        
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap_255]
    
#     jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
#     jet_heatmap = jet_heatmap.resize((width, height), resample=Image.BILINEAR)
    
#     img = img.convert("RGBA")
#     jet_heatmap = jet_heatmap.convert("RGBA")
#     blended = Image.blend(img, jet_heatmap, alpha=alpha)
#     blended = blended.convert("RGB")
    
#     buffered = io.BytesIO()
#     blended.save(buffered, format="JPEG")
#     return base64.b64encode(buffered.getvalue()).decode()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         # Load exactly how it worked before, but with the 224x224 shell
#         model = build_model_shell()
#         model.load_weights(MODEL_PATH)
#         ml_models["model"] = model
#         print("✅ Bulletproof Model & Grad-CAM Ready.")
#     except Exception as e:
#         print(f"❌ Initialization Error: {e}")
#     yield
#     ml_models.clear()

# app = FastAPI(lifespan=lifespan)

# @app.get("/", response_class=HTMLResponse)
# async def ui():
#     return """
#     <html>
#         <head>
#             <title>AI Forensic Lab</title>
#             <style>
#                 body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: white; display: flex; justify-content: center; padding: 40px; }
#                 .container { max-width: 1000px; width: 100%; background: #1e293b; padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); text-align: center; }
#                 .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px; display: none; }
#                 img { width: 100%; border-radius: 12px; border: 3px solid #334155; }
#                 .badge { display: inline-block; padding: 12px 24px; border-radius: 50px; font-weight: bold; font-size: 1.2rem; margin: 20px 0; }
#                 .REAL { background: #059669; border: 2px solid #10b981; }
#                 .FAKE { background: #dc2626; border: 2px solid #ef4444; }
#                 #loading { color: #fbbf24; font-weight: bold; font-size: 1.1rem; margin: 20px; display: none; }
#                 button { background: #6366f1; color: white; border: none; padding: 12px 30px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.3s; }
#             </style>
#         </head>
#         <body>
#             <div class="container">
#                 <h1>🔍 AI Forensic Analysis</h1>
#                 <form id="uploadForm">
#                     <input name="file" type="file" id="fileInput" accept="image/*" required><br><br>
#                     <button type="submit">Run Forensic Scan</button>
#                 </form>
#                 <div id="loading">📡 Scanning Pixels...</div>
#                 <div id="status"></div>
#                 <div class="result-grid" id="resultGrid">
#                     <div><p>Input Image</p><img id="origImg"></div>
#                     <div><p>Deepfake Artifacts</p><img id="heatImg"></div>
#                 </div>
#             </div>
#             <script>
#                 const form = document.getElementById('uploadForm');
#                 form.onsubmit = async (e) => {
#                     e.preventDefault();
#                     document.getElementById('loading').style.display = 'block';
#                     document.getElementById('resultGrid').style.display = 'none';
#                     const formData = new FormData(form);
#                     const response = await fetch('/predict', { method: 'POST', body: formData });
#                     const data = await response.json();
#                     document.getElementById('loading').style.display = 'none';
#                     if (data.error) { alert("Error: " + data.error); return; }
#                     document.getElementById('resultGrid').style.display = 'grid';
#                     document.getElementById('origImg').src = data.original;
#                     document.getElementById('heatImg').src = 'data:image/jpeg;base64,' + data.heatmap;
#                     document.getElementById('status').innerHTML = `<div class="badge ${data.label}">${data.label} (${data.confidence})</div>`;
#                 };
#             </script>
#         </body>
#     </html>
#     """

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     model = ml_models.get("model")
#     if not model: return {"error": "Model not loaded"}

#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         # 4. Colab exact preprocessing (NO / 255.0!)
#         img_resized = image.resize(TARGET_SIZE)
#         img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
#         img_array = np.expand_dims(img_array, axis=0)

#         # Prediction
#         preds = model.predict(img_array, verbose=0)
#         score = float(preds[0][0])
        
#         label = "REAL" if score > 0.5 else "FAKE"
#         confidence = score if label == "REAL" else (1.0 - score)

#         # Grad-CAM
#         heatmap = get_gradcam_bulletproof(img_array, model)
#         heatmap_base64 = apply_heatmap(contents, heatmap)

#         return {
#             "label": label,
#             "confidence": f"{round(confidence * 100, 1)}%",
#             "heatmap": heatmap_base64,
#             "original": f"data:image/jpeg;base64,{base64.b64encode(contents).decode()}"
#         }
#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


# import io
# import os
# import base64
# import numpy as np
# import tensorflow as tf
# import matplotlib.cm as cm
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse
# from PIL import Image
# from contextlib import asynccontextmanager

# # --- Configuration (Matching your working version) ---
# TARGET_SIZE = (128, 128)
# MODEL_FILENAME = "efficientnet_robust_latest (1).keras" 
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# ml_models = {}

# def build_model_shell():
#     """Architecture that you confirmed gives correct predictions."""
#     base_model = tf.keras.applications.EfficientNetB0(
#         include_top=False, 
#         weights=None, 
#         input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
#         pooling='avg',
#         name="efficientnetb0"
#     )
    
#     model = tf.keras.Sequential([
#         base_model,
#         tf.keras.layers.Dense(256, activation='relu', name="fc_dense"), 
#         tf.keras.layers.Dense(1, activation='sigmoid', name="fc_output")
#     ])
#     return model

# def get_gradcam_heatmap(img_array, model):
#     """Simplified Grad-CAM for your Sequential model."""
#     # EfficientNetB0's last conv layer is usually 'top_activation'
#     base_model = model.get_layer("efficientnetb0")
    
#     # We create a model that outputs the last conv layer AND the final prediction
#     grad_model = tf.keras.Model(
#         inputs=[model.inputs],
#         outputs=[base_model.get_layer("top_activation").output, model.output]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         loss = predictions[:, 0]

#     # Gradient of the output class with respect to the conv layer
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)

#     # ReLU and Normalize
#     heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
#     return heatmap.numpy()

# def apply_heatmap(img_bytes, heatmap, alpha=0.4):
#     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#     width, height = img.size
#     heatmap_255 = np.uint8(255 * heatmap)
    
#     try:
#         jet = cm.get_cmap("jet")
#     except:
#         import matplotlib
#         jet = matplotlib.colormaps.get_cmap("jet")
        
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap_255]
#     jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255)).resize((width, height))
    
#     img = img.convert("RGBA")
#     jet_heatmap = jet_heatmap.convert("RGBA")
#     blended = Image.blend(img, jet_heatmap, alpha=alpha).convert("RGB")
    
#     buf = io.BytesIO()
#     blended.save(buf, format="JPEG")
#     return base64.b64encode(buf.getvalue()).decode()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     try:
#         model = build_model_shell()
#         model.load_weights(MODEL_PATH)
#         ml_models["model"] = model
#         print("✅ System Ready (Using Correct Accuracy Logic)")
#     except Exception as e:
#         print(f"❌ Error: {e}")
#     yield
#     ml_models.clear()

# app = FastAPI(lifespan=lifespan)

# # --- UI (The Nice Dark Version) ---
# @app.get("/", response_class=HTMLResponse)
# async def ui():
#     return """
#     <html>
#         <head>
#             <title>AI Forensic Lab</title>
#             <style>
#                 body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: white; display: flex; justify-content: center; padding: 40px; }
#                 .container { max-width: 1000px; width: 100%; background: #1e293b; padding: 30px; border-radius: 20px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
#                 .result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-top: 30px; display: none; }
#                 img { width: 100%; border-radius: 12px; border: 2px solid #334155; }
#                 .badge { display: inline-block; padding: 12px 24px; border-radius: 50px; font-weight: bold; font-size: 1.2rem; margin: 20px 0; }
#                 .Real { background: #059669; } .Fake { background: #dc2626; }
#                 button { background: #6366f1; color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; }
#             </style>
#         </head>
#         <body>
#             <div class="container">
#                 <h1>🔍 AI Forensic Analysis</h1>
#                 <form id="uploadForm">
#                     <input name="file" type="file" accept="image/*" required><br><br>
#                     <button type="submit">Run Forensic Scan</button>
#                 </form>
#                 <div id="status"></div>
#                 <div class="result-grid" id="resultGrid">
#                     <div><p>Input Image</p><img id="origImg"></div>
#                     <div><p>Deepfake Artifacts</p><img id="heatImg"></div>
#                 </div>
#             </div>
#             <script>
#                 const form = document.getElementById('uploadForm');
#                 form.onsubmit = async (e) => {
#                     e.preventDefault();
#                     const formData = new FormData(form);
#                     const response = await fetch('/predict', { method: 'POST', body: formData });
#                     const data = await response.json();
#                     document.getElementById('resultGrid').style.display = 'grid';
#                     document.getElementById('origImg').src = data.original;
#                     document.getElementById('heatImg').src = 'data:image/jpeg;base64,' + data.heatmap;
#                     document.getElementById('status').innerHTML = `<div class="badge ${data.label}">${data.label} (${data.confidence})</div>`;
#                 };
#             </script>
#         </body>
#     </html>
#     """

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     model = ml_models.get("model")
#     try:
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB").resize(TARGET_SIZE)
#         img_array = np.array(image).astype('float32') / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # 1. Accurate Prediction Logic (Matches your working code)
#         prediction = model.predict(img_array)
#         score = float(prediction[0][0])
#         label = "Real" if score > 0.5 else "Fake"
#         confidence = score if label == "Real" else (1 - score)

#         # 2. Grad-CAM
#         heatmap = get_gradcam_heatmap(img_array, model)
#         heatmap_base64 = apply_heatmap(contents, heatmap)

#         return {
#             "label": label,
#             "confidence": f"{round(confidence * 100, 2)}%",
#             "heatmap": heatmap_base64,
#             "original": f"data:image/jpeg;base64,{base64.b64encode(contents).decode()}"
#         }
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)