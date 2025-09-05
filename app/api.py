import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI,File,UploadFile
import uvicorn
from src.predict import predict_class,load_model_and_preprocess_image,class_infos
from pathlib import Path
from pydantic import BaseModel
import tempfile


app = FastAPI()




class NutritionInfo(BaseModel):
    class_name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    reference:str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_extension = Path(file.filename).suffix
   
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:

        contents = await file.read()
        tmp_file.write(contents)
        image_path = tmp_file.name
    

    image,model,device=load_model_and_preprocess_image(image_path)
    predicted_class = predict_class(model, image, device)
    class_info = class_infos(predicted_class)
    
    return NutritionInfo(
        class_name=predicted_class,
        calories=class_info['Calories'],
        protein=class_info['Protein'],
        carbs=class_info['carbohydrates'],
        fat=class_info['fat'],
        reference="100gm"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)