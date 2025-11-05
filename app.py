import sys
import os
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from uvicorn import run as app_run

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipelines.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.main_utils.model.estimator import NetworkModel
from networksecurity.constants.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME
)

# ---------------------- Setup ----------------------
ca = certifi.where()
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL")

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
templates = Jinja2Templates(directory="./templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------------------------------------

@app.get("/", tags=["Root"])
async def index():
    return RedirectResponse(url="/upload")


# ---------------------- Train Model ----------------------
@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return RedirectResponse(url="/results", status_code=303)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
# ----------------------------------------------------------


# ---------------------- Upload CSV ------------------------
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_route(file: UploadFile = File(...), request: Request = None):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        y_pred = network_model.predict(df)
        df["Prediction"] = ["Phishing" if val == 1 else "Safe" for val in y_pred]
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        return templates.TemplateResponse(
            "table.html",
            {"request": request, "table": f"<p style='color:red;'>⚠️ Error: {e}</p>"}
        )
# ----------------------------------------------------------


# ---------------------- Manual Feature Input -------------
@app.get("/feature_form", response_class=HTMLResponse)
async def get_feature_form(request: Request):
    features = [
        "having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol",
        "double_slash_redirecting","Prefix_Suffix","having_Sub_Domain","SSLfinal_State",
        "Domain_registeration_length","Favicon","port","HTTPS_token","Request_URL",
        "URL_of_Anchor","Links_in_tags","SFH","Submitting_to_email","Abnormal_URL",
        "Redirect","on_mouseover","RightClick","popUpWidnow","Iframe","age_of_domain",
        "DNSRecord","web_traffic","Page_Rank","Google_Index","Links_pointing_to_page","Statistical_report"
    ]
    return templates.TemplateResponse("feature_form.html", {"request": request, "features": features})


from fastapi import Form, Request
import pandas as pd

@app.post("/feature_form", response_class=HTMLResponse)
async def predict_from_form(request: Request):
    try:
        # ✅ Dynamically read all form inputs
        form_data = await request.form()
        data = {k: int(v) for k, v in form_data.items() if v.strip() != ""}

        if not data:
            return templates.TemplateResponse(
                "feature_form.html",
                {
                    "request": request,
                    "verdict": "⚠️ Please fill in at least one feature value.",
                    "verdict_class": "error",
                }
            )

        df = pd.DataFrame([data])

        # ⚠️ Handle identical values (all 1 or all -1)
        if len(set(df.values.flatten())) == 1:
            return templates.TemplateResponse(
                "feature_form.html",
                {
                    "request": request,
                    "features": data.keys(),
                    "verdict": "⚠️ All feature values identical — model cannot classify reliably.",
                    "verdict_class": "error",
                    "selected": data
                }
            )

        # ✅ Load model and preprocessor
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=model)

        # ✅ Predict
        y_pred = network_model.predict(df)
        proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.5

        # ✅ Calibrated confidence
        true_conf = round(proba * 100, 2)
        display_conf = min(max(round(proba * 100 * 1.8, 2), 5.0), 95.0)

        if display_conf < 50:
            verdict_text = f"✅ Likely Safe (display {display_conf}%, true {true_conf}%)"
            verdict_class = "safe"
        elif display_conf < 75:
            verdict_text = f"⚠️ Suspicious (display {display_conf}%, true {true_conf}%)"
            verdict_class = "medium"
        else:
            verdict_text = f"🚨 Likely Phishing (display {display_conf}%, true {true_conf}%)"
            verdict_class = "malicious"

        return templates.TemplateResponse(
            "feature_form.html",
            {
                "request": request,
                "features": data.keys(),
                "verdict": verdict_text,
                "verdict_class": verdict_class,
                "selected": data
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "feature_form.html",
            {
                "request": request,
                "features": [],
                "verdict": f"❌ Error: {str(e)}",
                "verdict_class": "error"
            }
        )





# ---------------------- Results Page ----------------------
@app.get("/results", response_class=HTMLResponse)
async def show_results(request: Request):
    try:
        plots = {
            "Model Comparison": "/Artifacts/model_comparison.png",
            "Correlation Heatmap": "/Artifacts/correlation_heatmap.png",
            "Feature Importance": "/Artifacts/feature_importance.png",
            "Decision Tree": "/Artifacts/decision_tree.png",
        }

        mlflow_link = "http://127.0.0.1:5000"

        return templates.TemplateResponse(
            "results.html",
            {"request": request, "plots": plots, "mlflow_link": mlflow_link}
        )

    except Exception as e:
        return templates.TemplateResponse("results.html", {"request": request, "error": str(e)})



if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)
