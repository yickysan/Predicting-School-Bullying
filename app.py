from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import MakePredictDF, PredictPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-data", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")
    
    else:
        custom_data = MakePredictDF(
            bullied_not_school = request.form.get("bullied_not_school"),
            cyber_bullied = request.form.get("cyber_bullied"),
            custom_age = request.form.get("custom_age"),
            sex = request.form.get("sex"),
            close_friends = request.form.get("close_friends"),
            missed_school = request.form.get("missed_school"),
            physically_attacked = request.form.get("physically_attacked"),
            physical_fighting = request.form.get("physical_fighting"),
            felt_lonely = request.form.get("felt_lonely"),
            other_students_kind_and_helpful = request.form.get("other_students_kind_and_helpful"),
            parents_understand_problems = request.form.get("parents_understand_problems")
        )

        df = custom_data.make_df()
        tables = [df.to_html(classes="data", header=True)]
    

        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(df)[0]
        if pred == 1:
            result = "This student is probably being bullied".title()
        else:
            result = "This student is probably not being bullied".title()


        return render_template("predictions.html", tables=tables, result = result)



if __name__ == "__main__":
    app.run(host="0.0.0.0")
