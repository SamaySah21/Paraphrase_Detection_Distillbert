#Import Modules and Libraries
import config
import torch
import flask
from flask import Flask, request
from basemodel import DistilBertModel
import torch.nn as nn

# Intializing App
app = Flask(__name__)

MODEL = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = config.TOKENIZER
max_len = config.MAX_LEN

@app.route("/", methods= ["GET"])
def home():
    return "<marquee  direction = 'right'><h1 style = 'color:red'>Hi, Welcome to Paraphrase Detection App !!</h1></marquee><h3 style = 'color:blue; text-align: center'> Web page is under construction, Send Request via postman to test it !! <h3>"

@app.route("/predict", methods= ["POST"])
def prediction():
    if request.method == "POST":
        original_sentence = request.form["orig"]
        paraphrase_sentence = request.form["para"]
    
    inputs = tokenizer.encode_plus(
        original_sentence,
        paraphrase_sentence,
        add_special_tokens=True,
    )

    ids = torch.tensor([inputs["input_ids"]], dtype=torch.long).to(device, dtype=torch.long)
    paraphrase_label = 0
    with torch.no_grad():
        MODEL.eval()
        output = MODEL(ids=ids)
        prob = nn.Sigmoid()(output).item()
        if prob > config.THRESHOLD_PROB:
            paraphrase_label = 1
            
    return {"Is_paraphrase" : paraphrase_label, "probability" : prob}



if __name__ == "__main__":
    MODEL = DistilBertModel(config.DISTILED_BERT_VERSION)
    MODEL.load_state_dict(torch.load("paraphrase.h5", map_location = torch.device('cpu')))
    MODEL.to(device)
    MODEL.eval()
    #app.run(host="0.0.0.0", port="9999")
    app.run(port="9010")