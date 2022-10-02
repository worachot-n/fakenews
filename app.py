import os
from flask import Flask, render_template
from flask import request


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "worachot-n/WangchanBERTa_LimeSoda_FakeNews"

tokenizer = AutoTokenizer.from_pretrained(name,)

model = AutoModelForSequenceClassification.from_pretrained(name)

MAX_LENGTH = 416

class_names = {0: "True", 1: "Fake"}


def predict(text):

    batch = tokenizer(text, padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")

    with torch.no_grad():
        output_test = model(**batch)
        # print(output_test)
        pred_test = F.softmax(output_test.logits, dim=1)
        # print(pred_test)
        percent = pred_test.numpy()
        # print(percent[:,1])
        labels = torch.argmax(pred_test, dim=1)
        # print(labels)
        labels = [class_names[label] for label in labels.numpy()]
        # print(labels)

    return labels[0], percent[0][1]


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        form = request.form
        text = form['post']
        result, percent = predict(text)

        return render_template("index.html", result=result, percent=percent)

    return render_template("index.html")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
