from flask import Flask, render_template
from essai.bin.modules.BertSingle import BertTopicClass
from essai.config import settings

app = Flask(__name__)

@app.route('/')
def index():
    BERT_NAME = settings.bert_name
    BERT_MODEL = BertTopicClass(BERT_NAME,restore=1)
    return render_template('index.html')

@app.route('/plot_space', methods=['POST'])
def plot_space():
    # Instantiate the MyClass object with a name
    BERT_NAME = settings.bert_name
    BERT_MODEL = BertTopicClass(BERT_NAME,restore=1)

    # Call the my_method and get the result
    BERT_MODEL.Genereate_WC()

    # Update the greeting in the template
    greeting = BERT_MODEL.get_greeting()

    return render_template('index.html', url=greeting, method_result=method_result)


if __name__ == '__main__':
    app.run(debug=True)