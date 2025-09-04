from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        features = [float(request.form[f]) for f in ['sl', 'sw', 'pl', 'pw']]
        prediction = model.predict([features])[0]
        prob = max(model.predict_proba([features])[0]) * 100
        species_emoji = {'Iris-setosa': '🌸', 'Iris-versicolor': '🌺', 'Iris-virginica': '🌷'}.get(prediction, '🌻')
        return f'''
        <div style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; text-align: center; color: white; box-shadow: 0 20px 40px rgba(0,0,0,0.1);">
            <h1 style="font-size: 3em; margin: 0 0 20px 0;">{species_emoji}</h1>
            <h2 style="margin: 0 0 10px 0; font-size: 2em;">Predicted Species</h2>
            <h3 style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 20px 0; font-size: 1.5em;">{prediction}</h3>
            <p style="font-size: 1.2em; margin: 20px 0;">Confidence: {prob:.1f}%</p>
            <a href="/" style="display: inline-block; background: white; color: #667eea; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: bold; margin-top: 20px; transition: transform 0.2s;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">🔮 Try Another Prediction</a>
        </div>'''
    return '''
    <div style="font-family: Arial; max-width: 600px; margin: 50px auto; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; text-align: center; color: white; box-shadow: 0 20px 40px rgba(0,0,0,0.1);">
        <h1 style="font-size: 3em; margin: 0 0 20px 0;">🌸 Iris Species Predictor</h1>
        <p style="font-size: 1.2em; margin-bottom: 30px; opacity: 0.9;">Enter flower measurements to predict the iris species</p>
        <form method="post" style="background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; backdrop-filter: blur(10px);">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <input name="sl" placeholder="🌿 Sepal Length (cm)" required style="padding: 15px; border: none; border-radius: 10px; font-size: 1em; text-align: center;" type="number" step="0.1" min="0" max="10" value="5.1">
                <input name="sw" placeholder="🌿 Sepal Width (cm)" required style="padding: 15px; border: none; border-radius: 10px; font-size: 1em; text-align: center;" type="number" step="0.1" min="0" max="10" value="3.5">
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px;">
                <input name="pl" placeholder="🌺 Petal Length (cm)" required style="padding: 15px; border: none; border-radius: 10px; font-size: 1em; text-align: center;" type="number" step="0.1" min="0" max="10" value="1.4">
                <input name="pw" placeholder="🌺 Petal Width (cm)" required style="padding: 15px; border: none; border-radius: 10px; font-size: 1em; text-align: center;" type="number" step="0.1" min="0" max="10" value="0.2">
            </div>
            <button style="background: white; color: #667eea; border: none; padding: 15px 40px; border-radius: 25px; font-size: 1.2em; font-weight: bold; cursor: pointer; transition: all 0.3s; box-shadow: 0 5px 15px rgba(0,0,0,0.2);" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.3)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 5px 15px rgba(0,0,0,0.2)'">🔮 Predict Species</button>
        </form>
        <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <h4 style="margin: 0 0 15px 0;">Species Types:</h4>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <span style="margin: 5px;">🌸 Setosa</span>
                <span style="margin: 5px;">🌺 Versicolor</span>
                <span style="margin: 5px;">🌷 Virginica</span>
            </div>
        </div>
    </div>'''

if __name__ == '__main__': app.run(port=8080, debug=True)
