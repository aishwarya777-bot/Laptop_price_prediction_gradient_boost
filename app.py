import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Bulletproof path resolution for Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'gradient_boost_model.pkl')

model = None
model_error = None

# Safely load the model to prevent crashing the entire function on startup
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model_error = f"Model load failed: {str(e)}"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laptop Price Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-50 min-h-screen flex items-center justify-center p-6 font-sans">

    <div class="max-w-2xl w-full bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
        <div class="bg-linear-to-r from-blue-600 to-indigo-700 px-8 py-6 text-white text-center">
            <h1 class="text-2xl font-bold tracking-tight">Laptop Value Estimator</h1>
            <p class="text-blue-100 mt-1 text-sm">Enter the specifications below to predict the estimated value.</p>
        </div>

        {% if model_error %}
        <div class="bg-red-50 border-l-4 border-red-500 p-4 m-8 rounded-r-lg">
            <div class="flex">
                <div class="text-red-700 text-sm font-medium">
                    <strong>Server Configuration Error:</strong> {{ model_error }}. Check your scikit-learn version match.
                </div>
            </div>
        </div>
        {% endif %}

        <form method="POST" class="p-8 space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Brand (Numerical ID)</label>
                    <input type="number" step="any" name="Brand" required placeholder="e.g., 1" class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Processor Speed (GHz)</label>
                    <input type="number" step="any" name="Processor_Speed" required placeholder="e.g., 2.5" class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">RAM Size (GB)</label>
                    <input type="number" step="any" name="RAM_Size" required placeholder="e.g., 16" class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Storage Capacity (GB)</label>
                    <input type="number" step="any" name="Storage_Capacity" required placeholder="e.g., 512" class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Screen Size (Inches)</label>
                    <input type="number" step="any" name="Screen_Size" required placeholder="e.g., 15.6" class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 outline-none">
                </div>
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Weight (kg)</label>
                    <input type="number" step="any" name="Weight" required placeholder="e.g., 1.5" class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 outline-none">
                </div>
            </div>

            <div class="pt-4">
                <button type="submit" {% if model_error %}disabled class="w-full bg-slate-300 text-slate-500 font-semibold py-3 px-6 rounded-xl cursor-not-allowed"{% else %}class="w-full bg-linear-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-xl shadow-md transition-all cursor-pointer"{% endif %}>
                    Calculate Prediction
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="border-t border-slate-100 bg-slate-50 p-8 text-center">
            <h2 class="text-sm font-medium text-slate-500 uppercase tracking-wider">Estimated Prediction</h2>
            <div class="mt-2 text-4xl font-extrabold text-slate-900">
                {% if 'Error' in prediction|string %}
                    <span class="text-xl text-red-600">{{ prediction }}</span>
                {% else %}
                    ${{ "%.2f"|format(prediction) }}
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if model is None:
            return render_template_string(HTML_TEMPLATE, prediction=f"Prediction Error: Model didn't load properly. {model_error}", model_error=model_error)
        
        try:
            features = [
                float(request.form['Brand']),
                float(request.form['Processor_Speed']),
                float(request.form['RAM_Size']),
                float(request.form['Storage_Capacity']),
                float(request.form['Screen_Size']),
                float(request.form['Weight'])
            ]
            input_data = np.array([features])
            pred_array = model.predict(input_data)
            prediction = float(pred_array[0])
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, model_error=model_error)

if __name__ == '__main__':
    app.run(debug=True)
