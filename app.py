import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'gradient_boost_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML Template with Tailwind CSS for an attractive UI
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
            <p class="text-blue-100 mt-1 text-sm">Enter the specifications below to predict the estimated value using Gradient Boosting.</p>
        </div>

        <form method="POST" class="p-8 space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Brand (Numerical ID/Code)</label>
                    <input type="number" step="any" name="Brand" required placeholder="e.g., 1, 2, 3"
                           class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Processor Speed (GHz)</label>
                    <input type="number" step="any" name="Processor_Speed" required placeholder="e.g., 2.5"
                           class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">RAM Size (GB)</label>
                    <input type="number" step="any" name="RAM_Size" required placeholder="e.g., 8, 16, 32"
                           class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Storage Capacity (GB)</label>
                    <input type="number" step="any" name="Storage_Capacity" required placeholder="e.g., 512, 1024"
                           class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Screen Size (Inches)</label>
                    <input type="number" step="any" name="Screen_Size" required placeholder="e.g., 14.0, 15.6"
                           class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                </div>

                <div>
                    <label class="block text-sm font-semibold text-slate-700 mb-2">Weight (kg)</label>
                    <input type="number" step="any" name="Weight" required placeholder="e.g., 1.4, 2.1"
                           class="w-full px-4 py-2.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all">
                </div>
            </div>

            <div class="pt-4">
                <button type="submit" 
                        class="w-full bg-linear-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-xl shadow-md hover:shadow-lg transform active:scale-98 transition-all cursor-pointer">
                    Calculate Prediction
                </button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="border-t border-slate-100 bg-slate-50 p-8 text-center animate-fade-in">
            <h2 class="text-sm font-medium text-slate-500 uppercase tracking-wider">Estimated Prediction</h2>
            <div class="mt-2 text-4xl font-extrabold text-slate-900">
                ${{ "%.2f"|format(prediction) }}
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
        try:
            # Extract input features in the exact order required by the model
            features = [
                float(request.form['Brand']),
                float(request.form['Processor_Speed']),
                float(request.form['RAM_Size']),
                float(request.form['Storage_Capacity']),
                float(request.form['Screen_Size']),
                float(request.form['Weight'])
            ]
            
            # Convert to numpy array shape (1, 6)
            input_data = np.array([features])
            
            # Perform prediction
            pred_array = model.predict(input_data)
            prediction = float(pred_array[0])
        except Exception as e:
            prediction = f"Error: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

# Required wrapper for Vercel deployment
if __name__ == '__main__':
    app.run(debug=True)
