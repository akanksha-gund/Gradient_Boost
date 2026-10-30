import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Gradient Boosting model
with open("gradient.pkl", "rb") as f:
  model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.92);
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            --accent-glow: radial-gradient(circle, rgba(168, 85, 247, 0.4) 0%, rgba(0, 0, 0, 0) 70%);
            --text-main: #0f172a;
            --text-muted: #64748b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px 15px;
            color: var(--text-main);
            position: relative;
            overflow-x: hidden;
        }

        /* Animated Ambient Background Objects */
        body::before {
            content: '';
            position: absolute;
            width: 400px;
            height: 400px;
            background: var(--accent-glow);
            top: -50px;
            left: -50px;
            border-radius: 50%;
            z-index: 0;
            animation: pulse 8s infinite alternate;
        }

        body::after {
            content: '';
            position: absolute;
            width: 500px;
            height: 500px;
            background: var(--accent-glow);
            bottom: -100px;
            right: -100px;
            border-radius: 50%;
            z-index: 0;
            animation: pulse 10s infinite alternate-reverse;
        }

        @keyframes pulse {
            0% { transform: scale(1) translate(0, 0); opacity: 0.6; }
            100% { transform: scale(1.3) translate(40px, 40px); opacity: 0.9; }
        }

        .container {
            position: relative;
            z-index: 10;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.6);
            padding: 45px 35px;
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35), 0 0 20px rgba(168, 85, 247, 0.2);
            width: 100%;
            max-width: 560px;
            animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header-icon {
            width: 70px;
            height: 70px;
            background: var(--primary-gradient);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 15px auto;
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.4);
            animation: float 4s ease-in-out infinite;
        }

        .header-icon svg {
            width: 36px;
            height: 36px;
            fill: #ffffff;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-8px); }
        }

        h2 {
            text-align: center;
            font-size: 26px;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 28px;
            font-weight: 500;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .form-group {
            margin-bottom: 18px;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            display: block;
            margin-bottom: 7px;
            font-weight: 700;
            font-size: 13px;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input, select {
            width: 100%;
            padding: 13px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 500;
            color: #0f172a;
            background: #ffffff;
            transition: all 0.3s ease;
            outline: none;
        }

        input:focus, select:focus {
            border-color: #8b5cf6;
            box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.15);
            transform: translateY(-2px);
        }

        button {
            grid-column: span 2;
            width: 100%;
            padding: 16px;
            background: var(--primary-gradient);
            color: white;
            border: none;
            border-radius: 14px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
            box-shadow: 0 10px 25px rgba(168, 85, 247, 0.35);
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(168, 85, 247, 0.5);
        }

        button:active {
            transform: translateY(0);
        }

        /* Result Section */
        .result-card {
            margin-top: 28px;
            padding: 22px;
            background: #ffffff;
            border-radius: 16px;
            border: 2px solid #e2e8f0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            text-align: center;
            animation: fadeIn 0.5s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .result-value {
            font-size: 38px;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        .progress-container {
            width: 100%;
            background-color: #f1f5f9;
            border-radius: 10px;
            height: 12px;
            overflow: hidden;
        }

        .progress-bar {
            height: 100%;
            background: var(--primary-gradient);
            border-radius: 10px;
            transition: width 1s ease-in-out;
        }

        .error-card {
            margin-top: 28px;
            padding: 18px;
            background: #fef2f2;
            border: 1px solid #fca5a5;
            border-radius: 14px;
            color: #991b1b;
            font-weight: 600;
            text-align: center;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-icon">
            <svg viewBox="0 0 24 24">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
        </div>
        <h2>AI Predictive Analytics</h2>
        <p class="subtitle">Gradient Boosting Prediction Model</p>

        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="department">Department</label>
                    <select id="department" name="department" required>
                        <option value="Engineering">Engineering</option>
                        <option value="Sales">Sales</option>
                        <option value="HR">HR</option>
                        <option value="Marketing">Marketing</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="experience">Experience Years</label>
                    <input type="number" step="any" id="experience" name="experience" required placeholder="e.g. 4.5">
                </div>

                <div class="form-group">
                    <label for="education">Education</label>
                    <select id="education" name="education" required>
                        <option value="Bachelor">Bachelor</option>
                        <option value="Master">Master</option>
                        <option value="PhD">PhD</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" required placeholder="e.g. 28">
                </div>

                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select id="gender" name="gender">
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="city">City</label>
                    <select id="city" name="city" required>
                        <option value="Pune">Pune</option>
                        <option value="Mumbai">Mumbai</option>
                        <option value="Delhi">Delhi</option>
                    </select>
                </div>

                <button type="submit">
                    <span>Generate Analytics Prediction</span>
                    <svg style="width:20px;height:20px;fill:white;" viewBox="0 0 24 24">
                        <path d="M5 12h14M12 5l7 7-7 7"/>
                    </svg>
                </button>
            </div>
        </form>

        {% if prediction is not none %}
            {% if is_error %}
                <div class="error-card">
                    ⚠️ {{ prediction }}
                </div>
            {% else %}
                <div class="result-card">
                    <div class="result-title">Prediction Result</div>
                    <div class="result-value">{{ prediction }}</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {{ raw_value }}%;"></div>
                    </div>
                </div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
  prediction = None
  raw_value = 0
  is_error = False

  if request.method == "POST":
    try:
      dept_mapping = {
          "Engineering": 0,
          "Sales": 1,
          "HR": 2,
          "Marketing": 3,
      }
      edu_mapping = {"Bachelor": 0, "Master": 1, "PhD": 2}
      gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
      city_mapping = {"Pune": 0, "Mumbai": 1, "Delhi": 2}

      department = dept_mapping.get(request.form["department"], 0)
      experience = float(request.form["experience"])
      education = edu_mapping.get(request.form["education"], 0)
      age = float(request.form["age"])
      gender = gender_mapping.get(request.form["gender"], 0)
      city = city_mapping.get(request.form["city"], 0)

      features = np.array([[department, experience, education, age, gender, city]])

      pred = model.predict(features)

      raw_val = float(pred[0])
      # Percentage format madhe value set karne
      if raw_val <= 1.0:
        raw_value = min(max(raw_val * 100, 0), 100)
      else:
        raw_value = min(max(raw_val, 0), 100)

      prediction = f"{raw_value:.2f}%"

    except Exception as e:
      prediction = f"Error during prediction: {str(e)}"
      is_error = True

  return render_template_string(
      HTML_TEMPLATE,
      prediction=prediction,
      raw_value=raw_value,
      is_error=is_error,
  )


if __name__ == "__main__":
  app.run(debug=True)