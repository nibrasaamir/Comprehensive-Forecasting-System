from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

# Home page with options to select which app to use
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        selected_app = request.form['app_choice']
        if selected_app == 'app1':
            return redirect("http://localhost:5001/api/daily")  # Port for the first application
        elif selected_app == 'app2':
            return redirect("http://localhost:5002/api/daily")  # Port for the second application
    return render_template('home.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
