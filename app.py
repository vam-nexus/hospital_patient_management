from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    session,
    flash,
)
from datetime import datetime
import json
import os
import hashlib

app = Flask(__name__)
app.secret_key = "hospital_management_secret_key_2023"  # Change this in production

# Simple in-memory storage for patients and doctors (in a real app, you'd use a database)
patients = []
doctors = []


def load_patients():
    """Load patients from JSON file if it exists"""
    global patients
    if os.path.exists("patients.json"):
        try:
            with open("patients.json", "r") as f:
                patients = json.load(f)
        except:
            patients = []


def save_patients():
    """Save patients to JSON file"""
    with open("patients.json", "w") as f:
        json.dump(patients, f, indent=2)


def load_doctors():
    """Load doctors from JSON file if it exists"""
    global doctors
    if os.path.exists("doctors.json"):
        try:
            with open("doctors.json", "r") as f:
                doctors = json.load(f)
        except:
            doctors = []


def save_doctors():
    """Save doctors to JSON file"""
    with open("doctors.json", "w") as f:
        json.dump(doctors, f, indent=2)


def hash_password(password):
    """Simple password hashing"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed


def login_required(f):
    """Decorator to require login for routes"""
    from functools import wraps

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "doctor_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


@app.route("/")
@login_required
def index():
    """Main page showing all patients for logged-in doctor"""
    load_patients()
    doctor_id = session.get("doctor_id")
    # Filter patients for current doctor
    doctor_patients = [p for p in patients if p.get("doctor_id") == doctor_id]

    # Get doctor info
    doctor_name = session.get("doctor_name", "Doctor")

    return render_template(
        "index.html", patients=doctor_patients, doctor_name=doctor_name
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page"""
    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        email = data.get("email", "")
        password = data.get("password", "")

        load_doctors()

        # Find doctor by email
        doctor = next((d for d in doctors if d["email"] == email), None)

        if doctor and verify_password(password, doctor["password"]):
            session["doctor_id"] = doctor["id"]
            session["doctor_name"] = doctor["name"]

            if request.is_json:
                return jsonify({"success": True})
            else:
                return redirect(url_for("index"))
        else:
            if request.is_json:
                return jsonify({"success": False, "message": "Invalid credentials"})
            else:
                flash("Invalid credentials")
                return render_template("login.html")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup page"""
    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        name = data.get("name", "")
        email = data.get("email", "")
        password = data.get("password", "")
        specialization = data.get("specialization", "")

        load_doctors()

        # Check if email already exists
        if any(d["email"] == email for d in doctors):
            if request.is_json:
                return jsonify(
                    {"success": False, "message": "Email already registered"}
                )
            else:
                flash("Email already registered")
                return render_template("signup.html")

        # Create new doctor
        doctor_id = len(doctors) + 1
        new_doctor = {
            "id": doctor_id,
            "name": name,
            "email": email,
            "password": hash_password(password),
            "specialization": specialization,
            "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        doctors.append(new_doctor)
        save_doctors()

        # Auto login after signup
        session["doctor_id"] = doctor_id
        session["doctor_name"] = name

        if request.is_json:
            return jsonify({"success": True})
        else:
            return redirect(url_for("index"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for("login"))


@app.route("/add_patient", methods=["POST"])
@login_required
def add_patient():
    """Add a new patient"""
    data = request.get_json()
    doctor_id = session.get("doctor_id")

    # Generate simple ID
    patient_id = len(patients) + 1

    patient = {
        "id": patient_id,
        "doctor_id": doctor_id,
        "name": data.get("name", ""),
        "age": data.get("age", ""),
        "gender": data.get("gender", ""),
        "phone": data.get("phone", ""),
        "condition": data.get("condition", ""),
        "admission_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Active",
    }

    patients.append(patient)
    save_patients()

    return jsonify({"success": True, "patient": patient})


@app.route("/delete_patient/<int:patient_id>", methods=["DELETE"])
@login_required
def delete_patient(patient_id):
    """Delete a patient (only if it belongs to current doctor)"""
    global patients
    doctor_id = session.get("doctor_id")

    # Only delete if patient belongs to current doctor
    patients = [
        p
        for p in patients
        if not (p["id"] == patient_id and p.get("doctor_id") == doctor_id)
    ]
    save_patients()
    return jsonify({"success": True})


@app.route("/update_status/<int:patient_id>", methods=["POST"])
@login_required
def update_status(patient_id):
    """Update patient status (only if it belongs to current doctor)"""
    data = request.get_json()
    new_status = data.get("status", "Active")
    doctor_id = session.get("doctor_id")

    for patient in patients:
        if patient["id"] == patient_id and patient.get("doctor_id") == doctor_id:
            patient["status"] = new_status
            break

    save_patients()
    return jsonify({"success": True})


if __name__ == "__main__":
    load_patients()
    load_doctors()
    app.run(debug=True, port=5054)
