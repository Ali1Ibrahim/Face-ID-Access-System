import cv2
import face_recognition
import os
from tkinter import Tk, Button, messagebox, simpledialog
import numpy as np
from supabase import create_client, Client

# Supabase configuration
SUPABASE_URL = ""
SUPABASE_API_KEY = ""

supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# Directory to store face encodings
ENCODINGS_DIR = "face_encodings/"
os.makedirs(ENCODINGS_DIR, exist_ok=True)

# Known face encodings and names
known_face_encodings = []
known_face_names = []

def load_encodings():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for user_folder in os.listdir(ENCODINGS_DIR):
        user_path = os.path.join(ENCODINGS_DIR, user_folder)
        if os.path.isdir(user_path):
            for file in os.listdir(user_path):
                file_path = os.path.join(user_path, file)
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(user_folder)

# Insert user data to Supabase
def insert_to_supabase(name):
    try:
        response = supabase.table("faces").insert({"name": name, "access": False}).execute()
        if not response.data:
            raise Exception("Failed to insert data to Supabase.")
        print(f"User {name} added to Supabase successfully.")
    except Exception as e:
        print(f"Error inserting data to Supabase: {e}")

# Update user access in Supabase
def update_access_in_supabase(name, access):
    try:
        response = supabase.table("faces").update({"access": access}).eq("name", name).execute()
        if not response.data:
            raise Exception("Failed to update data in Supabase.")
        print(f"Access for {name} updated to {access}.")
    except Exception as e:
        print(f"Error updating data in Supabase: {e}")

# Calculate accuracy and check against threshold
def is_match(face_encoding, known_encodings, threshold=0.65):
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    accuracy = 1 - face_distances[best_match_index]
    if accuracy >= threshold:
        return best_match_index, accuracy
    return -1, None

# Live face identification with threshold
def live_identification(threshold=0.65):
    active_names = set()
    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible!")
        return

    messagebox.showinfo("Info", "Press 'q' to quit live identification.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_names = set()
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match_index, accuracy = is_match(face_encoding, known_face_encodings, threshold)
            name = "Unknown"
            access = False

            if match_index != -1:
                name = known_face_names[match_index]
                access = True

            detected_names.add(name)
            if name != "Unknown":
                update_access_in_supabase(name, access)

            # Draw rectangle and label
            color = (0, 255, 0) if access else (0, 0, 255)
            label = f"{name} ({accuracy:.2%})" if access else "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        for inactive_name in active_names - detected_names:
            update_access_in_supabase(inactive_name, False)

        active_names = detected_names
        cv2.imshow("Live Identification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Save a new user's face encoding
def save_user():
    root = Tk()
    root.withdraw()
    name = simpledialog.askstring("Input", "Enter the name of the new user:")
    if not name:
        messagebox.showerror("Error", "Name cannot be empty!")
        return

    user_dir = os.path.join(ENCODINGS_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible!")
        return

    messagebox.showinfo("Info", f"Capturing 10 frames for {name}. Look at the camera.")

    count = 0
    while count < 10:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Save User - Capture Mode", frame)
        file_path = os.path.join(user_dir, f"{name}_{count}.jpg")
        cv2.imwrite(file_path, frame)
        count += 1
        print(f"Captured frame {count} for {name}")
        cv2.waitKey(1000)

    cap.release()
    cv2.destroyAllWindows()
    load_encodings()
    insert_to_supabase(name)
    messagebox.showinfo("Info", f"Frames captured and {name} added.")

# GUI Application
def main():
    load_encodings()
    root = Tk()
    root.title("Face Recognition System")

    Button(root, text="Save New User", command=save_user, width=30, height=2).pack(pady=20)
    Button(root, text="Live Identification", command=lambda: live_identification(0.65), width=30, height=2).pack(pady=20)
    Button(root, text="Exit", command=root.destroy, width=30, height=2).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
