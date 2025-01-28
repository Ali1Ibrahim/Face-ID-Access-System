Features :
-Face Recognition: Uses face_recognition to detect and identify faces in real time.

-User Management: Allows new users to register their face data, stored locally in an organized directory.

-Supabase Integration: Synchronizes user data with a Supabase database, enabling real-time updates for access control.

-Customizable Accuracy: Set thresholds for recognition accuracy to balance precision and performance.

-Live Identification: Real-time face detection with a webcam and dynamic access updates in the database.

-User-Friendly GUI: Built with Tkinter, providing an intuitive interface for saving users, starting live identification, and managing access.

Tech Stack

-Programming Language: Python

Libraries:
-face_recognition for face detection and encoding

-opencv-python for webcam integration and video processing

-supabase for cloud-based user management

-tkinter for GUI

-numpy for numerical operations

-Database: Supabase (PostgreSQL backend)

How It Works

Register a User:

Capture 10 face images of the user via webcam.
Save the images locally and create a face encoding for the user.
Store the user’s name and access status in Supabase.


Live Identification:

-Start real-time face detection with the webcam.

-Compare detected faces with stored encodings to identify users.

-Automatically update access status in Supabase.
