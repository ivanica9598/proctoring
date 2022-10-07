import cv2
import os
import numpy as np

import firebase_admin
from firebase_admin import credentials, storage, firestore


class Database:
    def __init__(self):
        cred = credentials.Certificate("database/settings.json")
        firebase_admin.initialize_app(cred, {'storageBucket': 'online-testing-95618.appspot.com'})
        self.fs = firestore.client()
        self.bucket = storage.bucket()

    def add_user_to_database(self, id_number, first_name, last_name, email, image_path):
        user = self.fs.collection('students').document()
        data = {
            "id": user.id,
            "id_number": id_number,
            "first_name": first_name,
            "last_name": last_name,
            "email": email
        }

        # Add user to database
        self.fs.collection('students').document(user.id).set(data)
        # Add user img to storage
        storage_img_path = 'students/' + str(user.id) + "/profile_img"
        blob = self.bucket.blob(storage_img_path)
        blob.upload_from_filename(image_path)
        # Put url in database
        user.update({"image": storage_img_path})

    def load_user(self, id_number):
        user = self.fs.collection("students").where("id_number", "==", id_number).get()
        user = user[0].to_dict()

        blob = self.bucket.get_blob(user["image"])
        arr = np.frombuffer(blob.download_as_string(), np.uint8)
        img = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555)

        return user, img

    def add_report(self, id_number, video_path):
        # Add video report to storage
        storage_video_path = 'students/' + str(id_number) + "/" + os.path.basename(video_path)
        blob = self.bucket.blob(storage_video_path)
        blob.upload_from_filename(video_path)
