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
        data = {
            "id_number": id_number,
            "first_name": first_name,
            "last_name": last_name,
            "email": email
        }
        # Add user to database
        user = self.fs.collection('users').document()
        self.fs.collection('users').document(user.id).set(data)
        # Add user img to storage
        storage_img_path = 'images/profile_images/' + str(user.id) + "_" + os.path.basename(image_path)
        blob = self.bucket.blob(storage_img_path)
        blob.upload_from_filename(image_path)
        # Put url in database
        user.update({"image": storage_img_path})

    def load_user(self, id_number):
        user = self.fs.collection("users").where("id_number", "==", id_number).get()
        user = user[0].to_dict()

        blob = self.bucket.get_blob(user["image"])  # blob
        arr = np.frombuffer(blob.download_as_string(), np.uint8)  # array of bytes
        img = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555)

        return user, img
