import cv2
import numpy as np

import firebase_admin
from firebase_admin import credentials, storage, firestore


class Database:
    def __init__(self):
        cred = credentials.Certificate("database/settings.json")
        firebase_admin.initialize_app(cred, {'storageBucket': 'online-testing-95618.appspot.com'})
        self.fs = firestore.client()
        self.bucket = storage.bucket()

    def add_student(self, id_number, first_name, last_name, image_path):
        try:
            user = self.fs.collection('students').document()
            data = {
                "id": user.id,
                "id_number": id_number,
                "first_name": first_name,
                "last_name": last_name
            }

            # Add user to database
            self.fs.collection('students').document(user.id).set(data)
            # Add user img to storage
            storage_img_path = 'students/' + str(user.id) + "/profile_img"
            blob = self.bucket.blob(storage_img_path)
            blob.upload_from_filename(image_path)
            # Put url in database
            user.update({"image": storage_img_path})

            print("Add student: Done.")
        except:
            print("Add student: Error.")

    def load_student(self, id_number):
        try:
            user = self.fs.collection("students").where("id_number", "==", id_number).get()
            user = user[0].to_dict()

            blob = self.bucket.get_blob(user["image"])
            arr = np.frombuffer(blob.download_as_string(), np.uint8)
            img = cv2.imdecode(arr, cv2.COLOR_BGR2BGR555)

            return user, img
        except:
            print("Load student: Error")
            return None, None

    def delete_student(self, id_number):
        try:
            student = self.fs.collection("students").where("id_number", "==", id_number).get()[0]
            self.fs.collection("students").document(student.id).delete()
            print("Delete student: Done.")
        except:
            print("Delete student: Error.")

    def add_report(self, id_number, test_id, video_name, video_path):
        try:
            storage_video_path = 'students/' + str(id_number) + "/" + test_id + "/" + video_name
            blob = self.bucket.blob(storage_video_path)
            blob.upload_from_filename(video_path)
        except:
            print("Add report: Error")

    def add_test(self, id_number, h, m, s):
        try:
            test = self.fs.collection('tests').document()
            data = {
                "id": test.id,
                "id_number": id_number,
                "hours": h,
                "minutes": m,
                "seconds": s
            }
            self.fs.collection('tests').document(test.id).set(data)
            print("Add test: Done.")
        except:
            print("Add test: Error.")

    def load_test(self, id_number):
        try:
            test = self.fs.collection("tests").where("id_number", "==", id_number).get()
            test = test[0].to_dict()

            return test
        except:
            print("Load test: Error")
            return None

    def delete_test(self, id_number):
        try:
            test = self.fs.collection("tests").where("id_number", "==", id_number).get()[0]
            self.fs.collection("tests").document(test.id).delete()
            print("Delete test: Done.")
        except:
            print("Delete test: Error.")
