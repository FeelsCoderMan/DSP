import face_recognition
import numpy as np
image0 = face_recognition.load_image_file('./Known_Images/Muhammed_Halit.jpg')
image_encoding0 = face_recognition.face_encodings(image0)[0]

image1 = face_recognition.load_image_file('./Known_Images/Halit_Murat.jpg')
image_encoding1 = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file('./Known_Images/Onur_Aksoy.jpg')
image_encoding2 = face_recognition.face_encodings(image2)[0]

image3 = face_recognition.load_image_file('./Known_Images/Mahmut_hoca.png')
image_encoding3 = face_recognition.face_encodings(image3)[0]


encodings = np.append(image_encoding0,image_encoding1)
encodings = np.append(encodings,image_encoding2)


image_names = ['Muhammed Halit Tokluoğlu','Halit Murat','Onur Aksoy', 'Mahmut Emin Çelik']
