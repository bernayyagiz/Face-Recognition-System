#Importing libraries
import cv2
import os
import numpy as np
import tkinter as tk
from PIL import ImageTk
import PIL.Image
from tkinter import *

window = tk.Tk()
window.title("Face Recognition System")
window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
window.geometry("850x850")
window.resizable(0, 0)
message = tk.Label(window, text="Face Recognition System",bg="blue", fg="white", width=30, height=3, font=('times', 30, 'bold'))
message.place(x=50, y=20)

#Getting images and labels for training
def images_labels(path="Yaletrain"):
    width_d, height_d = 200, 200  #Resizing images for Eigenface and Fisherface methods
    imgPaths=[os.path.join(path, filename) for filename in os.listdir(path)]
    images=[]
    labels=[]
    for imgPath in imgPaths:
        img = PIL.Image.open(imgPath).convert('L')
        img_np = np.array(img,'uint8')
        faceId = int(os.path.split(imgPath)[1].split(".")[0].replace("subject", ""))
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = face_detector.detectMultiScale(img_np)
        for (x,y,w,h) in faces:
            images.append(cv2.resize(img_np[y:y+h,x:x+w], (width_d, height_d)))
            labels.append(faceId)
    res = "Training Finished!"
    message.configure(text=res)
    return images, labels

#Training classifiers with images and labels from images_labels function
def trainClassifier1():
    images,labels=images_labels("Yaletrain")
    recognizer1 = cv2.face.LBPHFaceRecognizer_create()
    recognizer1.train(images, np.array(labels))
    return recognizer1

def trainClassifier2():
    images, labels = images_labels("Yaletrain")
    recognizer2 = cv2.face.EigenFaceRecognizer_create()
    recognizer2.train(images, np.array(labels))
    return recognizer2

def trainClassifier3():
    images, labels = images_labels("Yaletrain")
    recognizer3 = cv2.face.FisherFaceRecognizer_create()
    recognizer3.train(images, np.array(labels))
    return recognizer3

def button_click_exit_mainloop(event): #For changing images with mouse-click
        event.widget.quit()

# Testing
def testimages1():
    window.bind("<Button>", button_click_exit_mainloop)
    TR = 0  #For keeping true recognized face number
    FR = 0  #For keeping false recognized face number
    imgPathtest = [os.path.join("Yaletest", filename) for filename in os.listdir("Yaletest")]
    old_label_image = None
    old_label=None
    width_d, height_d = 200, 200

    for imgPath in imgPathtest:
        test_img = PIL.Image.open(imgPath).convert('L')
        test_imgnp = np.array(test_img, 'uint8')
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(test_imgnp)
        for (x, y, w, h) in faces:
            face_recognizer = trainClassifier1()
            # Detected faces are used for prediction
            predicted_label, confidence = face_recognizer.predict(cv2.resize(test_imgnp[y:y + h, x:x + w], (width_d, height_d)))
            actual_label = int(os.path.split(imgPath)[1].split(".")[0].replace("subject", ""))
            actual_name = str(os.path.split(imgPath)[1])
            if actual_label == predicted_label:
                img = ImageTk.PhotoImage(test_img)
                label_image = tk.Label(window, image=img,bd=2,relief="solid")
                label_image.place(x=380, y=200, width=test_img.size[0], height=test_img.size[1])
                if old_label_image is not None: #For blocking to show images on top of each other
                    old_label_image.destroy()
                old_label_image = label_image

                label1 = tk.Label(window, text=actual_name, fg="green", bg="white",font=('times', 20, ' bold '),bd=1,relief="solid")
                label1.pack(anchor=W)
                label1.place(x=380, y=510)
                if old_label is not None: #For blocking to show labels on top of each other
                    old_label.destroy()
                old_label = label1
                window.mainloop()
                TR += 1
            else:
                img = ImageTk.PhotoImage(test_img)
                label_image = tk.Label(window, image=img,bd=2,relief="solid")
                label_image.place(x=380, y=200, width=test_img.size[0], height=test_img.size[1])
                if old_label_image is not None: #For blocking to show images on top of each other
                    old_label_image.destroy()
                old_label_image = label_image

                label1 = tk.Label(window, text=actual_name+" predicted as subject"+str(predicted_label), fg="red", bg="white", font=('times', 20, ' bold '),bd=1,relief="solid")
                label1.pack(anchor=W)
                label1.place(x=300, y=510)
                if old_label is not None: #For blocking to show labels on top of each other
                    old_label.destroy()
                old_label = label1
                window.mainloop()
                label1.destroy()
                FR += 1
    res1 = "Testing Finished!"
    message.configure(text=res1)
    RR=float((TR/(TR+FR))*100)
    label2 = tk.Label(window,text="Recognition Rate: %"+str("{0:.2f}".format(RR)),fg="blue",bg="white",font=('times', 20, ' bold '))
    label2.place(x=380, y=510)

def testimages2():
    window.bind("<Button>", button_click_exit_mainloop)
    TR = 0
    FR = 0
    imgPathtest = [os.path.join("Yaletest", filename) for filename in os.listdir("Yaletest")]
    old_label_image = None
    old_label = None
    width_d, height_d = 200, 200
    for imgPath in imgPathtest:
        test_img = PIL.Image.open(imgPath).convert('L')
        test_imgnp = np.array(test_img, 'uint8')
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(test_imgnp)
        for (x, y, w, h) in faces:
            face_recognizer = trainClassifier2()
            # Detected faces are used for prediction
            predicted_label, confidence = face_recognizer.predict(cv2.resize(test_imgnp[y:y + h, x:x + w], (width_d, height_d)))
            actual_label = int(os.path.split(imgPath)[1].split(".")[0].replace("subject", ""))
            actual_name = str(os.path.split(imgPath)[1])
            if actual_label == predicted_label:
                img = ImageTk.PhotoImage(test_img)
                label_image = tk.Label(window, image=img, bd=2, relief="solid")
                label_image.place(x=380, y=200, width=test_img.size[0], height=test_img.size[1])
                if old_label_image is not None: #For blocking to show images on top of each other
                    old_label_image.destroy()
                old_label_image = label_image

                label1 = tk.Label(window, text=actual_name, fg="green", bg="white", font=('times', 20, ' bold '), bd=1,relief="solid")
                label1.pack(anchor=W)
                label1.place(x=380, y=510)
                if old_label is not None: #For blocking to show labels on top of each other
                    old_label.destroy()
                old_label = label1
                window.mainloop()
                TR += 1
            else:
                img = ImageTk.PhotoImage(test_img)
                label_image = tk.Label(window, image=img, bd=2, relief="solid")
                label_image.place(x=380, y=200, width=test_img.size[0], height=test_img.size[1])
                if old_label_image is not None: #For blocking to show images on top of each other
                    old_label_image.destroy()
                old_label_image = label_image

                label1 = tk.Label(window, text=actual_name + " predicted as subject" + str(predicted_label), fg="red",
                                   bg="white", font=('times', 20, ' bold '), bd=1, relief="solid")
                label1.pack(anchor=W)
                label1.place(x=300, y=510)
                if old_label is not None: #For blocking to show labels on top of each other
                    old_label.destroy()
                old_label = label1
                window.mainloop()
                label1.destroy()
                FR += 1
    res1 = "Testing Finished!"
    message.configure(text=res1)
    RR = float((TR / (TR + FR)) * 100)
    label2 = tk.Label(window, text="Recognition Rate: %" + str("{0:.2f}".format(RR)), fg="blue", bg="white",
                       font=('times', 20, ' bold '))
    label2.place(x=380, y=510)

def testimages3():
    window.bind("<Button>", button_click_exit_mainloop)
    TR = 0
    FR = 0
    imgPathtest = [os.path.join("Yaletest", filename) for filename in os.listdir("Yaletest")]
    old_label_image = None
    old_label = None
    width_d, height_d = 200, 200
    for imgPath in imgPathtest:
        test_img = PIL.Image.open(imgPath).convert('L')
        test_imgnp = np.array(test_img, 'uint8')
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(test_imgnp)
        for (x, y, w, h) in faces:
            face_recognizer = trainClassifier3()
            # Detected faces are used for prediction
            predicted_label, confidence = face_recognizer.predict(cv2.resize(test_imgnp[y:y + h, x:x + w], (width_d, height_d)))
            actual_label = int(os.path.split(imgPath)[1].split(".")[0].replace("subject", ""))
            actual_name = str(os.path.split(imgPath)[1])
            if actual_label == predicted_label:
                img = ImageTk.PhotoImage(test_img)
                label_image = tk.Label(window, image=img, bd=2, relief="solid")
                label_image.place(x=380, y=200, width=test_img.size[0], height=test_img.size[1])
                if old_label_image is not None: #For blocking to show images on top of each other
                    old_label_image.destroy()
                old_label_image = label_image

                label1 = tk.Label(window, text=actual_name, fg="green", bg="white", font=('times', 20, ' bold '), bd=1,
                                   relief="solid")
                label1.pack(anchor=W)
                label1.place(x=380, y=510)
                if old_label is not None: #For blocking to show labels on top of each other
                    old_label.destroy()
                old_label = label1
                window.mainloop()
                TR += 1

            else:
                img = ImageTk.PhotoImage(test_img)
                label_image = tk.Label(window, image=img,bd=2, relief="solid")
                label_image.place(x=380, y=200, width=test_img.size[0], height=test_img.size[1])
                if old_label_image is not None: #For blocking to show images on top of each other
                    old_label_image.destroy()
                old_label_image = label_image

                label1 = tk.Label(window, text=actual_name + " predicted as subject" + str(predicted_label), fg="red",
                                   bg="white", font=('times', 20, ' bold '), bd=1, relief="solid")
                label1.pack(anchor=W)
                label1.place(x=300, y=510)
                if old_label is not None: #For blocking to show labels on top of each other
                    old_label.destroy()
                old_label = label1
                window.mainloop()
                label1.destroy()
                FR += 1
    res1 = "Testing Finished!"
    message.configure(text=res1)
    RR = float((TR / (TR + FR)) * 100)
    label2 = tk.Label(window, text="Recognition Rate: %" + str("{0:.2f}".format(RR)), fg="blue", bg="white",
                       font=('times', 20, ' bold '),bd=1, relief="solid")
    label2.place(x=380, y=510)

lbph = tk.Radiobutton(window, text="LBPH", value=1,
                  command=trainClassifier1,fg="blue",bg="white",font=('times', 20, ' bold '))
lbph.place(x=80, y=200)

eigen = tk.Radiobutton(window, text="Eigenface", value=2,
                  command=trainClassifier2,fg="blue",bg="white",font=('times', 20, ' bold '))
eigen.place(x=80, y=350)

fisher = tk.Radiobutton(window, text="Fisherface", value=3,
                  command=trainClassifier3,fg="blue",bg="white",font=('times', 20, ' bold '))
fisher.place(x=80, y=500)

testlbph = tk.Button(window, text="Test",
                     command=testimages1, fg="white", bg="blue",
                     width=15, height=1, activebackground="purple",
                     font=('times', 15, ' bold '))
testlbph.place(x=80, y=250)

testeigen = tk.Button(window, text="Test",
                     command=testimages2, fg="white", bg="blue",
                     width=15, height=1, activebackground="purple",
                     font=('times', 15, ' bold '))
testeigen.place(x=80, y=400)

testfisher = tk.Button(window, text="Test",
                     command=testimages3, fg="white", bg="blue",
                     width=15, height=1, activebackground="purple",
                     font=('times', 15, ' bold '))
testfisher.place(x=80, y=550)

window.mainloop()

