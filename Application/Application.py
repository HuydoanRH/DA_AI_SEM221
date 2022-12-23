from tkinter import *
from tkinter.filedialog import askopenfilename
import customtkinter
import os
from PIL import Image, ImageTk
import cv2
from keras.models import load_model
from gender_recognition import detect_recog
# import threading
from keras_preprocessing.image import img_to_array
import numpy as np

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.check_camera = False
        self.classes = ['woman', 'man']
        global cam
        global frame
        global ret
        global photo
        global img_save
        global status_predict
        self.title("GENDER RECOGNITION")  # title of the GUI window
        self.geometry("700x450")

        self.model = load_model('gender_detect.model')       
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./Asset")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "bk_logo.png")), size=(50, 50))
        self.image_icon_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "bk_logo.png")), size=(20, 20))
        self.cameraIcon = customtkinter.CTkImage(Image.open(os.path.join(image_path, "cctv.jpg")), size=(40, 40))
        self.home_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "home_light.png")), size=(20, 20))
        self.detect_realtime = customtkinter.CTkImage(Image.open(os.path.join(image_path, "logo_cam.png")), size=(20, 19))
        self.detect_image = customtkinter.CTkImage(Image.open(os.path.join(image_path, "logo_folder.png")),dark_image=Image.open(os.path.join(image_path, "logo_folder.png")), size=(20, 20))

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.realtime_button = customtkinter.CTkButton(self.navigation_frame, height=40, border_spacing=10, text="Detect Realtime",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.detect_realtime, anchor="w", command=self.realtime_button_event)
        self.realtime_button.grid(row=2, column=0, sticky="ew")

        self.image_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Detect Image",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.detect_image, anchor="w", command=self.image_button_event)
        self.image_button.grid(row=3, column=0, sticky="ew")



        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        # self.home_frame.grid_columnconfigure(0, weight=1)
        self.home_frame_introduce1 = customtkinter.CTkLabel(self.home_frame, text="FACE DETECTION & GENDER RECOGNITION USING DEEP LEARNING",font=customtkinter.CTkFont(size=15, weight="bold"))
        self.home_frame_introduce2 = customtkinter.CTkLabel(self.home_frame, text="GVHD: VŨ VĂN TIẾN & TRẦN HUY ",font=customtkinter.CTkFont(size=14, weight="bold"))
        self.home_frame_introduce3 = customtkinter.CTkLabel(self.home_frame, text="=================",font=customtkinter.CTkFont(size=13))
        self.home_frame_introduce4 = customtkinter.CTkLabel(self.home_frame, text="SVTH: NGUYẼN TÙNG DƯƠNG ",font=customtkinter.CTkFont(size=14, weight="bold"))
        self.home_frame_introduce5 = customtkinter.CTkLabel(self.home_frame, text="ĐOÀN ĐỨC HUY",font=customtkinter.CTkFont(size=14, weight="bold"))
        self.home_frame_introduce6 = customtkinter.CTkLabel(self.home_frame, text="HOÀNG NHẬT LINH KIỀU",font=customtkinter.CTkFont(size=14, weight="bold"))
        self.home_frame_introduce7 = customtkinter.CTkLabel(self.home_frame, text="TRẦN VIỆT TRUNG",font=customtkinter.CTkFont(size=14, weight="bold"))


        self.home_frame_introduce1.place(x = 35, y= 120)          
        self.home_frame_introduce2.place(x = 150, y = 150)
        self.home_frame_introduce3.place(x = 210, y = 180)
        self.home_frame_introduce4.place(x = 150, y = 210)
        self.home_frame_introduce5.place(x = 196, y = 230)
        self.home_frame_introduce6.place(x = 196, y = 250)
        self.home_frame_introduce7.place(x = 196, y = 270)

        # create realtime frame
        self.realtime_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.realtime_frame_label = customtkinter.CTkLabel(self.realtime_frame, text="")
        self.realtime_frame_label.place(x = -100, y= -80)  
        self.detect_btn = customtkinter.CTkButton(self.realtime_frame, text="START",font=customtkinter.CTkFont(size=14, weight="bold"),
                                                fg_color="white", text_color=('black'), hover_color=("gray70", "gray30"),
                                                anchor='center', command=self.realtime_detect)
                                                # customtkinter.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="bottom", anchor="w")
        self.detect_btn.place(x = 196, y = 350)
        self.stop_btn = customtkinter.CTkButton(self.realtime_frame, text="STOP",font=customtkinter.CTkFont(size=14, weight="bold"),
                                                fg_color="white", text_color=('black'), hover_color=("gray70", "gray30"),
                                                anchor='center', command=self.realtime_stop)
                                                # customtkinter.CTkButton(self.home_frame, text="CTkButton", image=self.image_icon_image, compound="bottom", anchor="w")
        self.stop_btn.place(x = 196, y = 390)
        # create image frame
        self.image_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.image_frame_label = customtkinter.CTkLabel(self.image_frame, text="") 
        self.image_frame_label.place(x = -100, y= -80) 

        self.upload_image_btn = customtkinter.CTkButton(self.image_frame, text="UPLOAD",font=customtkinter.CTkFont(size=14, weight="bold"),
                                                fg_color="white", text_color=('black'), hover_color=("gray70", "gray30"),
                                                anchor='center', command=self.image_button_event_upload)
        self.upload_image_btn.place(x = 196, y = 350)

        self.detect_btn = customtkinter.CTkButton(self.image_frame, text="DETECT",font=customtkinter.CTkFont(size=14, weight="bold"),
                                                fg_color="white", text_color=('black'), hover_color=("gray70", "gray30"),
                                                anchor='center', command=self.image_button_event_detect)
        self.detect_btn.place(x = 196, y= 390) 

        # select default frame
        self.select_frame_by_name("home")

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.realtime_button.configure(fg_color=("gray75", "gray25") if name == "realtime" else "transparent")
        self.image_button.configure(fg_color=("gray75", "gray25") if name == "image" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "realtime":
            self.realtime_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.realtime_frame.grid_forget()
        if name == "image":
            self.image_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.image_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def realtime_button_event(self):
        self.select_frame_by_name("realtime")

    def image_button_event(self):
        self.select_frame_by_name("image")

    def realtime_detect(self):
        print("Using realtime")
        self.check_camera = True
        if self.check_camera:
            self.cam = cv2.VideoCapture(0)
            while self.check_camera:
                if self.check_camera == False:
                    self.cam.release()
                self.ret, self.frame = self.cam.read()
                frame_cpy = self.frame.copy()
                frame_cpy = cv2.cvtColor(frame_cpy, cv2.COLOR_BGR2GRAY)
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                face = self.face_cascade.detectMultiScale(
                    frame_cpy,
                    scaleFactor=1.2,
                    minNeighbors=2,
                    minSize=(30, 30))

                # loop through detected faces
                for idx, f in enumerate(face):
                    print("Detecting face %d:" % (idx + 1))
                    # get corner points of face rectangle
                    (startX, startY) = f[0], f[1]
                    (endX, endY) = f[2], f[3]
                    print("startX", startX)
                    print("startY", startY)
                    print("endX", endX)
                    print("endY", endY)

                    # draw rectangle over face
                    cv2.rectangle(self.frame, (startX, startY), (startX + endX, startY + endY), (0, 255, 0), 2)

                    # crop the detected face region
                    face_crop = self.frame[startY:(startY + endY), startX:(startX + endX)]
                    print("shape[0]", face_crop.shape[0])
                    print("shape[1]", face_crop.shape[1])
                    print("shape[2]", face_crop.shape[2])

                    if (face_crop.shape[0]) < 30 or (face_crop.shape[1]) < 30:
                        continue

                    # draw rectangle over face
                    cv2.rectangle(self.frame, (startX, startY), (startX + endX, startY + endY), (0, 255, 0), 2)

                    # preprocessing for gender detection model
                    face_crop = cv2.resize(face_crop, (96, 96))
                    face_crop = face_crop.astype("float") / 255.0
                    face_crop = img_to_array(face_crop)
                    face_crop = np.expand_dims(face_crop, axis=0)

                    # apply gender detection on face
                    conf = self.model.predict(face_crop, batch_size=32)  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
                    # get label with max accuracy
                    idx = np.argmax(conf)
                    print("result data:", idx)
                    label = self.classes[idx]
                    percents = conf[0][idx] * 100

                    # label = conf[idx] * 100 + "," + label
                    label = "{gender},{percent}".format(gender=label, percent="{:.2f}%".format(percents))
                    Y = startY - 10 if startY - 10 > 10 else startY + 10

                    # write label and confidence above face rectangle
                    cv2.putText(self.frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                self.img_save = self.frame
                img_update = ImageTk.PhotoImage(Image.fromarray(self.frame))
                self.realtime_frame_label.configure(image=img_update, width=750, height=500)
                self.realtime_frame_label.image = img_update
                self.realtime_frame_label.update()
    def realtime_stop(self):
            self.check_camera = False


    def image_button_event_upload(self):
        print("use image")
        if self.check_camera == True:
            self.cam.release()
            self.check_camera = False

        self.filename = askopenfilename(
            filetypes=(("jpg file", "*.jpg"), ("png file", '*.png'), ("All files", " *.* "),))
        self.photo = Image.open(self.filename)
        if (self.photo.width > 650 and  self.photo.height > 400) or (self.photo.width > 650) or (self.photo.height > 400):
            self.photo = self.photo.resize((650, 400))
        self.photo = ImageTk.PhotoImage(self.photo)
        self.image_frame_label.configure(image=self.photo,width=750, height=500)
        self.image_frame_label.image = self.photo

    def image_button_event_detect(self):
        self.temp = self.filename
        self.filename.replace("/", "/")
        print(self.filename)

        self.photo = cv2.imread(self.filename, 1)
        self.photo = detect_recog(self.photo, self.face_cascade, self.model)
        # pritn(self.photo)
        self.photo = cv2.cvtColor(self.photo, cv2.COLOR_BGR2RGB)
        self.img_save = self.photo
        self.photo = Image.fromarray(self.photo)
        if (self.photo.width > 650 and self.photo.height > 400) or (self.photo.width > 650) or (self.photo.height > 400):
            self.photo = self.photo.resize((650, 400))
            self.photo = ImageTk.PhotoImage(self.photo)
            self.image_frame_label.configure(image=self.photo, width=750, height=500)
            self.image_frame_label.image = self.photo
            # self.file = asksaveasfile(mode='w', defaultextension=".png")
            # cv2.imwrite(self.file.name,self.photo)
        print("use predited")    

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)
if __name__ == "__main__":
    app = App()
    app.mainloop()
