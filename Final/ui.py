import tkinter as tk
from tkinter import *
import csv
import numpy as np
import random
import tkinter.filedialog as filedialog
import requests
from tkinter import *  
from tkinter.filedialog import askopenfilename  
from PIL import Image, ImageTk

dept = []
opt = []
def choosepic():  
    path_=askopenfilename()  
    img_open = Image.open(path_) 
    img_open = img_open.resize((540,540),Image.ANTIALIAS) 
    img=ImageTk.PhotoImage(img_open)  
    l1.config(image=img)  
    l1.image=img
    
def predict():
    print('prdeict')

chosen_taglist = []

def construct_gui():
    global window
    window = tk.Tk()
    window.title('Cognitive Computing Final Demo')
    window.geometry('1000x500')
    global path; global e1; global l1;
    path=StringVar()  
    Button(window,text='upload image',command=choosepic).place(x=120,y=5)
    Button(window,text='predict image',command=predict).place(x=250,y=5)  
    l1=Label(window)  
    l1.place(x=10,y=60)

    tag = tk.StringVar()
    l = tk.Label(window, 
                textvariable=tag,
                bg='white',     
                font=('Arial', 15),     
                width=15, height=2
        )
    l.place(x=743,y = 10)


    
    chosen_var = tk.StringVar(window)
    chosen_var.set("null")
    w = OptionMenu(window, chosen_var, *opt)
    w.place(x=802,y=60)

    picked_tag = tk.StringVar()
    def add_to_list():
        tmp = chosen_var.get()
        chosen_taglist.append(tmp)
        picked_tag.set(','.join(chosen_taglist))

    picked = tk.Label(window,
                     textvariable=picked_tag,
                     bg='white',
                     font=('Arial', 15),
                     width=50, height=2)
    picked.place(x=20,y=100)

    button = Button(window, text="OK", command=add_to_list)
    button.place(x=802,y=100)
    
    window.mainloop()


if __name__ == '__main__':
    opt = [*range(10)]
    construct_gui()
