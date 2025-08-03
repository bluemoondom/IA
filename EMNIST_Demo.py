# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:06:32 2023

@author: dominika
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from PIL import ImageGrab,ImageChops
from tensorflow import keras as tf
from tensorflow import expand_dims as ed
import tkinter as tk

MonReseau = tf.models.load_model('MonReseau_new.h5')

W_Root=tk.Tk()
W_Root.title('Rozpoznávání znaků')
W_Root.resizable(False,False)
W_Root.configure(bg="darkgrey")

def Fct_ClicMouse(event):
    global X0,Y0
    X0,Y0 = event.x,event.y

def Fct_MvtClicMouse(event):
    global X0,Y0
    x1,y1 = event.x,event.y
    C_Draw.create_line(X0,Y0,x1,y1,width=20,capstyle='round',fill='black')
    X0,Y0=x1,y1

def Fct_Recognize():
    x,y = C_Draw.winfo_rootx(),C_Draw.winfo_rooty()
    w,h = C_Draw.winfo_width(),C_Draw.winfo_height()
    ImgPIL = ImageGrab.grab(bbox=(x+2,y+2,x+w-2,y+h-2))
    ImgPIL = ImageChops.invert(ImgPIL.convert('L'))
    (xMin,yMin,xMax,yMax) = ImgPIL.getbbox()
    MyLen = min(400, 1.2*max(xMax-xMin,yMax-yMin))
    dx,dy = (MyLen-(xMax-xMin))//2,(MyLen-(yMax-yMin))//2
    ImgPIL = ImgPIL.crop((xMin-dx,yMin-dy,xMax+dx,yMax+dy))
    ImgArray = tf.utils.img_to_array(ImgPIL.resize((28,28)))/255
    S = MonReseau.predict(ed(ImgArray,axis=0),verbose=0)[0]
    L_Conclusion["text"] = '{}/{}'.format(chr(65+S.argmax()),chr(97+S.argmax()))
    L_Confiance["text"] = 'Pravděpodobnost: {:.2f} %'.format(100*S.max())
    
def Fct_Delete():
    C_Draw.delete(tk.ALL)
    L_Conclusion['text'] = '?'
    L_Confiance['text'] = 'Pravděpodobnost: ?'
    
C_Draw = tk.Canvas(W_Root,width=400,height=400,bg='white',cursor='dot')
C_Draw.pack(side=tk.LEFT,padx=10,pady=10)

C_Draw.bind('<Button-1>',Fct_ClicMouse)
C_Draw.bind('<B1-Motion>',Fct_MvtClicMouse)

B_Delete = tk.Button(W_Root,font=('Arial',16,'bold'),text="Smazat",command=Fct_Delete)
B_Delete.pack(ipady=10,padx=10,pady=10,expand=True,fill=tk.X)

F_Reco = tk.Frame(bg='white',bd=3,relief='ridge')
F_Reco.pack(padx=10,pady=10,expand=True,fill=tk.X)

B_Recognize = tk.Button(F_Reco,font=('Arial',16),text="Rozpoznat",command=Fct_Recognize)
B_Recognize.pack(ipadx=10,padx=15,pady=15,expand=True,fill=tk.X)

L_Conclusion = tk.Label(F_Reco,font=('Arial',50,'bold'),bg='white',text='?')
L_Conclusion.pack(pady=10,expand=True)

L_Confiance = tk.Label(F_Reco,font=('Arial',12,'italic'),bg='white',text='Pravděpodobnost: ?')
L_Confiance.pack(pady=10,expand=True)

B_Close = tk.Button(W_Root,font=('Arial',12,'italic'),text='Ukončit',command=W_Root.destroy)
B_Close.pack(ipady=10,padx=10,pady=10,expand=True,fill=tk.X)

W_Root.mainloop()
    
    
    
    