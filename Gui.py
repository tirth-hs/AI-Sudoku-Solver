import tkinter as tk
from PIL import ImageTk, Image
import sys
import os

class CanvasButton:
    def __init__(self, canvas):
        self.canvas = canvas
        self.button = tk.Button(canvas,image=bt_photo,command=self.action)
        self.id = canvas.create_window(780, 325, width=140, height=50,window=self.button)
    def action(self):
        os.system('python Sudoku.py')
        root.destroy()
root = tk.Tk()
root.resizable(width=False, height=False)

imgpath = 'background.png'
img = Image.open(imgpath)
photo = ImageTk.PhotoImage(img)
bt_img = Image.open('button.png')
bt_photo = ImageTk.PhotoImage(bt_img)
canvas = tk.Canvas(root,width=990,height=557)
canvas.create_image(0,0,anchor=tk.NW,image=photo)
canvas.pack()

CanvasButton(canvas) 

root.mainloop()