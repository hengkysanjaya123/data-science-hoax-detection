

from tkinter import filedialog
from tkinter import *
from tkinter import scrolledtext
from tkinter import messagebox
# import PIL as pl
# from PIL import Image, ImageTk
# import PIL.Image
# import PIL.ImageTk
import os
import tkinter.font as tkFont


class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("Code to Flowchart")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

def read():
    code = txt.get("1.0","end-1c")
    text_var.set('')
    print(code)

    # print("error (ui.py)", error_message)
    #
    # if(error_message != None):
    #     text_var.set(str(error_message))
root=Tk()
root.title("Hoax_Detection_Project")

scrollbar = Scrollbar(root)

#size of the window
root.geometry("730x630")


bigFont = tkFont.Font(family='Monaco', size=11, weight='bold')
txt = scrolledtext.ScrolledText(root, height=23, width=77,font = bigFont, bg='#222222', fg='#EEEEEE')

txt.grid(column=0, row=0)
txt.place(x=10, y= 190)
txt.config(insertbackground='#EEEEEE')

txt.tag_config('comment', foreground='green')

menu = Menu(root)
root.config(menu=menu)

file = Menu(menu)


convertBtn = Button(root, text="Convert", command=lambda: read(),bg='#00579a',fg='#ecf0f1', height=1, width=98)

convertBtn.place(x=10, y=590)

text_var = StringVar(root)
lbl = Label(root, textvariable=text_var, fg='red')
lbl.place(x=10, y=167)

root.mainloop()
