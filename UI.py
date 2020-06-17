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
from main import hoax_detection
import tkinter  as tk


class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("Hoax_Detection_Project")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)
        
    def changeText(self):
        self.text.set("Text updated") 

def read():
    code = txt.get("1.0","end-1c")
    # text_var.set('abc') 
    
    text.set('Result: ')
    text2.set('')
    text3.set('')
    text4.set('')
    
    result = hoax_detection(code)
    

    # text.set('Result: \nMultinomial:' + str(result['multinomial']) + '\nPassive Aggresive:' + str(result['passive']) + '\nSVM:' + str(result['svm']))
    text2.set('Multinomial:' + str(result['multinomial']))
    text3.set('Passive Aggresive:' + str(result['passive']))
    text4.set('SVM:' + str(result['svm']))
    
    
root=Tk()
root.title("Hoax Detection Project")

scrollbar = Scrollbar(root)

#size of the window
root.geometry("730x630")


bigFont = tkFont.Font(family='Roboto', size=11)
txt = scrolledtext.ScrolledText(root, height=19, width=85,font = bigFont, bg='#ffffff', fg='#000000')

txt.grid(column=0, row=0)
txt.place(x=10, y= 10)
txt.config(insertbackground='#EEEEEE')

text = tk.StringVar(root)
text.set("Result: ")
label = tk.Label(root, textvariable=text)
label.place(x=10, y=450, anchor='sw')


text2 = tk.StringVar(root)
text2.set("")
label2 = tk.Label(root, textvariable=text2)
label2.place(x=10, y=475, anchor='sw')



text3 = tk.StringVar(root)
text3.set("")
label3 = tk.Label(root, textvariable=text3)
label3.place(x=10, y=500, anchor='sw')


text4 = tk.StringVar(root)
text4.set("")
label4 = tk.Label(root, textvariable=text4)
label4.place(x=10, y=525, anchor='sw')

# w = tk.Label(root, text="Hello Tkinter!")
# w.place(x=10, y=450, anchor='sw')
# w.pack()

menu = Menu(root)
root.config(menu=menu)

file = Menu(menu)


convertBtn = Button(root, text="Detect", command=lambda: read(),bg='#00579a',fg='#ecf0f1', height=1, width=98)

convertBtn.place(x=10, y=590)

text_var = StringVar(root)
lbl = Label(root, textvariable=text_var, fg='red')
lbl.place(x=10, y=167)

root.mainloop()
