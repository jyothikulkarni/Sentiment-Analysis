import tkinter as tk
from tkinter import ttk, Canvas, NW
from PIL import Image, ImageTk
import tkinter.font as tkFont
import prediction



window = tk.Tk()
style = ttk.Style()
style.configure("BW.TLabel",foreground="white",background="black")

large_font = ('Verdana',20)
window.title("Sentiment Analysis - Helper for the Online Shoping")
window.minsize(800, 600)

fontStyle = tkFont.Font(family="Lucida Grande", size=15)
titleStyle = tkFont.Font(family="Lucida Grande", size=24)
resultStyle = tkFont.Font(family="Lucida Grande", size=20)

def clickMe():
    user_input = name.get()
    ans = prediction.predictFromModel(user_input)
    print(ans)
    path = "images/" + ans.lower() + ".jpg"
    print(path)
    global image, img
    image = Image.open(path)
    image = image.resize((122, 130), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    canvas.create_image((0, 0), image=img, anchor="nw")
    window.update_idletasks()
    if str(ans)=='pos':
        resultTxt.configure(text='Customer like this product. ' + 'we can keep this product.')
    elif str(ans)=='neg':
        resultTxt.configure(text="Customer don't like this product. " + 'we should not keep this product.')
    else:
        resultTxt.configure(text='Customer somewhat satisfied by this product')
    #resultTxt.configure(text='Prediction -- ' + str(ans))


label = ttk.Label(window, text="Online Shoping Products review Analyzer", font=titleStyle)
label.grid(column=0, row=0)
label.place(x=400, y=50, anchor="center")



name = tk.StringVar(value='')
nameEntered = ttk.Entry(window, width=35, textvariable=name, font=large_font)
nameEntered.grid(column=1, row=1)
nameEntered.place(x=400, y=190, anchor="center")

button = ttk.Button(window, text="Analyse", command=clickMe)
button.grid(column=0, row=2)
button.place(x=400, y=280, anchor="center",width=160,height=30)

resultTxt = ttk.Label(window, text="", font=resultStyle)
resultTxt.grid(column=0, row=0)
resultTxt.place(x=400, y=360, anchor="center")

canvas = Canvas(window, width = 120, height = 130)
canvas.place(x=330, y=400)
# canvas.pack()
image = Image.open("images/white.png")
image = image.resize((122,130), Image.ANTIALIAS)
img = ImageTk.PhotoImage(image)
canvas.create_image(0, 0, anchor="nw", image=img)

window.mainloop()
