import tkinter as tk
#from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk



##############################################+=============================================================
root = tk.Tk()
root.configure(background="black")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("GUI_Main")



#
label_l2 = tk.Label(root, text="___Spam Email Detection System___",font=("times", 30, 'bold','italic'),
                    background="black", fg="white", width=70, height=2)
label_l2.place(x=0, y=0)

Entry_frame = tk.LabelFrame(root,text="Register or Login",width=1700,height=100,background="teal",font=("Tempus Sanc ITC",15,"bold"))
Entry_frame.place(x=0,y=100)





img=ImageTk.PhotoImage(Image.open("Slide1.jpg"))

img2=ImageTk.PhotoImage(Image.open("Slide2.jpeg"))

img3=ImageTk.PhotoImage(Image.open("Slide3.jpg"))


logo_label=tk.Label()
logo_label.place(x=0,y=200)



# using recursion to slide to next image
x = 1

# function to change to next image
def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img,width=2000,height=1000)
	elif x == 2:
		logo_label.config(image=img2,width=2000,height=1000)
	elif x == 3:
		logo_label.config(image=img3,width=2000,height=1000)
	x = x+1
	root.after(2000, move)

# calling the function
move()

# frame_alpr = tk.LabelFrame(root, text=" --Login & Register-- ", width=800, height=100, bd=5, font=('times', 14, ' bold '),bg="grey")
# frame_alpr.grid(row=0, column=0, sticky='nw')
# frame_alpr.place(x=400, y=600)


#T1.tag_configure("center", justify='center')
#T1.tag_add("center", 1.0, "end")

################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def log():
    from subprocess import call
    call(["python","login.py"])
    #root.destroy()
    
def reg():
    from subprocess import call
    call(["python","registration.py"])
    #root.destroy()
    
def window():
  root.destroy()
  
  

#####################################################################################################################

button1 = tk.Button(Entry_frame, text="Login", command=log, width=15, height=1,font=('times', 15, ' bold '), bg="black", fg="white")
button1.place(x=300, y=20)

button2 = tk.Button(Entry_frame, text="Registration",command=reg,width=15, height=1,font=('times', 15, ' bold '), bg="black", fg="white")
button2.place(x=700, y=20)

button3 = tk.Button(Entry_frame, text="Exit",command=window,width=15, height=1,font=('times', 15, ' bold '), bg="red", fg="white")
button3.place(x=1100, y=20)





root.mainloop()