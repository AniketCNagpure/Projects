# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:39:46 2024

@author: admin
"""
import pandas as pd
import tkinter as tk
from tkinter import ttk

def Data_Display():
    # Create the root window
    root = tk.Tk()
    root.geometry("800x600")
    root.title("Data Display")
    root.configure(background="skyblue")
    root.resizable(False, False)

    # Define columns for the treeview
    columns = ['Category', 'Message']
    print(columns)

    # Read data from CSV file
    data1 = pd.read_csv("spam.csv", encoding='unicode_escape')

    # Create labels for each column
    Category = data1.iloc[:, 0]
    Message = data1.iloc[:, 1]

    # Create a label frame for displaying the data
    display = tk.LabelFrame(root, text=" Display", width=750, height=900)
    display.pack(padx=10, pady=10)

    # Create a treeview widget
    tree = ttk.Treeview(display, columns=('Category', 'Message'))

    # Configure treeview style
    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Calibri', 10), background="black")

    # Set column widths and headings
    tree.column("#0", width=0, stretch=tk.NO)  # Hide the first empty column
    tree.column("Category", width=150, anchor=tk.CENTER)
    tree.column("Message", width=600, anchor=tk.W)
    tree.heading("Category", text="Category")
    tree.heading("Message", text="Message")

    # Insert data into the treeview
    for i in range(len(Category)):
        tree.insert("", 'end', values=(Category[i], Message[i]))

    tree.pack(expand=True, fill="both")

    root.mainloop()

# Call the function to display the data
Data_Display()