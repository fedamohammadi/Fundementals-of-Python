import tkinter as tk

# Function to handle button click
def on_button_click():
    output_label.config(text=f"Hello, {name_entry.get()}!")

# Timer function for animation (simple label moving right)
def move_label():
    global moving  # Use a global variable to track the timer ID
    x, y = label_animation.winfo_x(), label_animation.winfo_y()
    if x < 200:  # Move the label until it reaches a certain point
        label_animation.place(x=x + 5, y=y)  # Move 5 pixels to the right
        moving = root.after(50, move_label)  # Schedule the function again after 50 ms
    else:
        cancel_animation()  # Stop the animation

# Function to start the animation
def start_animation():
    move_label()

# Function to stop the animation
def cancel_animation():
    if moving is not None:
        root.after_cancel(moving)

# Create the main window
root = tk.Tk()
root.title("Tkinter GUI Sample")
root.geometry("400x300")

# Creating a Label widget
welcome_label = tk.Label(root, text="Welcome to Tkinter GUI", font=("Arial", 16))
welcome_label.pack(pady=10)  # Pack layout

# Creating an Entry widget for single value data entry
name_entry = tk.Entry(root)
name_entry.pack(pady=5)
name_entry.insert(0, "Enter your name")

# Creating a Button widget with a command event
button = tk.Button(root, text="Greet Me!", command=on_button_click)
button.pack(pady=5)

# Label to show the output message
output_label = tk.Label(root, text="", font=("Arial", 14))
output_label.pack(pady=10)

# Frame for grouping animation controls
frame = tk.Frame(root)
frame.pack(pady=10)

# Start animation button
start_button = tk.Button(frame, text="Start Animation", command=start_animation)
start_button.grid(row=0, column=0, padx=5)

# Stop animation button
stop_button = tk.Button(frame, text="Stop Animation", command=cancel_animation)
stop_button.grid(row=0, column=1, padx=5)

# Label for animation
label_animation = tk.Label(root, text="Moving Text", font=("Arial", 12), fg="blue")
label_animation.place(x=0, y=200)  # Initial position

# Focus and binding an event to the entry
name_entry.focus()
name_entry.bind("<Return>", lambda event: on_button_click())  # Enter key triggers greeting

# Initialize global variable for animation control
moving = None

# Start the Tkinter main loop
root.mainloop()
