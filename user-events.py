import tkinter as tk
from tkinter import Frame, Canvas, Radiobutton, StringVar, NW
from tkinter.ttk import Button
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("Segmented Image")

        self.original_image = cv2.imread('imagen_prueba/image.jpg')
        if self.original_image is None:
            raise FileNotFoundError("Image not found or unable to load.")
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Guardar el estado inicial de la imagen para la función de deshacer
        self.undo_stack = [self.original_image.copy()]

        self.color_options = Frame(master)
        self.color_options.pack(side="top")
        self.color_var = StringVar(value="red")
        colors = {"Azul (Water)": "blue", "Rojo (Urbano)": "red", "Verde (Forestal)": "green", "Amarillo (Agricultura)": "yellow"}
        for text, value in colors.items():
            Radiobutton(self.color_options, text=text, variable=self.color_var, value=value).pack(side="left")

        self.undo_button = Button(self.color_options, text="Undo", command=self.undo)
        self.undo_button.pack(side="right")

        self.image_frame = Frame(master)
        self.image_frame.pack(side="bottom")

        self.canvas = Canvas(self.image_frame, cursor="arrow", width=self.original_image.shape[1], height=self.original_image.shape[0])
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<B1-Motion>", self.handle_motion)

        self.show_segmented_image()

    def show_segmented_image(self):
        height, width, _ = self.original_image.shape
        num_rows, num_cols = 32, 32
        segment_height = height // num_rows
        segment_width = width // num_cols
        self.segments = []
        self.photos = []
        self.coords = []
        
        for i in range(num_rows):
            for j in range(num_cols):
                x0 = j * segment_width
                y0 = i * segment_height
                x1 = x0 + segment_width
                y1 = y0 + segment_height
                segment = self.original_image[y0:y1, x0:x1]
                img = Image.fromarray(segment)
                photo = ImageTk.PhotoImage(img)
                self.segments.append(segment)
                self.photos.append(photo)
                self.coords.append((x0, y0, x1, y1))
                self.canvas.create_image(x0, y0, image=photo, anchor=NW)

        # Dibujar líneas de la cuadrícula
        for i in range(1, num_rows):
            self.canvas.create_line(0, i * segment_height, width, i * segment_height, fill="green")
        for j in range(1, num_cols):
            self.canvas.create_line(j * segment_width, 0, j * segment_width, height, fill="green")

    def handle_click(self, event):
        self.undo_stack.append(self.original_image.copy())  # Guardar estado antes de modificar
        self.paint(event.x, event.y)

    def handle_motion(self, event):
        self.paint(event.x, event.y)

    def paint(self, x, y):
        color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
        chosen_color = color_map[self.color_var.get()]
        for idx, (x0, y0, x1, y1) in enumerate(self.coords):
            if x0 <= x < x1 and y0 <= y < y1:
                seed_point = (x - x0, y - y0)
                mask = np.zeros((y1 - y0 + 2, x1 - x0 + 2), np.uint8)
                cv2.floodFill(self.segments[idx], mask, seed_point, chosen_color, (10,) * 3, (10,) * 3, 8)
                img = Image.fromarray(self.segments[idx])
                photo = ImageTk.PhotoImage(img)
                self.photos[idx] = photo
                self.canvas.create_image(x0, y0, image=photo, anchor=NW)
                break  # Detener una vez que el segmento correcto es modificado

    def undo(self):
        if len(self.undo_stack) > 1:
            self.undo_stack.pop()
            last_image = self.undo_stack[-1]
            self.original_image = last_image.copy()
            self.show_segmented_image()
        else:
            print("No more actions to undo")

root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()
