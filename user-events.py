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
            raise FileNotFoundError("Imagen no encontrada.")
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.current_image = self.original_image.copy()  # almacenar cambios de imagen original
        self.zoom_level = 1
        self.displayed_image = self.original_image.copy()

        self.color_options = Frame(master)
        self.color_options.pack(side="top")
        self.color_var = StringVar(value="red")
        colors = {"Azul (Mar)": "blue", "Rojo (Urbano)": "red", "Verde (Forestal)": "green", "Amarillo (Agricultura)": "yellow"}
        for text, value in colors.items():
            Radiobutton(self.color_options, text=text, variable=self.color_var, value=value).pack(side="left")

        self.undo_button = Button(self.color_options, text="Undo", command=self.undo)
        self.undo_button.pack(side="right")

        self.zoom_button = Button(self.color_options, text="Zoom In", command=self.zoom_in)
        self.zoom_button.pack(side="left")
        self.unzoom_button = Button(self.color_options, text="Zoom Out", command=self.zoom_out)
        self.unzoom_button.pack(side="left")

        self.image_frame = Frame(master)
        self.image_frame.pack(side="bottom")

        self.canvas = Canvas(self.image_frame, cursor="arrow")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.show_segmented_image()

    def show_segmented_image(self):
        self.canvas.config(width=self.displayed_image.shape[1], height=self.displayed_image.shape[0])
        self.display_segments()

    def display_segments(self):
        num_rows, num_cols = 64, 64
        segment_height = self.displayed_image.shape[0] // num_rows
        segment_width = self.displayed_image.shape[1] // num_cols
        self.segments = []
        self.photos = []
        self.coords = []

        for i in range(num_rows):
            for j in range(num_cols):
                x0 = j * segment_width
                y0 = i * segment_height
                x1 = x0 + segment_width
                y1 = y0 + segment_height
                segment = self.displayed_image[y0:y1, x0:x1]
                img = Image.fromarray(segment)
                photo = ImageTk.PhotoImage(img)
                self.segments.append(segment)
                self.photos.append(photo)
                self.coords.append((x0, y0, x1, y1))
                self.canvas.create_image(x0, y0, image=photo, anchor=NW)

    
        for i in range(1, num_rows):
            self.canvas.create_line(0, i * segment_height, self.displayed_image.shape[1], i * segment_height, fill="green")
        for j in range(1, num_cols):
            self.canvas.create_line(j * segment_width, 0, j * segment_width, self.displayed_image.shape[0], fill="green")  #lineas separadoras de segmentos

    def handle_click(self, event):
        self.paint(event.x, event.y)

    def update_display_image(self):
        new_width = int(self.current_image.shape[1] * self.zoom_level)
        new_height = int(self.current_image.shape[0] * self.zoom_level)
        self.displayed_image = cv2.resize(self.current_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.show_segmented_image()

    def paint(self, x, y):
        color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
        chosen_color = color_map[self.color_var.get()]
        for idx, (x0, y0, x1, y1) in enumerate(self.coords):
            if x0 <= x < x1 and y0 <= y < y1:
                zoomed_x0 = x0 * self.zoom_level
                zoomed_y0 = y0 * self.zoom_level
                zoomed_x1 = x1 * self.zoom_level
                zoomed_y1 = y1 * self.zoom_level
                seed_x = int((x - x0) * self.zoom_level)
                seed_y = int((y - y0) * self.zoom_level)
                seed_point = (seed_x, seed_y) #coordenadas nuevas

                # crear la mascara para el zoom
                mask_height = zoomed_y1 - zoomed_y0 + 2
                mask_width = zoomed_x1 - zoomed_x0 + 2  # PROPIEDAD IMPORTANTE FLOODFILL
                mask = np.zeros((mask_height, mask_width), dtype=np.uint8)  # Creacion de la mascara para floodfill

                segment = self.current_image[zoomed_y0:zoomed_y1, zoomed_x0:zoomed_x1]
                cv2.floodFill(segment, mask, seed_point, chosen_color, (10,) * 3, (10,) * 3, cv2.FLOODFILL_FIXED_RANGE)
                self.current_image[zoomed_y0:zoomed_y1, zoomed_x0:zoomed_x1] = segment
                self.update_display_image()
                break




    def undo(self):
        print("Acá irá el control z")

    def zoom_in(self):
        if self.zoom_level < 4:
            self.zoom_level *= 2
            self.apply_zoom()

    def zoom_out(self):
        if self.zoom_level > 1:
            self.zoom_level //= 2
            self.apply_zoom()

    def apply_zoom(self):
        new_width = int(self.current_image.shape[1] * self.zoom_level)
        new_height = int(self.current_image.shape[0] * self.zoom_level)
        self.displayed_image = cv2.resize(self.current_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.show_segmented_image()

root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()
