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
        self.zoom_level = 1
        self.displayed_image = self.original_image.copy()

        self.color_options = Frame(master)
        self.color_options.pack(side="top")
        self.color_var = StringVar(value="red")
        colors = {"Azul (Water)": "blue", "Rojo (Urbano)": "red", "Verde (Forestal)": "green", "Amarillo (Agricultura)": "yellow"}
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
        num_rows, num_cols = 8, 8
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

    def handle_click(self, event):
        self.paint(event.x, event.y)

    def paint(self, x, y):
        color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
        chosen_color = color_map[self.color_var.get()]
        loDiff = (10, 10, 10)  # Definir la tolerancia hacia colores más oscuros
        upDiff = (10, 10, 10)  # Definir la tolerancia hacia colores más claros

        for idx, (x0, y0, x1, y1) in enumerate(self.coords):
            # Calcular las coordenadas en relación al nivel de zoom actual
            zoom_x0, zoom_y0 = x0 * self.zoom_level, y0 * self.zoom_level
            zoom_x1, zoom_y1 = x1 * self.zoom_level, y1 * self.zoom_level
            if zoom_x0 <= x < zoom_x1 and zoom_y0 <= y < zoom_y1:
                # Calcular el punto de semilla en relación al segmento ampliado
                seed_x = int((x - zoom_x0))
                seed_y = int((y - zoom_y0))
                seed_point = (seed_x, seed_y)

                # Preparar la máscara para el floodFill
                mask = np.zeros((int(zoom_y1 - zoom_y0 + 2), int(zoom_x1 - zoom_x0 + 2)), dtype=np.uint8)

                # Extraer el segmento correspondiente ampliado
                segment = self.displayed_image[zoom_y0:zoom_y1, zoom_x0:zoom_x1]

                # Aplicar floodFill al segmento
                cv2.floodFill(segment, mask, seed_point, chosen_color, loDiff, upDiff, cv2.FLOODFILL_FIXED_RANGE)

                # Actualizar el segmento en la imagen y en el canvas
                self.displayed_image[zoom_y0:zoom_y1, zoom_x0:zoom_x1] = segment
                img = Image.fromarray(segment)
                photo = ImageTk.PhotoImage(img)
                self.photos[idx] = photo
                self.canvas.create_image(zoom_x0, zoom_y0, image=photo, anchor=NW)
                break


    def undo(self):
        print("Undo functionality not implemented yet")

    def zoom_in(self):
        self.zoom_level *= 2
        self.apply_zoom()

    def zoom_out(self):
        if self.zoom_level > 1:
            self.zoom_level //= 2
            self.apply_zoom()

    def apply_zoom(self):
        self.displayed_image = cv2.resize(self.original_image, None, fx=self.zoom_level, fy=self.zoom_level, interpolation=cv2.INTER_LINEAR)
        self.display_segments()

root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()
