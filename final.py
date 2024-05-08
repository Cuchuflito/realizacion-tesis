import tkinter as tk
from tkinter import Frame, Canvas, Entry, Button, StringVar, Radiobutton, NW
from PIL import Image, ImageTk
import cv2
import numpy as np
from sklearn.cluster import KMeans

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("Aplicación de Segmentación y Pintura de Imagen")

        self.original_image = cv2.imread('imagen_prueba/image.jpg')
        if self.original_image is None:
            raise FileNotFoundError("Imagen no encontrada.")
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.segmented_image = self.original_image.copy()
        self.current_image = self.original_image.copy()  # para pintar sobre ella
        self.displayed_image = self.original_image.copy()

        self.scale = 1.0  # Escala inicial de la imagen

        self.color_options = Frame(master)
        self.color_options.pack(side="top")
        self.color_var = StringVar(value="red")
        colors = {"Azul (Mar)": "blue", "Rojo (Urbano)": "red", "Verde (Forestal)": "green", "Amarillo (Agricultura)": "yellow"}
        for text, value in colors.items():
            Radiobutton(self.color_options, text=text, variable=self.color_var, value=value).pack(side="left")

        self.k_entry = Entry(self.color_options, width=5)
        self.k_entry.pack(side="left")
        self.k_entry.insert(0, "4")  # valor predeterminado para k
        self.kmeans_button = Button(self.color_options, text="Segmentar", command=self.apply_kmeans)
        self.kmeans_button.pack(side="left")

        # Botones de zoom
        self.zoom_in_button = Button(self.color_options, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side="left")
        self.zoom_out_button = Button(self.color_options, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side="left")

        self.image_frame = Frame(master)
        self.image_frame.pack(side="bottom")

        self.canvas = Canvas(self.image_frame, cursor="cross")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.show_segmented_image()

    def apply_kmeans(self):
        k = int(self.k_entry.get())
        kmeans = KMeans(n_clusters=k, random_state=0)
        data = self.original_image.reshape((-1, 3)).astype(np.float32)
        kmeans.fit(data)
        self.segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(self.original_image.shape).astype(np.uint8)
        self.current_image = self.segmented_image.copy()
        self.displayed_image = self.segmented_image.copy()
        self.show_segmented_image()

    def show_segmented_image(self):
        resized_image = cv2.resize(self.displayed_image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(resized_image)
        self.photo_image = ImageTk.PhotoImage(image=img)
        self.canvas.config(width=resized_image.shape[1], height=resized_image.shape[0])
        self.canvas.create_image(0, 0, image=self.photo_image, anchor=NW)

    def handle_click(self, event):
        color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
        chosen_color = color_map[self.color_var.get()]
        x, y = int(event.x / self.scale), int(event.y / self.scale)  # Ajustar las coordenadas a la escala
        self.paint_segment(x, y, chosen_color)

    def paint_segment(self, x, y, color):
        mask = np.zeros((self.current_image.shape[0] + 2, self.current_image.shape[1] + 2), np.uint8)
        cv2.floodFill(self.current_image, mask, (x, y), color, (1,) * 3, (1,) * 3, flags=cv2.FLOODFILL_FIXED_RANGE)
        self.displayed_image = self.current_image.copy()
        self.show_segmented_image()

    def zoom_in(self):
        self.scale *= 1.25  # Incrementa el nivel de zoom
        self.show_segmented_image()

    def zoom_out(self):
        self.scale = max(0.25, self.scale / 1.25)  # Disminuye el nivel de zoom
        self.show_segmented_image()

root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()
