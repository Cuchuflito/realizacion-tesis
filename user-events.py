import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("LAND USE MAP CREATION")
        self.original_image = cv2.imread('imagen_prueba/image.jpg')
        if self.original_image is None:
            raise FileNotFoundError("No se encontró la imagen.")

        # Convertir la imagen a un tipo de datos compatible
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Frame para la imagen
        self.image_frame = Frame(master)
        self.image_frame.pack()

        # Crear el canvas para mostrar la imagen
        self.canvas = Canvas(self.image_frame, cursor="arrow", width=self.original_image.shape[1],
                              height=self.original_image.shape[0])
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.zoom_segment)  # Enlazar el evento de clic izquierdo

        # Mostrar la imagen dividida en segmentos
        self.show_segmented_image()

    def show_segmented_image(self):
        # Obtener dimensiones de la imagen
        height, width, _ = self.original_image.shape

        # Número de filas y columnas de segmentos
        num_rows = 8
        num_cols = 8

        # Tamaño de cada segmento
        segment_height = height // num_rows
        segment_width = width // num_cols

        # Lista para almacenar las imágenes PhotoImage y sus coordenadas
        self.images = []
        self.coords = []

        for i in range(num_rows):
            for j in range(num_cols):
                # Calcular coordenadas del segmento
                x0 = j * segment_width
                y0 = i * segment_height
                x1 = (j + 1) * segment_width
                y1 = (i + 1) * segment_height

                # Dibujar segmento
                segment = self.original_image[y0:y1, x0:x1]
                segment = cv2.resize(segment, (segment_width, segment_height))
                img = Image.fromarray(segment)
                img = ImageTk.PhotoImage(img)
                # Agregar la imagen a la lista
                self.images.append(img)  
                # Agregar las coordenadas a la lista
                self.coords.append((x0, y0, x1, y1))  
                self.canvas.create_image(x0, y0, image=img, anchor=NW)

            # Dibujar líneas de cuadrícula horizontales
            self.canvas.create_line(0, y0, width, y0, fill="green")
            self.canvas.create_line(0, y1, width, y1, fill="green")

        # Dibujar líneas de cuadrícula verticales
        for j in range(num_cols):
            x = j * segment_width
            self.canvas.create_line(x, 0, x, height, fill="green")

    def zoom_segment(self, event):
        x, y = event.x, event.y
        # Buscar el segmento correspondiente al clic
        for i, (x0, y0, x1, y1) in enumerate(self.coords):
            if x0 <= x < x1 and y0 <= y < y1:
                # Imagen de segmento 
                segment = self.original_image[y0:y1, x0:x1]

                # Ventana para mostrar el segmento ampliado
                zoom_window = Toplevel(self.master)
                zoom_window.title("Segment")

                # Calcular el factor de ampliación
                scale_factor = 2  # Puedes ajustar este valor según tus necesidades

                # Crear un canvas para mostrar el segmento ampliado
                zoom_canvas = Canvas(zoom_window, width=segment.shape[1] * scale_factor,
                                     height=segment.shape[0] * scale_factor)
                zoom_canvas.pack()

                # Convertir la imagen a un formato compatible con Tkinter
                segment = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
                segment = Image.fromarray(segment)
                segment = segment.resize((segment.width * scale_factor, segment.height * scale_factor), 3)
                segment_image = ImageTk.PhotoImage(segment)

                # Mostrar el segmento ampliado en el canvas
                zoom_canvas.create_image(0, 0, image=segment_image, anchor=NW)

                # Mantener una referencia a la imagen para evitar que se elimine
                zoom_canvas.image = segment_image

                # Variables para la selección rectangular
                self.start_x, self.start_y = None, None
                self.end_x, self.end_y = None, None
                self.rect_id = None

                # Enlazar eventos de ratón para selección rectangular
                zoom_canvas.bind("<Button-1>", lambda event, canvas=zoom_canvas: self.start_rect(event, canvas))
                zoom_canvas.bind("<B1-Motion>", lambda event, canvas=zoom_canvas: self.draw_rect(event, canvas))
                zoom_canvas.bind("<ButtonRelease-1>", lambda event, canvas=zoom_canvas, segment=segment: self.end_rect(event, canvas, segment))

                break

    def start_rect(self, event, canvas):
        # Iniciar la selección rectangular
        self.start_x, self.start_y = event.x, event.y

    def draw_rect(self, event, canvas):
        # Dibujar el rectángulo de selección
        if self.rect_id:
            canvas.delete(self.rect_id)

        self.end_x, self.end_y = event.x, event.y
        self.rect_id = canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red")

    def end_rect(self, event, canvas, segment):
        # Finalizar la selección rectangular y agrupar los píxeles por colores
        self.end_x, self.end_y = event.x, event.y

        # Obtener la región seleccionada
        selection = segment.crop((self.start_x // 2, self.start_y // 2,
                                  self.end_x // 2, self.end_y // 2))

        # Convertir la selección a un array de NumPy
        selection_array = np.array(selection)

        # Agrupar los píxeles por colores similares
        unique_colors, counts = np.unique(selection_array.reshape(-1, 3), axis=0, return_counts=True)

        # Mostrar los colores agrupados en una nueva ventana
        group_window = Toplevel(self.master)
        group_window.title("Colores agrupados")

        for color, count in zip(unique_colors, counts):
            color_label = Label(group_window, text=f"Color: {color}, Cantidad: {count}", fg="#%02x%02x%02x" % tuple(color))
            color_label.pack()

root = Tk()
app = ImageSegmentationApp(root)
root.mainloop()