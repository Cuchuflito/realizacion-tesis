from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("Creación de Mapas de Uso de Suelo")

        try:
            # Cargar la imagen
            self.original_image = cv2.imread('imagen_prueba/image.jpg')
            if self.original_image is None:
                raise FileNotFoundError("No se encontró la imagen o no se pudo cargar.")

        # Convertir la imagen a un tipo de datos compatible
            self.original_image = cv2.convertScaleAbs(self.original_image)

            self.segmented_image = self.original_image.copy()
        except (IOError, cv2.error, FileNotFoundError) as e:
            print(f"Error al cargar la imagen: {e}")
        # Usar una imagen predeterminada o continuar con valores predeterminados
            self.original_image = None
            self.segmented_image = None

        # Frame para la imagen
        self.image_frame = Frame(master)
        self.image_frame.pack()

        # Crear el canvas para mostrar la imagen
        self.canvas = Canvas(self.image_frame, width=self.original_image.shape[1], height=self.original_image.shape[0], cursor="fleur")
        self.canvas.pack()

        # Frame para los botones
        self.button_frame = Frame(master)
        self.button_frame.pack()

        # Campo de entrada para el número de centroides
        self.k_entry_label = Label(self.button_frame, text="Número de centroides (k):")
        self.k_entry_label.pack(side=LEFT)
        self.k_entry = Entry(self.button_frame)
        self.k_entry.pack(side=LEFT)
        self.k_entry.insert(0, "4")  # Valor predeterminado para k

        # Botón para aplicar K-Means clustering
        self.kmeans_button = Button(self.button_frame, text="K-Means Segmentation", command=self.apply_kmeans_segmentation)
        self.kmeans_button.pack(side=LEFT)

        # Botones para etiquetar áreas
        self.categories = ['Urbano', 'Agrícola', 'Forestal', 'Otros']
        for category in self.categories:
            button = Button(self.button_frame, text=category, command=lambda cat=category: self.set_current_category(cat))
            button.pack(side=LEFT)

        # Bind eventos del mouse para seleccionar áreas
        self.canvas.bind('<Button-1>', self.start_selection)
        self.canvas.bind('<B1-Motion>', self.update_selection)
        self.canvas.bind('<ButtonRelease-1>', self.end_selection)

        # Inicializar variables de selección
        self.selection_start = None
        self.selection_end = None
        self.selections = []
        self.current_category = None
        self.zoom_factor = 1.0

        # Mostrar la imagen inicial
        self.update_image()

    def start_selection(self, event):
        self.selection_start = (event.x, event.y)

    def update_selection(self, event):
        self.selection_end = (event.x, event.y)
        self.draw_selection_rectangle()

    def end_selection(self, event):
        self.selection_end = (event.x, event.y)
        self.draw_selection_rectangle()
        self.selections.append((self.selection_start, self.selection_end, self.current_category))
        self.apply_kmeans_to_selection()
        self.selection_start = None
        self.selection_end = None
        self.draw_selection_rectangle()

    def draw_selection_rectangle(self):
        self.canvas.delete('selection')
        if self.selection_start and self.selection_end:
            x0, y0 = self.selection_start
            x1, y1 = self.selection_end
            self.canvas.create_rectangle(x0, y0, x1, y1, outline='red', tags='selection')

    def set_current_category(self, category):
        self.current_category = category

    def apply_kmeans_to_selection(self):
        if self.selection_start is None or self.selection_end is None:
            return

        x0, y0 = self.selection_start
        x1, y1 = self.selection_end

        # Recortar la región seleccionada de la imagen original
        selected_region = self.original_image[y0:y1, x0:x1]

        # Normalizar la imagen
        selected_region = selected_region.astype(np.float32) / 255.0

        # Obtener el número de centroides desde el campo de entrada
        k = int(self.k_entry.get())

        # Aplicar K-Means clustering a la región seleccionada
        kmeans = KMeans(n_clusters=k, init='k-means++')
        pixels = selected_region.reshape((-1, 3))
        kmeans.fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Asignar colores a los clusters
        segmented_region = centers[labels].reshape(selected_region.shape)

        # Escalar nuevamente los valores a 0-255
        segmented_region = (segmented_region * 255).astype(np.uint8)

        # Reemplazar la región seleccionada con la región segmentada en la imagen original
        self.segmented_image[y0:y1, x0:x1] = segmented_region

        # Actualizar la imagen mostrada en la aplicación
        self.update_image()

        if self.selection_start is None or self.selection_end is None:
            return

        x0, y0 = self.selection_start
        x1, y1 = self.selection_end

        # Recortar la región seleccionada de la imagen original
        selected_region = self.original_image[y0:y1, x0:x1]

        # Obtener el número de centroides desde el campo de entrada
        k = int(self.k_entry.get())

        # Aplicar K-Means clustering a la región seleccionada
        kmeans = KMeans(n_clusters=k)
        pixels = selected_region.reshape((-1, 3))
        kmeans.fit(pixels)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Asignar colores a los clusters
        segmented_region = centers[labels].reshape(selected_region.shape)

        # Reemplazar la región seleccionada con la región segmentada en la imagen original
        self.segmented_image[y0:y1, x0:x1] = segmented_region

        # Actualizar la imagen mostrada en la aplicación
        self.update_image()

    def apply_kmeans_segmentation(self):
        # No se aplicará K-Means en esta versión
        pass

    def update_image(self):
        self.segmented_image = cv2.convertScaleAbs(self.segmented_image)
        img = cv2.cvtColor(self.segmented_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((int(img.width * self.zoom_factor), int(img.height * self.zoom_factor)), 3)
        img = ImageTk.PhotoImage(img)
        self.canvas.delete("img")
        self.canvas.create_image(self.img_x, self.img_y, image=img, anchor=NW, tags="img")
        self.canvas.image = img

root = Tk()
app = ImageSegmentationApp(root)
root.mainloop()
