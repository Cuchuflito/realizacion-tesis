import tkinter as tk
from tkinter import Frame, Canvas, Entry, Button, StringVar, Radiobutton, simpledialog, NW, Label, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Polygon
import pickle
from shapely.validation import make_valid
import time
from scipy import interpolate

start_time = time.perf_counter()

class creacion_de_mapas:
    def __init__(self, master):
        self.master = master
        master.title("Creación de Mapas")
        master.geometry("1280x720")
        master.resizable(False, False)

        self.canvas_width = 1040
        self.canvas_height = 720
        self.scale = 1.0

        self.init_variables()
        self.ui()

        try:
            self.font = ImageFont.truetype("arial.ttf", 12)
        except:
            self.font = ImageFont.load_default()
            
        init_time = time.perf_counter() - start_time
        print(f"Tiempo inicialización: {init_time:.6f} segundos.")
            
    def init_variables(self):
        self.historia = []
        self.labels = []
        self.polygon_points = []
        self.original_polygon_points = []
        self.is_drawing_polygon = False
        self.current_polygon = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.mode_var = StringVar(value="lazo")
        self.color_var = StringVar(value="orange")
        self.color_map = {"orange": 1, "blue": 2, "green": 3, "purple": 4, "brown": 5}
        self.existing_polygons = []
        self.font = ImageFont.load_default()

    def cargar_imagen(self):
        start_time = time.perf_counter()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                self.original_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if self.original_image is None:
                    raise ValueError("Imagen no encontrada.")
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.segmented_image = self.original_image.copy()
                self.current_image = self.segmented_image.copy()
                self.painted_image = self.segmented_image.copy()
                self.displayed_image = self.painted_image.copy()
                self.imagen_segmentada()
                print(f"Tiempo de carga de la imagen: {time.perf_counter() - start_time:.6f} segundos.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                print(f"Error al cargar la imagen: {file_path} - {e}")     

    def guardar_png(self, file_path):
        #se copia la imagen actual para no dañar la imagen original en caso de error
        img_with_labels = self.painted_image.copy()
        #si la imagen está en escala de grises, se convierte a RGB
        if len(img_with_labels.shape) == 2 or img_with_labels.shape[2] == 1:
            img_with_labels = cv2.cvtColor(img_with_labels, cv2.COLOR_GRAY2RGB)
        imagen = Image.fromarray(img_with_labels)
        draw = ImageDraw.Draw(imagen)
        draw.font = self.font
        #se redimensiona y guarda la imagen
        resized_image = cv2.resize(np.array(imagen), None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        resized_imagen = Image.fromarray(resized_image)
        resized_imagen.save(file_path, "PNG")

    def guardar_asc(self, file_path):
        # Get the dimensions of the image
        rows, cols = self.painted_image.shape[:2]
        asc_array = np.zeros((rows, cols), dtype=int)
        
        # Definir el mapa de colores en el orden correcto
        color_bgr_map = {
            "blue": [17, 131, 168],     # Mar (1)
            "orange": [238, 191, 17],   # Urbano (2)
            "green": [107, 229, 68],    # Forestal (3)
            "purple": [149, 29, 205],   # Edificio Histórico (4)
            "brown": [109, 93, 33]      # Tierra (5)
        }
        
        # Asignar valores a las celdas pintadas
        for value, (color_name, color_bgr) in enumerate(color_bgr_map.items(), start=1):
            mask = np.all(self.painted_image == color_bgr, axis=-1)
            asc_array[mask] = value
        
        # Crear una máscara de valores no cero
        mask = asc_array != 0
        
        # Obtener las coordenadas de los valores no cero
        y, x = np.where(mask)
        
        if len(x) == 0 or len(y) == 0:
            raise ValueError("No hay suficientes datos no nulos para crear el interpolador.")
        
        # Crear un interpolador con los valores no cero
        values = asc_array[mask]
        interp = interpolate.NearestNDInterpolator(list(zip(x, y)), values)
        
        # Aplicar la interpolación a toda la matriz
        xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
        asc_array = interp(xx, yy).astype(int)
        
        # Aplicar una corrección de bordes para evitar errores de píxeles
        for i in range(rows):
            for j in range(cols):
                if asc_array[i, j] == 0:
                    distances = []
                    values = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols and asc_array[ni, nj] != 0:
                                distances.append(np.sqrt(di**2 + dj**2))
                                values.append(asc_array[ni, nj])
                    if values:
                        asc_array[i, j] = values[np.argmin(distances)]
        
        header = f"ncols {cols}\nnrows {rows}\nxllcorner 0.0\nyllcorner 0.0\ncellsize 1.0\nNODATA_value -9999\n"
        np.savetxt(file_path, asc_array, fmt='%d', header=header, comments='')

    def guardar_estado(self, file_path):
        #se guarda cada estado actual en una variable distinta, con el fin de que el sistema identifique estas mismas variables al momento de abrir un estado.
        state = {
            'original_image': self.original_image,
            'segmented_image': self.segmented_image,
            'current_image': self.current_image,
            'painted_image': self.painted_image,
            'displayed_image': self.displayed_image,
            'labels': self.labels,
            'polygon_points': self.polygon_points,
            'original_polygon_points': self.original_polygon_points,
            'is_drawing_polygon': self.is_drawing_polygon,
            'scale': self.scale,
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'mode_var': self.mode_var.get(),
            'color_var': self.color_var.get(),
            'existing_polygons': self.existing_polygons
        }
        with open(file_path, 'wb') as file:
            pickle.dump(state, file)

    def cargar_estado(self, file_path):
        #se busca en el computador un archivo que tenga las variables previamente mencionadas en su poder. La razón de almacenar cada cambio por separado es poder guardar el estado de la imagen que se esté editando
        try:
            with open(file_path, 'rb') as file:
                state = pickle.load(file)
                self.original_image = state['original_image']
                self.segmented_image = state['segmented_image']
                self.current_image = state['current_image']
                self.painted_image = state['painted_image']
                self.displayed_image = state['displayed_image']
                self.labels = state['labels']
                self.polygon_points = state['polygon_points']
                self.original_polygon_points = state['original_polygon_points']
                self.is_drawing_polygon = state['is_drawing_polygon']
                self.scale = state['scale']
                self.offset_x = state['offset_x']
                self.offset_y = state['offset_y']
                self.mode_var.set(state['mode_var'])
                self.color_var.set(state['color_var'])
                self.existing_polygons = state['existing_polygons']
                self.imagen_segmentada()
        except FileNotFoundError:
            print("No se encontró el archivo de estado.")
    def abrir(self):
        file_path = filedialog.askopenfilename(defaultextension=".state", filetypes=[("State Files", "*.state")])
        if file_path:
            self.cargar_estado(file_path)
    def guardar_como(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".state", filetypes=[("State Files", "*.state")])
        if file_path:
            self.guardar_estado(file_path)

    def exportar_png(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if file_path:
            self.guardar_png(file_path)

    def exportar_asc(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".asc", filetypes=[("ASC Files", "*.asc")])
        if file_path:
            self.guardar_asc(file_path)

    def ui(self):
        # Marco superior
        self.top_frame = Frame(self.master)
        self.top_frame.pack(side="top", fill="x", expand=True)

        # Barra menú
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)
        file_menu = tk.Menu(self.menu_bar)
        self.menu_bar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Abrir", command=self.abrir)
        file_menu.add_command(label="Guardar como", command=self.guardar_como)
        file_menu.add_command(label="Exportar como PNG", command=self.exportar_png)
        file_menu.add_command(label="Exportar como ASC", command=self.exportar_asc)

        # Operaciones y herramientas en orden: cargar imagen, segmentar, zoom in, zoom out
        self.load_image_button = Button(self.top_frame, text="Cargar Imagen", command=self.cargar_imagen)
        self.load_image_button.pack(side="left")
        self.k_entry = Entry(self.top_frame, width=5)
        self.k_entry.pack(side="left")
        self.k_entry.insert(0, "4")
        self.kmeans_button = Button(self.top_frame, text="Segmentar", command=self.kmeans)
        self.kmeans_button.pack(side="left")
        self.zoom_in_button = Button(self.top_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side="left")
        self.zoom_out_button = Button(self.top_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side="left")

        # Botones de radio para cambiar de modo y color
        self.color_options = Frame(self.master, width=200)
        self.color_options.pack(side="right", fill="y")
        colors = {"Azul (Mar)": "blue", "Naranja (Urbano)": "orange", "Verde (Forestal)": "green", "Morado (Edificio Historico)": "purple", "Café (Tierra)":"brown"}
        for text, value in colors.items():
            frame = Frame(self.color_options)
            frame.pack(fill="x")
            Label(frame, width=2, bg=value).pack(side="left")
            Radiobutton(frame, text=text, variable=self.color_var, value=value).pack(side="left")
        self.mode_frame = Frame(self.top_frame)
        self.mode_frame.pack(side="left")
        Radiobutton(self.mode_frame, text="Lazo", variable=self.mode_var, value="lazo").pack(side="left")
        Radiobutton(self.mode_frame, text="Arrastrar", variable=self.mode_var, value="drag").pack(side="left")

        # Botón para finalizar polígono
        self.finish_polygon_button = Button(self.top_frame, text="Finalizar Polígono", command=self.terminar_etiquetado)
        self.finish_polygon_button.pack(side="left")

        # Configuración del canvas
        self.image_frame = Frame(self.master)
        self.image_frame.pack(side="left", fill="both", expand=True)
        self.canvas = Canvas(self.image_frame, width=1040, height=720, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.reset_drag)

    def save_to_historia(self):
        current_state = {
            'displayed_image': self.displayed_image.copy(),
            'labels': list(self.labels),
            'polygon_points': list(self.polygon_points),
            'original_polygon_points': list(self.original_polygon_points),
            'existing_polygons': list(self.existing_polygons)
        }
        self.historia.append(current_state)

    def clear_current_canvas(self):
        self.canvas.delete("all")

    def kmeans(self):
        start_time = time.perf_counter()
        self.save_to_historia()
        
        # Se crea el cuadro de diálogo para elegir la forma del cálculo de distancias.
        method_window = tk.Toplevel(self.master)
        method_window.title("Seleccione el método de segmentación")
        
        method_var = StringVar(value="euclidiana")
        # se crea el mensaje para preguntarle al usuario como quiere segmentar
        Label(method_window, text="Seleccione el método de segmentación:").pack(pady=10)
        Radiobutton(method_window, text="Según distancia Euclidiana", variable=method_var, value="euclidiana").pack(anchor="w")
        Radiobutton(method_window, text="Según distancia Manhattan", variable=method_var, value="manhattan").pack(anchor="w")
        Radiobutton(method_window, text="Según distancia Chebyshev", variable=method_var, value="chebyshev").pack(anchor="w")
        # se crea metodo para ubicar ventana selección metodo de segmentacion, y dejarla al centro de la herramienta
        def on_ok():
            method_window.destroy()
        Button(method_window, text="OK", command=on_ok).pack(pady=10)
        master_width = self.master.winfo_width()
        master_height = self.master.winfo_height()
        master_x = self.master.winfo_x()
        master_y = self.master.winfo_y()
        method_window.update_idletasks() 
        window_width = method_window.winfo_width()
        window_height = method_window.winfo_height()
        position_x = master_x + (master_width // 2) - (window_width // 2)
        position_y = master_y + (master_height // 2) - (window_height // 2)
        method_window.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")  
        method_window.transient(self.master)
        method_window.grab_set()
        self.master.wait_window(method_window)            
        method = method_var.get()
        
        # Crear una máscara para los polígonos existentes
        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        for polygon in self.existing_polygons:
            points = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        
        # Obtener clusters (colores) según la entrada del usuario
        k = int(self.k_entry.get())
        data = self.original_image[mask == 0].reshape((-1, 3)).astype(np.float32)
        #segmentar según cualquiera de las 3 opciones que escoja el usuario
        if method == "euclidiana":
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data)
            labels = kmeans.predict(data)
            centroids = kmeans.cluster_centers_
        
        elif method == "manhattan":
            centroids = data[np.random.choice(data.shape[0], k, replace=False)]
            for _ in range(100):  
                distances = np.abs(data[:, np.newaxis] - centroids).sum(axis=2)
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
                if np.all(centroids == new_centroids):
                    break
                centroids = new_centroids
        
        elif method == "chebyshev":
            centroids = data[np.random.choice(data.shape[0], k, replace=False)]
            for _ in range(100):  
                distances = np.max(np.abs(data[:, np.newaxis] - centroids), axis=2)
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
                if np.all(centroids == new_centroids):
                    break
                centroids = new_centroids

        # Crear la imagen segmentada
        self.segmented_image = self.original_image.copy()
        self.segmented_image[mask == 0] = centroids[labels].astype(np.uint8)
        
        # Combinar la imagen segmentada con los polígonos existentes
        self.current_image = self.segmented_image.copy()
        self.painted_image = self.segmented_image.copy()
        self.displayed_image = self.painted_image.copy()
        
        # Volver a dibujar los polígonos existentes
        for polygon in self.existing_polygons:
            points = np.array(polygon.exterior.coords, dtype=np.int32)
            color = self.painted_image[points[0][1], points[0][0]].tolist()  # Convertir a lista
            cv2.fillPoly(self.painted_image, [points], color)
            cv2.fillPoly(self.displayed_image, [points], color)       
        self.imagen_segmentada()
        print(f"Tiempo de ejecución de K-means ({method}): {time.perf_counter() - start_time:.6f} segundos.")

    def imagen_segmentada(self):
        # Redimensiona la imagen para poder trabajarla
        resized_image = cv2.resize(self.displayed_image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        # Se convierte la imagen a objeto PIL para poder dibujar sobre ella
        img = Image.fromarray(resized_image)
        draw = ImageDraw.Draw(img)
        draw.font = self.font
        # Comienza el dibujado de etiquetas
        for label, (center_x, center_y) in self.labels:
            if label:  
                screen_x = int(center_x * self.scale)
                screen_y = int(center_y * self.scale)
                box_width, box_height = draw.font.getbbox(label, anchor='lt')[2:]
                box_x = screen_x - box_width // 2
                box_y = screen_y - box_height // 2
                draw.rectangle([box_x - 2, box_y - 2, box_x + box_width + 2, box_y + box_height + 2], fill='black')
                draw.text((box_x, box_y), label, fill='white', font=self.font)
                # Se reconvierte la imagen nueva PIL a un PhotoImage para poder visualizar en Tkinter
        self.photo_image = ImageTk.PhotoImage(image=img)
        self.canvas.config(width=self.canvas_width * self.scale, height=self.canvas_height * self.scale)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo_image, anchor=NW)

    def centroide_poligono_lazo(self, points):
        poly = Polygon(points)
        point = poly.representative_point()
        return point.x, point.y

    def handle_click(self, event):
        mode = self.mode_var.get()
        # Variación entre lazo y arrastrar
        if mode == "lazo":
            if not self.is_drawing_polygon:
                self.polygon_points = []
                self.original_polygon_points = []
                # Si la imagen está arrastrada o posee zoom, hacer los cálculos para que el polígono se dibuje en la imagen original cargada
                scaled_x = (event.x - self.offset_x) / self.scale
                scaled_y = (event.y - self.offset_y) / self.scale
                self.polygon_points.append((event.x, event.y))
                self.original_polygon_points.append((scaled_x, scaled_y))
                self.is_drawing_polygon = True
                if self.current_polygon:
                    self.canvas.delete(self.current_polygon)
                self.current_polygon = self.canvas.create_polygon(self.polygon_points, outline='red', fill='', width=2)
            else:
                scaled_x = (event.x - self.offset_x) / self.scale
                scaled_y = (event.y - self.offset_y) / self.scale
                self.polygon_points.append((event.x, event.y))
                self.original_polygon_points.append((scaled_x, scaled_y))
                self.canvas.coords(self.current_polygon, *[coord for point in self.polygon_points for coord in point])
        # Si es arrastrar        
        elif mode == "drag":
            self.is_drawing_polygon = False
            self.start_drag(event)

    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag(self, event):
        if self.mode_var.get() == "drag" and hasattr(self, 'drag_start_x') and self.drag_start_x is not None:
            # Cálculo de desplazamiento: restar posiciones de arrastre a posición inicial de imagen
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            # Offset para ajustar posición imagen
            self.offset_x += dx
            self.offset_y += dy
            # Se actualizan coordenadas de arrastre
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            # Redibujo de imagen movida en el canvas
            self.imagen_segmentada()

    def reset_drag(self, event):
        if self.mode_var.get() == "drag":
            self.drag_start_x = None
            self.drag_start_y = None
    # Reiniciar posición de arrastre (para volver a arrastrar sin problemas de superposición de coordenadas en el canvas)

    def zoom_in(self):
        if self.scale < 5:
            # Si la escala de zoom es menor a 5, se agranda y redibuja la imagen según el zoom
            self.scale *= 1.1
            self.imagen_segmentada()
            # Si se hace zoom, se reinicia todo dibujo de polígonos en curso, para limpiar el zoom
            self.is_drawing_polygon = False
            self.polygon_points = []

    def zoom_out(self):
        if self.scale > 0.5:
            self.scale *= 0.9
            self.imagen_segmentada()
            self.is_drawing_polygon = False
            self.polygon_points = []

    def terminar_etiquetado(self):
        self.save_to_historia()
        # Verificación inicial si hay polígonos en curso o puntos de polígono definidos
        if self.is_drawing_polygon and self.original_polygon_points:
            label = simpledialog.askstring("Etiqueta", "Introduce el nombre del sector:", initialvalue="")
            if label is None:
                label = ""
            # Mapa de colores RGB seleccionables para polígonos (el sistema lo obtiene según el radiobutton que de el usuario).
            color_map = {"orange": (238, 191, 17), "blue": (17, 131, 168), "green": (107, 229, 68), "purple": (149, 29, 205), "brown": (109, 93, 33)}
            chosen_color = color_map[self.color_var.get()]
            # Se crea una máscara binaria, se convierten los puntos del polígono a un formato adecuado para poder pintar, y se rellena el polígono
            mask = np.zeros((self.painted_image.shape[0], self.painted_image.shape[1]), dtype=np.uint8)
            points = np.array(self.original_polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
            # Se aplican los colores en los estados actuales
            self.painted_image[mask == 1] = chosen_color
            self.displayed_image[mask == 1] = chosen_color
            # Si se etiqueta el polígono, se calcula su centroide para ubicar la etiqueta misma. Etiqueta se añade a lista de etiquetas
            if label:
                centroid_x, centroid_y = self.centroide_poligono_lazo(self.original_polygon_points)
                self.labels.append((label, (centroid_x, centroid_y)))
            self.existing_polygons.append(Polygon(self.original_polygon_points))
            # Se actualiza el canvas
            self.imagen_segmentada()
            self.is_drawing_polygon = False
            self.polygon_points = []
            self.original_polygon_points = []
            if self.current_polygon:
                self.canvas.delete(self.current_polygon)
                self.current_polygon = None
        else:
            print("Nada más")

root = tk.Tk()
app = creacion_de_mapas(root)
root.mainloop()

total_startup_time = time.perf_counter() - start_time
print(f"Tiempo total de uso: {total_startup_time:.6f} segundos.")