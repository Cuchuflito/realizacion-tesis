import tkinter as tk
from tkinter import Frame, Canvas, Entry, Button, StringVar, Radiobutton, simpledialog, NW, Label, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Polygon
import pickle

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("Aplicación de Segmentación y Pintura de Imagen")
        master.geometry("1280x720")
        master.resizable(False, False)

        self.canvas_width = 1040
        self.canvas_height = 720
        self.scale = 1.0

        self.init_variables()
        self.setup_ui()

    def init_variables(self): #se inicializan las variables que se utilizarán (las pilas de historial y polígonos, la escala, el offset, el modo y el color)
        self.historia = []
        self.labels = []
        self.polygon_points = []
        self.is_drawing_polygon = False
        self.current_polygon = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.mode_var = StringVar(value="paint")
        self.color_var = StringVar(value="red")
        self.load_image('imagen_prueba/image.jpg')
        self.font = ImageFont.load_default()

    def load_image(self, image_path): #se carga la imagen inicial
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError("Imagen no encontrada.")
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.segmented_image = self.original_image.copy()
        self.current_image = self.segmented_image.copy()
        self.painted_image = self.segmented_image.copy()
        self.displayed_image = self.painted_image.copy()
        

    def save_state(self, file_path): #se guarda el estado de la imagen
        state = {
            'original_image': self.original_image,
            'segmented_image': self.segmented_image,
            'current_image': self.current_image,
            'painted_image': self.painted_image,
            'displayed_image': self.displayed_image,
            'labels': self.labels,
            'polygon_points': self.polygon_points,
            'is_drawing_polygon': self.is_drawing_polygon,
            'scale': self.scale,
            'offset_x': self.offset_x,
            'offset_y': self.offset_y,
            'mode_var': self.mode_var.get(),
            'color_var': self.color_var.get(),
        }
        with open(file_path, 'wb') as file:
            pickle.dump(state, file)
            
    def load_state(self, file_path):  #se carga el estado de la imagen
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
                self.is_drawing_polygon = state['is_drawing_polygon']
                self.scale = state['scale']
                self.offset_x = state['offset_x']
                self.offset_y = state['offset_y']
                self.mode_var.set(state['mode_var'])
                self.color_var.set(state['color_var'])
                self.show_segmented_image()
        except FileNotFoundError:
            print("No se encontró el archivo de estado.")

    def open_state(self): #se abre el estado de la imagen cargada en load_state
        file_path = filedialog.askopenfilename(defaultextension=".state", filetypes=[("State Files", "*.state")]) #manejo de formatos al abrir
        if file_path:
            self.load_state(file_path)

    def save_state_as(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".state", filetypes=[("State Files", "*.state")]) #manejo de formatos al guardar
        if file_path:
            self.save_state(file_path)

    def setup_ui(self): #se crea la interfaz gráfica
        self.top_frame = Frame(self.master)
        self.top_frame.pack(side="top", fill="x", expand=True)
        
        #guardar y cargar    
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)
        file_menu = tk.Menu(self.menu_bar)
        self.menu_bar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Abrir", command=self.open_state)
        file_menu.add_command(label="Guardar como", command=self.save_state_as)

        self.k_entry = Entry(self.top_frame, width=5) #setups de segmentación
        self.k_entry.pack(side="left")
        self.k_entry.insert(0, "4")
        self.kmeans_button = Button(self.top_frame, text="Segmentar", command=self.apply_kmeans)
        self.kmeans_button.pack(side="left")

        self.zoom_in_button = Button(self.top_frame, text="Zoom In", command=self.zoom_in) #setups de zoom
        self.zoom_in_button.pack(side="left")
        self.zoom_out_button = Button(self.top_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side="left")

        self.color_options = Frame(self.master, width=200) #setups opciones color
        self.color_options.pack(side="right", fill="y")
        self.undo_button = Button(self.master, text="Deshacer", command=self.undo_last_action)
        self.undo_button.pack(side="bottom", fill="y")
        colors = {"Azul (Mar)": "blue", "Rojo (Urbano)": "red", "Verde (Forestal)": "green", "Amarillo (Agricultura)": "yellow"}
        for text, value in colors.items(): #for para agregar colores a los radiobuttons
            frame = Frame(self.color_options)
            frame.pack(fill="x")
            Label(frame, width=2, bg=value).pack(side="left")
            Radiobutton(frame, text=text, variable=self.color_var, value=value).pack(side="left")
        self.mode_frame = Frame(self.top_frame) #setups de modo de click
        self.mode_frame.pack(side="left")
        Radiobutton(self.mode_frame, text="Arrastrar", variable=self.mode_var, value="drag").pack(side="left")
        Radiobutton(self.mode_frame, text="Pintar", variable=self.mode_var, value="paint").pack(side="left")
        Radiobutton(self.mode_frame, text="Lazo", variable=self.mode_var, value="lazo").pack(side="left")
        self.finish_polygon_button = Button(self.top_frame, text="Finalizar Polígono", command=self.finish_polygon)
        self.finish_polygon_button.pack(side="left")

        self.image_frame = Frame(self.master) #setups de imagen inicial
        self.image_frame.pack(side="left", fill="both", expand=True)
        self.canvas = Canvas(self.image_frame, width=1040, height=720, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.handle_click) #poscicionamiento de click	
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.reset_drag)

        
        self.show_segmented_image()
        
        
    def save_to_historia(self): #se guarda el estado de la imagen en la pila de historial
        current_state = {
            'displayed_image': self.displayed_image.copy(),
            'labels': list(self.labels), #se copian las etiquetas (labels) y los puntos del polígono
            'polygon_points': list(self.polygon_points)
            }
        self.historia.append(current_state)

    def clear_current_canvas(self):
        self.canvas.delete("all") #se limpia el canvas actual
        
    def undo_last_action(self):
        if self.historia:
            self.clear_current_canvas()  # Limpiar el canvas para evitar superposiciones visuales
            # Restaurar el último estado guardado
            state = self.historia.pop()
            self.displayed_image = np.copy(state['displayed_image'])
            self.labels = list(state['labels'])
            self.polygon_points = list(state['polygon_points'])
            self.is_drawing_polygon = state.get('is_drawing_polygon', False)
            
            # Si había un polígono siendo dibujado, eliminarlo visualmente
            if self.current_polygon:
                self.canvas.delete(self.current_polygon)
                self.current_polygon = None

            # Actualizar la visualización con el estado restaurado
            self.show_segmented_image()
        else:
            print("No hay más acciones para deshacer.")


    def apply_kmeans(self): #aplicar k-means a la imagen según el imput que diga el usuario
        self.save_to_historia()
        k = int(self.k_entry.get())
        kmeans = KMeans(n_clusters=k, random_state=0)
        data = self.original_image.reshape((-1, 3)).astype(np.float32)
        kmeans.fit(data)
        self.segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(self.original_image.shape).astype(np.uint8)
        self.current_image = self.segmented_image.copy()
        self.painted_image = self.segmented_image.copy()
        self.displayed_image = self.painted_image.copy()
        self.show_segmented_image()

    def show_segmented_image(self): #muestra la imagen segmentada por k-means
        resized_image = cv2.resize(self.displayed_image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(resized_image)
        draw = ImageDraw.Draw(img)
        draw.font = self.font

        for label, (center_x, center_y) in self.labels:
            screen_x = int(center_x * self.scale)
            screen_y = int(center_y * self.scale)

            box_width, box_height = draw.font.getbbox(label, anchor='lt')[2:]
            box_x = screen_x - box_width // 2
            box_y = screen_y - box_height // 2
            draw.rectangle([box_x - 2, box_y - 2, box_x + box_width + 2, box_y + box_height + 2], fill='black')
            draw.text((box_x, box_y), label, fill='white', font=self.font)

        self.photo_image = ImageTk.PhotoImage(image=img)
        self.canvas.config(width=self.canvas_width * self.scale, height=self.canvas_height * self.scale)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo_image, anchor=NW)
        
        
    def safe_polygon_centroid(self, points):         #Encuentra un punto seguro dentro del polígono usando shapely.
        poly = Polygon(points)
        point = poly.representative_point()  # Devuelve un punto garantizado dentro del polígono
        return point.x, point.y


    def handle_click(self, event): #manejo de clicks. Implementación de lógica entre lazo, pintar y arrastrar
        mode = self.mode_var.get()
        if mode == "lazo":
            if not self.is_drawing_polygon:
                self.polygon_points = [(event.x, event.y)]
                self.is_drawing_polygon = True
                if self.current_polygon:
                    self.canvas.delete(self.current_polygon)
                self.current_polygon = self.canvas.create_polygon(self.polygon_points, outline='red', fill='', width=2)
            else:
                self.polygon_points.append((event.x, event.y))
                self.canvas.coords(self.current_polygon, sum(self.polygon_points, ()))
        elif mode == "paint":
            color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
            chosen_color = color_map[self.color_var.get()]
            original_x = int((event.x - self.offset_x) / self.scale)
            original_y = int((event.y - self.offset_y) / self.scale)
            if 0 <= original_x < self.painted_image.shape[1] and 0 <= original_y < self.painted_image.shape[0]:
                self.paint_segment(original_x, original_y, chosen_color)
        elif mode == "drag":
            self.start_drag(event)

    def paint_segment(self, x, y, color): #aplicar floodfill a la imagen (herramienta bote de pintura)
        self.save_to_historia()
        tolerance = 20
        mask = np.zeros((self.painted_image.shape[0] + 2, self.painted_image.shape[1] + 2), np.uint8)
        cv2.floodFill(self.painted_image, mask, (x, y), color, (tolerance,) * 3, (tolerance,) * 3, flags=cv2.FLOODFILL_FIXED_RANGE)
        self.displayed_image = self.painted_image.copy()
        self.show_segmented_image()

    def start_drag(self, event): #manejo de arrastre
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag(self, event): #arrastrar la imagen
        if hasattr(self, 'drag_start_x') and self.drag_start_x is not None:
            self.offset_x += event.x - self.drag_start_x
            self.offset_y += event.y - self.drag_start_y
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.show_segmented_image()

    def reset_drag(self, event): #luego de arrastrada la imagen, se resetea el arrastre
        self.drag_start_x = None
        self.drag_start_y = None

    def zoom_in(self):
        if self.scale < 5:
            self.scale *= 1.1
            self.show_segmented_image()

    def zoom_out(self):
        if self.scale > 0.5:
            self.scale *= 0.9
            self.show_segmented_image()

    def finish_polygon(self): #finalizar el etiquetado de un polígono pintado. El nombre elegido queda en medio y se guarda en la pila de historial
        self.save_to_historia()
        if self.is_drawing_polygon and self.polygon_points:
            label = simpledialog.askstring("Etiqueta", "Introduce el nombre del sector:")
            if label:
                centroid_x, centroid_y = self.safe_polygon_centroid(self.polygon_points)
                scaled_centroid_x = int((centroid_x - self.offset_x) / self.scale)
                scaled_centroid_y = int((centroid_y - self.offset_y) / self.scale)
                self.labels.append((label, (scaled_centroid_x, scaled_centroid_y)))
                self.show_segmented_image()
                self.is_drawing_polygon = False
                self.polygon_points = []
            if self.current_polygon:
                self.canvas.delete(self.current_polygon)
                self.current_polygon = None
        else:
            print("No hay más acciones para deshacer.")


    def apply_kmeans(self): #aplicar k-means a la imagen según el imput que diga el usuario
        self.save_to_historia()
        k = int(self.k_entry.get())
        kmeans = KMeans(n_clusters=k, random_state=0)
        data = self.original_image.reshape((-1, 3)).astype(np.float32)
        kmeans.fit(data)
        self.segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(self.original_image.shape).astype(np.uint8)
        self.current_image = self.segmented_image.copy()
        self.painted_image = self.segmented_image.copy()
        self.displayed_image = self.painted_image.copy()
        self.show_segmented_image()

    def show_segmented_image(self): #muestra la imagen segmentada por k-means
        resized_image = cv2.resize(self.displayed_image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(resized_image)
        draw = ImageDraw.Draw(img)
        draw.font = self.font

        for label, (center_x, center_y) in self.labels:
            screen_x = int(center_x * self.scale)
            screen_y = int(center_y * self.scale)

            box_width, box_height = draw.font.getbbox(label, anchor='lt')[2:]
            box_x = screen_x - box_width // 2
            box_y = screen_y - box_height // 2
            draw.rectangle([box_x - 2, box_y - 2, box_x + box_width + 2, box_y + box_height + 2], fill='black')
            draw.text((box_x, box_y), label, fill='white', font=self.font)

        self.photo_image = ImageTk.PhotoImage(image=img)
        self.canvas.config(width=self.canvas_width * self.scale, height=self.canvas_height * self.scale)
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo_image, anchor=NW)
        
        
    def safe_polygon_centroid(self, points):         #Encuentra un punto seguro dentro del polígono usando shapely.
        poly = Polygon(points)
        point = poly.representative_point()  # Devuelve un punto garantizado dentro del polígono
        return point.x, point.y


    def handle_click(self, event): #manejo de clicks. Implementación de lógica entre lazo, pintar y arrastrar
        mode = self.mode_var.get()
        if mode == "lazo":
            if not self.is_drawing_polygon:
                self.polygon_points = [(event.x, event.y)]
                self.is_drawing_polygon = True
                if self.current_polygon:
                    self.canvas.delete(self.current_polygon)
                self.current_polygon = self.canvas.create_polygon(self.polygon_points, outline='red', fill='', width=2)
            else:
                self.polygon_points.append((event.x, event.y))
                self.canvas.coords(self.current_polygon, sum(self.polygon_points, ()))
        elif mode == "paint":
            color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0), "yellow": (255, 255, 0)}
            chosen_color = color_map[self.color_var.get()]
            original_x = int((event.x - self.offset_x) / self.scale)
            original_y = int((event.y - self.offset_y) / self.scale)
            if 0 <= original_x < self.painted_image.shape[1] and 0 <= original_y < self.painted_image.shape[0]:
                self.paint_segment(original_x, original_y, chosen_color)
        elif mode == "drag":
            self.start_drag(event)

    def paint_segment(self, x, y, color): #aplicar floodfill a la imagen (herramienta bote de pintura)
        self.save_to_historia()
        tolerance = 20
        mask = np.zeros((self.painted_image.shape[0] + 2, self.painted_image.shape[1] + 2), np.uint8)
        cv2.floodFill(self.painted_image, mask, (x, y), color, (tolerance,) * 3, (tolerance,) * 3, flags=cv2.FLOODFILL_FIXED_RANGE)
        self.displayed_image = self.painted_image.copy()
        self.show_segmented_image()

    def start_drag(self, event): #manejo de arrastre
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag(self, event): #arrastrar la imagen
        if hasattr(self, 'drag_start_x') and self.drag_start_x is not None:
            self.offset_x += event.x - self.drag_start_x
            self.offset_y += event.y - self.drag_start_y
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.show_segmented_image()

    def reset_drag(self, event): #luego de arrastrada la imagen, se resetea el arrastre
        self.drag_start_x = None
        self.drag_start_y = None

    def zoom_in(self):
        if self.scale < 5:
            self.scale *= 1.1
            self.show_segmented_image()

    def zoom_out(self):
        if self.scale > 0.5:
            self.scale *= 0.9
            self.show_segmented_image()

    def finish_polygon(self): #finalizar el etiquetado de un polígono pintado. El nombre elegido queda en medio y se guarda en la pila de historial
        self.save_to_historia()
        if self.is_drawing_polygon and self.polygon_points:
            label = simpledialog.askstring("Etiqueta", "Introduce el nombre del sector:")
            if label:
                centroid_x, centroid_y = self.safe_polygon_centroid(self.polygon_points)
                scaled_centroid_x = int((centroid_x - self.offset_x) / self.scale)
                scaled_centroid_y = int((centroid_y - self.offset_y) / self.scale)
                self.labels.append((label, (scaled_centroid_x, scaled_centroid_y)))
                self.show_segmented_image()
                self.is_drawing_polygon = False
                self.polygon_points = []
            if self.current_polygon:
                self.canvas.delete(self.current_polygon)
                self.current_polygon = None

root = tk.Tk()
app = ImageSegmentationApp(root)
root.mainloop()