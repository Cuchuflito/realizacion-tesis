import tkinter as tk
from tkinter import Frame, Canvas, Entry, Button, StringVar, Radiobutton, simpledialog, NW, Label, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from sklearn.cluster import KMeans
from shapely.geometry import Polygon, MultiPolygon
import pickle
from shapely.validation import make_valid

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        master.title("Creación de Mapas")
        master.geometry("1280x720")
        master.resizable(False, False)

        self.canvas_width = 1040
        self.canvas_height = 720
        self.scale = 1.0

        self.init_variables()
        self.ui_usuario()

        try:
            self.font = ImageFont.truetype("arial.ttf", 12)
        except:
            self.font = ImageFont.load_default()

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
        self.color_map = {"orange": 1, "blue": 2, "green": 3, "purple": 4}
        self.existing_polygons = []
        self.font = ImageFont.load_default()

    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Imagen no encontrada.")
                return
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.segmented_image = self.original_image.copy()
            self.current_image = self.segmented_image.copy()
            self.painted_image = self.segmented_image.copy()
            self.displayed_image = self.painted_image.copy()
            self.imagen_segmentada()

    def guardar_png(self, file_path):
        img_with_labels = self.painted_image.copy()
        if len(img_with_labels.shape) == 2 or img_with_labels.shape[2] == 1:
            img_with_labels = cv2.cvtColor(img_with_labels, cv2.COLOR_GRAY2RGB)
        imagen = Image.fromarray(img_with_labels)
        draw = ImageDraw.Draw(imagen)
        draw.font = self.font
        for label, (center_x, center_y) in self.labels:
            if label:  # Solo dibujar si la etiqueta no está vacía
                label_x = int(center_x)
                label_y = int(center_y)
                box_width, box_height = draw.font.getbbox(label, anchor='lt')[2:]
                box_x = label_x - box_width // 2
                box_y = label_y - box_height // 2
                draw.rectangle([box_x - 2, box_y - 2, box_x + box_width + 2, box_y + box_height + 2], fill='black')
                draw.text((box_x, box_y), label, fill='white', font=self.font)
        resized_image = cv2.resize(np.array(imagen), None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        resized_imagen = Image.fromarray(resized_image)
        resized_imagen.save(file_path, "PNG")

    def guardar_asc(self, file_path):
        rows, cols, _ = self.painted_image.shape
        asc_array = np.zeros((rows, cols), dtype=int)

        original_mask = np.all(self.painted_image == self.original_image, axis=-1)
        asc_array[original_mask] = 0

        color_bgr_map = {
            "blue": [17, 131, 168],
            "orange": [238, 191, 17],
            "green": [107, 229, 68],
            "purple": [149, 29, 205],
            "brown": [109, 93, 33]
        }
        value = 1
        for color_name, color_bgr in color_bgr_map.items():
            mask = np.all(self.painted_image == color_bgr, axis=-1)
            asc_array[mask] = value
            value += 1

        header = f"ncols {cols}\nnrows {rows}\nxllcorner 0.0\nyllcorner 0.0\ncellsize 1.0\nNODATA_value -9999\n"
        np.savetxt(file_path, asc_array, fmt='%d', header=header, comments='')

    def guardar_estado(self, file_path):
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

    def ui_usuario(self):
        self.top_frame = Frame(self.master)
        self.top_frame.pack(side="top", fill="x", expand=True)

        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)
        file_menu = tk.Menu(self.menu_bar)
        self.menu_bar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Abrir", command=self.abrir)
        file_menu.add_command(label="Guardar como", command=self.guardar_como)
        file_menu.add_command(label="Exportar como PNG", command=self.exportar_png)
        file_menu.add_command(label="Exportar como ASC", command=self.exportar_asc)

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

        self.finish_polygon_button = Button(self.top_frame, text="Finalizar Polígono", command=self.terminar_etiquetado)
        self.finish_polygon_button.pack(side="left")

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
        self.save_to_historia()
        k = int(self.k_entry.get())
        kmeans = KMeans(n_clusters=k, random_state=0)
        data = self.original_image.reshape((-1, 3)).astype(np.float32)
        kmeans.fit(data)
        self.segmented_image = kmeans.cluster_centers_[kmeans.labels_].reshape(self.original_image.shape).astype(np.uint8)
        self.current_image = self.segmented_image.copy()
        self.painted_image = self.segmented_image.copy()
        self.displayed_image = self.painted_image.copy()
        self.is_drawing_polygon = False
        self.polygon_points = []
        self.original_polygon_points = []
        self.existing_polygons = []
        self.imagen_segmentada()

    def imagen_segmentada(self):
        resized_image = cv2.resize(self.displayed_image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(resized_image)
        draw = ImageDraw.Draw(img)
        draw.font = self.font
        for label, (center_x, center_y) in self.labels:
            if label:  # Solo dibujar si la etiqueta no está vacía
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

    def centroide_poligono_lazo(self, points):
        poly = Polygon(points)
        point = poly.representative_point()
        return point.x, point.y

    def check_polygon_overlap(self, new_polygon_points):
        new_polygon = Polygon(new_polygon_points)
        new_polygon = make_valid(new_polygon)  # Validar y corregir la geometría
        overlap_threshold = 0.6  # 20% de superposición

        for poly in self.existing_polygons:
            poly = make_valid(poly)  # Validar y corregir la geometría existente
            intersection = new_polygon.intersection(poly)
            if intersection.area / new_polygon.area > overlap_threshold:
                return True

        return False

    def handle_click(self, event):
        mode = self.mode_var.get()
        if mode == "lazo":
            if not self.is_drawing_polygon:
                self.polygon_points = []
                self.original_polygon_points = []
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
        elif mode == "drag":
            self.is_drawing_polygon = False
            self.start_drag(event)

    def start_drag(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag(self, event):
        if self.mode_var.get() == "drag" and hasattr(self, 'drag_start_x') and self.drag_start_x is not None:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.offset_x += dx
            self.offset_y += dy
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.imagen_segmentada()

    def reset_drag(self, event):
        if self.mode_var.get() == "drag":
            self.drag_start_x = None
            self.drag_start_y = None

    def zoom_in(self):
        if self.scale < 5:
            self.scale *= 1.1
            self.imagen_segmentada()
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
        if self.is_drawing_polygon and self.original_polygon_points:
            label = simpledialog.askstring("Etiqueta", "Introduce el nombre del sector:", initialvalue="")
            if label is None:
                label = ""
            color_map = {"orange": (238, 191, 17), "blue": (17, 131, 168), "green": (107, 229, 68), "purple": (149, 29, 205), "brown": (109, 93, 33)}
            chosen_color = color_map[self.color_var.get()]
            mask = np.zeros((self.painted_image.shape[0], self.painted_image.shape[1]), dtype=np.uint8)
            points = np.array(self.original_polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
            self.painted_image[mask == 1] = chosen_color
            self.displayed_image[mask == 1] = chosen_color
            if label:
                centroid_x, centroid_y = self.centroide_poligono_lazo(self.original_polygon_points)
                self.labels.append((label, (centroid_x, centroid_y)))
            self.existing_polygons.append(Polygon(self.original_polygon_points))
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
app = ImageSegmentationApp(root)
root.mainloop()
