"""Module responsible for creating the GUI and handling the ehooke module"""

import sys
import matplotlib

if sys.platform == "darwin":
    matplotlib.use("MACOSX")
elif sys.platform == "linux":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")

from tkinter import messagebox as tkMessageBox
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cellprocessing as cp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from ehooke import EHooke
from skimage.segmentation import mark_boundaries
from skimage.exposure import rescale_intensity
from skimage.util import img_as_uint, img_as_float
from skimage.color import gray2rgb


class Interface(object):
    """Main class of the module. Used to create the GUI"""

    def __init__(self):
        self.ehooke = EHooke()
        self.default_params = self.ehooke.parameters
        self.current_step = None

        self.images = {}
        self.current_image = None

        self.base_min = 0.0
        self.base_max = 1.0
        self.fluor_min = 0.0
        self.fluor_max = 1.0
        self.optional_min = 0.0
        self.optional_max = 1.0

        self.cid = None
        self.event_connected = False

        self.dark_mode = True

        self.gui_font = ("Verdana", 9, "normal")
        self.gui_font_bold = ("Verdana", 9, "bold")

        self.pady = (9, 3)
        self.parameters_panel_width = 32

        self.image_buttons_width = 20
        self.status_bar_width = 40
        self.status_length = 200

        self.main_window = tk.Tk()
        self.main_window.wm_title("eHooke")

        self.top_frame = tk.Frame(self.main_window, height=10)
        self.top_frame.pack(fill="x")

        self.middle_frame = tk.Frame(self.main_window)
        self.middle_frame.pack(fill="x")

        self.parameters_panel = tk.Frame(self.middle_frame, width=600)
        self.parameters_panel.pack(side="left", fill="y")

        self.figure_frame = tk.Frame(self.middle_frame)
        self.figure_frame.pack(side="left")

        self.right_frame = tk.Frame(self.middle_frame)
        self.right_frame.pack(side="right", fill="y")

        self.intensities_frame = tk.Frame(self.right_frame)
        self.intensities_frame.pack(side="top")
        self.intensities_frame.config(pady=5)

        self.images_frame = tk.Frame(self.right_frame)
        self.images_frame.pack(side="top", fill="y")

        self.current_image_label = tk.Label(self.images_frame, text="")
        self.current_image_label.pack(side="top")

        self.fig = plt.figure(figsize=self.calculate_fisize(), frameon=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.middle_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top")

        self.ax = plt.subplot(111)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
        self.ax.axis("off")
        plt.autoscale(False)

        self.canvas.draw()

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.middle_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side="top")

        self.min_label = tk.Label(self.intensities_frame, text="Min Intensity: ")
        self.min_label.pack(side="top")

        self.min_scale = tk.Scale(self.intensities_frame, from_=0, to=100, tickinterval=0,
                                  length=150, orient="horizontal", width=10, showvalue=0)
        self.min_scale.bind("<ButtonRelease-1>", self.adjust_min)
        self.min_scale.pack(side="top")

        self.max_label = tk.Label(self.intensities_frame, text="Max Intensity: ")
        self.max_label.pack(side="top")

        self.max_scale = tk.Scale(self.intensities_frame, from_=0, to=100, tickinterval=0,
                                  length=150, orient="horizontal", width=10, showvalue=0)
        self.max_scale.bind("<ButtonRelease-1>", self.adjust_max)
        self.max_scale.pack(side="top")
        self.max_scale.set(100)

        self.status = tk.StringVar()
        self.status.set("Load Base Image")
        self.status_bar = tk.Label(
            self.parameters_panel, textvariable=self.status, wraplength=self.status_length)
        self.status_bar.pack(side="bottom")

        self.main_window.bind("m", self.m_shortcut)
        self.main_window.bind("s", self.s_shortcut)
        self.main_window.bind("n", self.n_shortcut)
        self.main_window.bind("u", self.u_shortcut)
        self.main_window.bind("l", self.l_shortcut)
        self.main_window.bind("k", self.k_shortcut)

        self.set_imageloader()

    def m_shortcut(self, event=None):
        if self.current_step == "CellsComputed":
            self.force_merge()
        else:
            print("Shortcut inactive on this step")

    def s_shortcut(self, event=None):
        if self.current_step == "CellsComputed":
            self.split_cell()
        else:
            print("Shortcut inactive on this step")

    def n_shortcut(self, event=None):
        if self.current_step == "CellsComputed":
            self.declare_as_noise()
        else:
            print("Shortcut inactive on this step")

    def u_shortcut(self, event=None):
        if self.current_step == "CellsComputed":
            self.undo_as_noise()
        else:
            print("Shortcut inactive on this step")

    def l_shortcut(self, event=None):
        if self.current_step == "CellsProcessed":
            self.add_line_linescan()
        else:
            print("Shortcut inactive on this step")

    def k_shortcut(self, event=None):
        if self.current_step == "CellsProcessed":
            self.remove_line_linescan()
        else:
            print("Shortcut inactive on this step")

    def adjust_min(self, event):
        current_min = float(self.min_scale.get())

        if self.current_image is None:
            pass
        elif self.current_image == "Base" or self.current_image == "Base_mask" or self.current_image == "Base_features" or self.current_image == "Base_cells_outlined":
            self.base_min = current_min / 100.0
            self.show_image(self.current_image)
        elif self.current_image == "Fluor" or self.current_image == "Fluor_mask" or self.current_image == "Fluor_features" or self.current_image == "Fluor_cells_outlined" or self.current_image == "Fluor_with_lines":
            self.fluor_min = current_min / 100.0
            self.show_image(self.current_image)
        elif self.current_image == "Optional" or self.current_image == "Optional_cells_outlined":
            self.optional_min = current_min / 100.0
            self.show_image(self.current_image)

    def adjust_max(self, event):
        current_max = float(self.max_scale.get())

        if self.current_image is None:
            pass
        elif self.current_image == "Base" or self.current_image == "Base_mask" or self.current_image == "Base_features" or self.current_image == "Base_cells_outlined":
            self.base_max = current_max / 100.0
            self.show_image(self.current_image)
        elif self.current_image == "Fluor" or self.current_image == "Fluor_mask" or self.current_image == "Fluor_features" or self.current_image == "Fluor_cells_outlined" or self.current_image == "Fluor_with_lines":
            self.fluor_max = current_max / 100.0
            self.show_image(self.current_image)
        elif self.current_image == "Optional" or self.current_image == "Optional_cells_outlined":
            self.optional_max = current_max / 100.0
            self.show_image(self.current_image)

    def calculate_fisize(self):

        root = self.main_window

        h_inch = root.winfo_screenmmheight() * 0.0393701

        return ((h_inch - h_inch * 0.2) * 1.5, h_inch - h_inch * 0.2)

    def remove_coord(self, x, y):
        """"Hack" to remove the mpl coordinates"""
        return ""

    def load_parameters(self):
        """Loads a .cfg with the parameters and sets them as the default
        params"""
        self.ehooke.parameters.load_parameters()
        self.default_params = self.ehooke.parameters
        self.load_default_params_imgloader()

    def save_parameters(self):
        """Saves the current parameters in a .cfg file"""
        self.ehooke.parameters.save_parameters()

    def load_default_params_imgloader(self):
        """Loads the default params for the image loading"""
        self.mask_algorithm_value.set(
            self.default_params.imageloaderparams.mask_algorithm)
        self.border_value.set(self.default_params.imageloaderparams.border)
        self.auto_align_value.set(
            self.default_params.imageloaderparams.auto_align)
        self.x_align_value.set(self.default_params.imageloaderparams.x_align)
        self.y_align_value.set(self.default_params.imageloaderparams.y_align)
        self.fluor_as_base_value.set(
            self.default_params.imageloaderparams.invert_base)
        self.mask_blocksize_value.set(
            self.default_params.imageloaderparams.mask_blocksize)
        self.mask_offset_value.set(
            self.default_params.imageloaderparams.mask_offset)
        self.mask_fillholes_value.set(
            self.default_params.imageloaderparams.mask_fill_holes)
        self.mask_closing_value.set(
            self.default_params.imageloaderparams.mask_closing)
        self.mask_dilation_value.set(
            self.default_params.imageloaderparams.mask_dilation)
        self.pixel_size_value.set(self.default_params.imageloaderparams.pixel_size)
        self.units_value.set(self.default_params.imageloaderparams.units)

    def load_default_params_segments(self):
        """Loads the default params for the segments computation"""
        self.peak_min_distance_edge_value.set(
            self.default_params.imageprocessingparams.peak_min_distance_from_edge)
        self.peak_min_distance_value.set(
            self.default_params.imageprocessingparams.peak_min_distance)
        self.peak_min_height_value.set(
            self.default_params.imageprocessingparams.peak_min_height)
        self.max_peaks_value.set(
            self.default_params.imageprocessingparams.max_peaks)
        self.use_base_mask_value.set(
            self.default_params.imageprocessingparams.outline_use_base_mask)

    def load_default_params_cell_computation(self):
        """Loads the default params for the cell computation"""

        self.axial_step_value.set(
            self.default_params.cellprocessingparams.axial_step)
        self.force_merge_below_value.set(
            self.default_params.cellprocessingparams.cell_force_merge_below)
        self.merge_dividing_value.set(
            self.default_params.cellprocessingparams.merge_dividing_cells)
        self.merge_length_tolerance_value.set(
            self.default_params.cellprocessingparams.merge_length_tolerance)
        self.merge_min_interface_value.set(
            self.default_params.cellprocessingparams.merge_min_interface)

    def load_default_params_cell_processing(self):
        """Loads the default params for cell processing"""
        if self.default_params.cellprocessingparams.find_septum:
            self.find_septum_menu_value.set("Closed")
        elif self.default_params.cellprocessingparams.find_openseptum:
            self.find_septum_menu_value.set("Closed+Open")
        else:
            self.find_septum_menu_value.set("No")

        if self.default_params.cellprocessingparams.look_for_septum_in_base:
            self.look_for_septum_in_menu_value.set("Base")
        elif self.default_params.cellprocessingparams.look_for_septum_in_optional:
            self.look_for_septum_in_menu_value.set("Secondary")
        else:
            self.look_for_septum_in_menu_value.set("Fluorescence")

        self.optional_signal_ratio_value.set(
            self.default_params.cellprocessingparams.signal_ratio)
        self.classify_cells_checkbox_value.set(
            self.default_params.cellprocessingparams.classify_cells
        )
        self.microscope_value.set(self.default_params.cellprocessingparams.microscope)
        self.membrane_thickness_value.set(
            self.default_params.cellprocessingparams.inner_mask_thickness)

        self.areafilter_checkbox_value.set(False)
        self.areafilter_min_value.set(0)
        self.areafilter_max_value.set(1000)

        self.perimeterfilter_checkbox_value.set(False)
        self.perimeterfilter_min_value.set(0)
        self.perimeterfilter_max_value.set(500)

        self.eccentricityfilter_checkbox_value.set(False)
        self.eccentricityfilter_min_value.set(-10)
        self.eccentricityfilter_max_value.set(10)

        self.irregularityfilter_checkbox_value.set(False)
        self.irregularityfilter_min_value.set(0)
        self.irregularityfilter_max_value.set(20)

        self.neighboursfilter_checkbox_value.set(False)
        self.neighboursfilter_min_value.set(0)
        self.neighboursfilter_max_value.set(10)

        for filter in self.default_params.cellprocessingparams.cell_filters:
            name = filter[0]
            min = filter[1]
            max = filter[2]

            if name == "Area":
                self.areafilter_checkbox_value.set(True)
                self.areafilter_min_value.set(min)
                self.areafilter_max_value.set(max)
            elif name == "Perimeter":
                self.perimeterfilter_checkbox_value.set(True)
                self.perimeterfilter_min_value.set(min)
                self.perimeterfilter_max_value.set(max)
            elif name == "Eccentricity":
                self.eccentricityfilter_checkbox_value.set(True)
                self.eccentricityfilter_min_value.set(min)
                self.eccentricityfilter_max_value.set(max)
            elif name == "Irregularity":
                self.irregularityfilter_checkbox_value.set(True)
                self.irregularityfilter_max_value.set(max)
                self.irregularityfilter_min_value.set(min)
            elif name == "Neighbours":
                self.neighboursfilter_checkbox_value.set(True)
                self.neighboursfilter_min_value.set(min)
                self.neighboursfilter_max_value.set(max)

    def show_image(self, image):
        """Method use to display the selected image on the canvas"""

        if self.current_image is None:
            self.ax.cla()
            self.ax.axis("off")

        else:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            self.ax.cla()
            self.ax.axis("off")

            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.current_image = image

        if image == "Base":
            img = rescale_intensity(self.images[image], in_range=(self.base_min, self.base_max))
            self.min_scale.set(int(self.base_min * 100))
            self.max_scale.set(int(self.base_max * 100))
            self.current_image_label.configure(text="Base")
        elif image == "Base_mask":
            img = rescale_intensity(self.images["Base"], in_range=(self.base_min, self.base_max))
            img = mark_boundaries(img, img_as_uint(self.images["Mask"]), color=(0, 1, 1), outline_color=None)
            self.min_scale.set(int(self.base_min * 100))
            self.max_scale.set(int(self.base_max * 100))
            self.current_image_label.configure(text="Base with mask")
        elif image == "Base_features":
            img = rescale_intensity(self.images["Base"], in_range=(self.base_min, self.base_max))
            places = self.ehooke.segments_manager.features > 0.5
            img[places] = 1
            self.min_scale.set(int(self.base_min * 100))
            self.max_scale.set(int(self.base_max * 100))
            self.current_image_label.configure(text="Base with Features")
        elif image == "Base_cells_outlined":
            img = rescale_intensity(self.images["Base"], in_range=(self.base_min, self.base_max))
            img = cp.overlay_cells(self.ehooke.cell_manager.cells, img, self.ehooke.cell_manager.cell_colors)
            self.min_scale.set(int(self.base_min * 100))
            self.max_scale.set(int(self.base_max * 100))
            self.current_image_label.configure(text="Base Outlined")

        elif image == "Mask":
            img = self.images[image]
            self.current_image_label.configure(text="Mask")

        elif image == "Fluor":
            img = rescale_intensity(self.images[image], in_range=(self.fluor_min, self.fluor_max))
            self.min_scale.set(int(self.fluor_min * 100))
            self.max_scale.set(int(self.fluor_max * 100))
            self.current_image_label.configure(text="Fluorescence")
        elif image == "Fluor_mask":
            img = rescale_intensity(self.images["Fluor"], in_range=(self.fluor_min, self.fluor_max))
            img = mark_boundaries(img, img_as_uint(self.images["Mask"]), color=(0, 1, 1), outline_color=None)
            self.min_scale.set(int(self.fluor_min * 100))
            self.max_scale.set(int(self.fluor_max * 100))
            self.current_image_label.configure(text="Fluor with Mask")
        elif image == "Fluor_features":
            img = rescale_intensity(self.images["Fluor"], in_range=(self.fluor_min, self.fluor_max))
            places = self.ehooke.segments_manager.features > 0.5
            img[places] = 0
            self.min_scale.set(int(self.fluor_min * 100))
            self.max_scale.set(int(self.fluor_max * 100))
            self.current_image_label.configure(text="Fluor with Features")
        elif image == "Fluor_cells_outlined":
            img = rescale_intensity(self.images["Fluor"], in_range=(self.fluor_min, self.fluor_max))
            img = cp.overlay_cells(self.ehooke.cell_manager.cells, img, self.ehooke.cell_manager.cell_colors)
            self.min_scale.set(int(self.fluor_min * 100))
            self.max_scale.set(int(self.fluor_max * 100))
            self.current_image_label.configure(text="Fluor Outlined")
        elif image == "Fluor_with_lines":
            color = (0, 1, 1)
            img = gray2rgb(rescale_intensity(self.images["Fluor"], in_range=(self.fluor_min, self.fluor_max)))
            for key in self.ehooke.linescan_manager.lines.keys():
                ln = self.ehooke.linescan_manager.lines[key]
                for i in range(len(ln.line_bg_mem[0])):
                    img[ln.line_bg_mem[0][i], ln.line_bg_mem[1][i]] = color

                for i in range(len(ln.line_cyt_sept[0])):
                    img[ln.line_cyt_sept[0][i], ln.line_cyt_sept[1][i]] = color
            self.min_scale.set(int(self.fluor_min * 100))
            self.max_scale.set(int(self.fluor_max * 100))
            self.current_image_label.configure(text="Linescan")

        elif image == "Optional":
            img = rescale_intensity(self.images[image], in_range=(self.optional_min, self.optional_max))
            self.min_scale.set(int(self.optional_min * 100))
            self.max_scale.set(int(self.optional_max * 100))
            self.current_image_label.configure(text="Secondary")
        elif image == "Optional_cells_outlined":
            img = rescale_intensity(self.images[image], in_range=(self.optional_min, self.optional_max))
            img = cp.overlay_cells(self.ehooke.cell_manager.cells, img, self.ehooke.cell_manager.cell_colors)
            self.min_scale.set(int(self.optional_min * 100))
            self.max_scale.set(int(self.optional_max * 100))
            self.current_image_label.configure(text="Secondary Outlined")

        self.ax.imshow(img, interpolation="none", cmap=cm.Greys_r)

        plt.subplots_adjust(left=0.005, bottom=0.005, right=0.995, top=0.995)
        # figZoom = self.zoom_factory(self.ax)
        # figPan = self.pan_factory(self.ax)
        self.canvas.draw()

    def load_base_image(self):
        """Loads the base image"""
        self.ehooke.parameters.imageloaderparams.border = \
            self.border_value.get()
        self.ehooke.load_base_image()
        self.images["Base"] = self.ehooke.image_manager.base_image

        self.show_image("Base")
        self.compute_mask_button.config(state="active")
        self.base_button.config(state="active")
        self.status.set("Base Image Loaded. Proceed with mask computation")
        self.main_window.wm_title(
            "eHooke - Base:" + str(self.ehooke.base_path))

    def compute_mask(self):
        """Computes the mask of the cell regions"""
        self.ehooke.parameters.imageloaderparams.invert_base = \
            self.fluor_as_base_value.get()
        self.ehooke.parameters.imageloaderparams.mask_algorithm = \
            self.mask_algorithm_value.get()
        self.ehooke.parameters.imageloaderparams.mask_blocksize = \
            self.mask_blocksize_value.get()
        self.ehooke.parameters.imageloaderparams.mask_offset = \
            self.mask_offset_value.get()
        self.ehooke.parameters.imageloaderparams.mask_fill_holes = \
            self.mask_fillholes_value.get()
        self.ehooke.parameters.imageloaderparams.mask_closing = \
            self.mask_closing_value.get()
        self.ehooke.parameters.imageloaderparams.mask_dilation = \
            self.mask_dilation_value.get()
        self.ehooke.compute_mask()
        self.images["Mask"] = self.ehooke.image_manager.mask
        self.images["Base_mask"] = self.ehooke.image_manager.base_w_mask

        if self.ehooke.image_manager.fluor_image is not None:
            self.images["Fluor_mask"] = self.ehooke.image_manager.fluor_w_mask
            self.show_image("Fluor_mask")
        else:
            self.show_image("Base_mask")

        self.load_fluorescence_button.config(state="active")
        self.mask_button.config(state="active")
        self.base_with_mask_button.config(state="active")
        self.save_mask_button.config(state="active")
        self.status.set("Mask computation finished. Load Fluorescence Image")

    def load_fluor(self):
        """Loads the fluor image"""
        self.ehooke.parameters.imageloaderparams.auto_align = \
            self.auto_align_value.get()
        self.ehooke.parameters.imageloaderparams.x_align = \
            self.x_align_value.get()
        self.ehooke.parameters.imageloaderparams.y_align = \
            self.y_align_value.get()
        self.ehooke.load_fluor_image()
        self.images["Fluor"] = \
            rescale_intensity(img_as_float(self.ehooke.image_manager.original_fluor_image))
        self.images["Fluor_mask"] = self.ehooke.image_manager.fluor_w_mask
        self.show_image("Fluor_mask")
        self.next_button.config(state="active")
        self.fluor_button.config(state="active")
        self.fluor_with_mask_button.config(state="active")
        self.load_optional_button.config(state="active")
        self.status.set("Fluorescence Image Loaded. Proceed to the next step")
        self.main_window.wm_title("eHooke - Base:" + str(self.ehooke.base_path) +
                                  " - Fluorescence: " + str(self.ehooke.fluor_path))

    def load_optional(self):
        self.ehooke.load_option_image()
        self.images["Optional"] = rescale_intensity(img_as_float(self.ehooke.image_manager.optional_image))
        self.show_image("Optional")
        self.optional_button.config(state="active")
        self.status.set("Optional Image Loaded. Proceed to the next step")

    def save_mask(self):
        self.ehooke.save_mask()

    def set_imageloader(self):
        """Method used to change the interface to the Image Loader Step"""
        plt.clf()
        self.ax = plt.subplot(111)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.ax.axis("off")
        plt.autoscale(False)
        self.ax.format_coord = self.remove_coord

        self.canvas.draw()

        for w in self.top_frame.winfo_children():
            w.destroy()

        for w in self.parameters_panel.winfo_children():
            w.destroy()

        for w in self.images_frame.winfo_children():
            w.destroy()

        self.current_step = "ImageLoading"

        self.load_base_button = tk.Button(self.top_frame,
                                          text="Load Base Image",
                                          command=self.load_base_image)
        self.load_base_button.pack(side="left")

        self.compute_mask_button = tk.Button(self.top_frame,
                                             text="Compute Mask",
                                             command=self.compute_mask)
        self.compute_mask_button.pack(side="left")
        self.compute_mask_button.config(state="disabled")

        self.load_fluorescence_button = tk.Button(self.top_frame,
                                                  text="Load Fluorescence",
                                                  command=self.load_fluor)
        self.load_fluorescence_button.pack(side="left")
        self.load_fluorescence_button.config(state="disabled")

        self.load_optional_button = tk.Button(self.top_frame,
                                              text="Load Secondary Channel",
                                              command=self.load_optional)
        self.load_optional_button.pack(side="left")
        self.load_optional_button.config(state="disabled")

        self.load_params_button = tk.Button(self.parameters_panel,
                                            text="Load Parameters",
                                            command=self.load_parameters)
        self.load_params_button.pack(side="top", fill="x")

        self.base_parameters_label = tk.Label(self.parameters_panel,
                                              text="Load Base Parameters:")
        self.base_parameters_label.config(width=self.parameters_panel_width)
        self.base_parameters_label.pack(side="top", fill="x", pady=self.pady)

        self.border_frame = tk.Frame(self.parameters_panel)
        self.border_frame.pack(side="top", fill="both")
        self.border_label = tk.Label(self.border_frame, text="Border: ")
        self.border_label.pack(side="left")
        self.border_value = tk.IntVar()
        self.border_entry = tk.Entry(self.border_frame,
                                     textvariable=self.border_value, width=4)
        self.border_entry.pack(side="left")
        self.border_value.set(self.ehooke.parameters.imageloaderparams.border)

        self.fluor_as_base_frame = tk.Frame(self.parameters_panel)
        self.fluor_as_base_frame.pack(side="top", fill="x")
        self.fluor_as_base_label = tk.Label(self.fluor_as_base_frame,
                                            text="Use Fluorescence Image as Base: ")
        self.fluor_as_base_label.pack(side="left")
        self.fluor_as_base_value = tk.BooleanVar()
        self.fluor_as_base_checkbox = tk.Checkbutton(self.fluor_as_base_frame,
                                                     variable=self.fluor_as_base_value, onvalue=True, offvalue=False)
        self.fluor_as_base_checkbox.pack(side="left")
        self.fluor_as_base_value.set(
            self.ehooke.parameters.imageloaderparams.invert_base)

        self.mask_parameters_label = tk.Label(
            self.parameters_panel, text="Mask Parameters:")
        self.mask_parameters_label.pack(side="top", fill="x", pady=self.pady)

        self.mask_algorithm_frame = tk.Frame(self.parameters_panel)
        self.mask_algorithm_frame.pack(side="top", fill="x")
        self.mask_algorithm_label = tk.Label(
            self.mask_algorithm_frame, text="Mask Algorithm: ")
        self.mask_algorithm_label.pack(side="left")
        self.mask_algorithm_value = tk.StringVar()
        self.mask_algorithm_menu = tk.OptionMenu(self.mask_algorithm_frame, self.mask_algorithm_value,
                                                 'Local Average', 'Isodata', 'StarDist', 'StarDist_BF')
        self.mask_algorithm_menu.pack(side="left")
        self.mask_algorithm_value.set(self.ehooke.parameters.imageloaderparams.mask_algorithm)

        self.mask_blocksize_frame = tk.Frame(self.parameters_panel)
        self.mask_blocksize_frame.pack(side="top", fill="x")
        self.mask_blocksize_label = tk.Label(
            self.mask_blocksize_frame, text="Local Average Blocksize: ")
        self.mask_blocksize_label.pack(side="left")
        self.mask_blocksize_value = tk.IntVar()
        self.mask_blocksize_entry = tk.Entry(
            self.mask_blocksize_frame, textvariable=self.mask_blocksize_value, width=4)
        self.mask_blocksize_entry.pack(side="left")
        self.mask_blocksize_value.set(
            self.ehooke.parameters.imageloaderparams.mask_blocksize)

        self.mask_offset_frame = tk.Frame(self.parameters_panel)
        self.mask_offset_frame.pack(side="top", fill="x")
        self.mask_offset_label = tk.Label(
            self.mask_offset_frame, text="Local Average Offset: ")
        self.mask_offset_label.pack(side="left")
        self.mask_offset_value = tk.DoubleVar()
        self.mask_offset_entry = tk.Entry(
            self.mask_offset_frame, textvariable=self.mask_offset_value, width=4)
        self.mask_offset_entry.pack(side="left")
        self.mask_offset_value.set(
            self.ehooke.parameters.imageloaderparams.mask_offset)

        self.mask_fillholes_frame = tk.Frame(self.parameters_panel)
        self.mask_fillholes_frame.pack(side="top", fill="x")
        self.mask_fillholes_label = tk.Label(
            self.mask_fillholes_frame, text="Fill Holes: ")
        self.mask_fillholes_label.pack(side="left")
        self.mask_fillholes_value = tk.BooleanVar()
        self.mask_fillholes_checkbox = tk.Checkbutton(self.mask_fillholes_frame,
                                                      variable=self.mask_fillholes_value, onvalue=True, offvalue=False)
        self.mask_fillholes_checkbox.pack(side="left")
        self.mask_fillholes_value.set(
            self.ehooke.parameters.imageloaderparams.mask_fill_holes)

        self.mask_closing_frame = tk.Frame(self.parameters_panel)
        self.mask_closing_frame.pack(side="top", fill="x")
        self.mask_closing_label = tk.Label(
            self.mask_closing_frame, text="Mask Closing: ")
        self.mask_closing_label.pack(side="left")
        self.mask_closing_value = tk.DoubleVar()
        self.mask_closing_entry = tk.Entry(
            self.mask_closing_frame, textvariable=self.mask_closing_value, width=4)
        self.mask_closing_entry.pack(side="left")
        self.mask_closing_value.set(
            self.ehooke.parameters.imageloaderparams.mask_closing)

        self.mask_dilation_frame = tk.Frame(self.parameters_panel)
        self.mask_dilation_frame.pack(side="top", fill="x")
        self.mask_dilation_label = tk.Label(
            self.mask_dilation_frame, text="Mask Dilation: ")
        self.mask_dilation_label.pack(side="left")
        self.mask_dilation_value = tk.IntVar()
        self.mask_dilation_entry = tk.Entry(
            self.mask_dilation_frame, textvariable=self.mask_dilation_value, width=4)
        self.mask_dilation_entry.pack(side="left")
        self.mask_dilation_value.set(
            self.ehooke.parameters.imageloaderparams.mask_dilation)

        self.load_fluor_label = tk.Label(self.parameters_panel,
                                         text="Fluorescence Image Parameters:")
        self.load_fluor_label.pack(side="top", fill="x", pady=self.pady)

        self.auto_align_frame = tk.Frame(self.parameters_panel)
        self.auto_align_frame.pack(side="top", fill="x")
        self.auto_align_label = tk.Label(
            self.auto_align_frame, text="Auto-align: ")
        self.auto_align_label.pack(side="left")
        self.auto_align_value = tk.BooleanVar()
        self.auto_align_checkbox = tk.Checkbutton(self.auto_align_frame,
                                                  variable=self.auto_align_value, onvalue=True, offvalue=False)
        self.auto_align_checkbox.pack(side="left")
        self.auto_align_value.set(
            self.ehooke.parameters.imageloaderparams.auto_align)

        self.x_align_frame = tk.Frame(self.parameters_panel)
        self.x_align_frame.pack(side="top", fill="x")
        self.x_align_label = tk.Label(self.x_align_frame, text="X align: ")
        self.x_align_label.pack(side="left")
        self.x_align_value = tk.IntVar()
        self.x_align_entry = tk.Entry(
            self.x_align_frame, textvariable=self.x_align_value, width=4)
        self.x_align_entry.pack(side="left")
        self.x_align_value.set(
            self.ehooke.parameters.imageloaderparams.x_align)

        self.y_align_frame = tk.Frame(self.parameters_panel)
        self.y_align_frame.pack(side="top", fill="x")
        self.y_align_label = tk.Label(self.y_align_frame, text="Y align: ")
        self.y_align_label.pack(side="left")
        self.y_align_value = tk.IntVar()
        self.y_align_entry = tk.Entry(
            self.y_align_frame, textvariable=self.y_align_value, width=4)
        self.y_align_entry.pack(side="left")
        self.y_align_value.set(
            self.ehooke.parameters.imageloaderparams.y_align)

        self.pixel_size_section = tk.Label(self.parameters_panel, text="Pixel Size Parameters:")
        self.pixel_size_section.pack(side="top", fill="x")

        self.pixel_size_frame = tk.Frame(self.parameters_panel)
        self.pixel_size_frame.pack(side="top", fill="x")
        self.pixel_size_label = tk.Label(self.pixel_size_frame, text="Pixel size: ")
        self.pixel_size_label.pack(side="left")
        self.pixel_size_value = tk.StringVar()
        self.pixel_size_entry = tk.Entry(
            self.pixel_size_frame, textvariable=self.pixel_size_value, width=8)
        self.pixel_size_entry.pack(side="left")
        self.pixel_size_value.set(
            self.ehooke.parameters.imageloaderparams.pixel_size)

        self.units_frame = tk.Frame(self.parameters_panel)
        self.units_frame.pack(side="top", fill="x")
        self.units_label = tk.Label(self.units_frame, text="Units: ")
        self.units_label.pack(side="left")
        self.units_value = tk.StringVar()
        self.units_entry = tk.Entry(
            self.units_frame, textvariable=self.units_value, width=8)
        self.units_entry.pack(side="left")
        self.units_value.set(
            self.ehooke.parameters.imageloaderparams.units)

        self.imgloader_params_default_button = tk.Button(self.parameters_panel, text="Default Parameters",
                                                         command=self.load_default_params_imgloader)
        self.imgloader_params_default_button.pack(side="top", fill="x")

        self.next_button = tk.Button(
            self.top_frame, text="Next", command=self.set_segmentscomputation)
        self.next_button.pack(side="right")
        self.next_button.config(state="disabled")

        self.save_mask_button = tk.Button(self.top_frame, text="Save Mask", command=self.save_mask)
        self.save_mask_button.pack(side="right")
        self.save_mask_button.config(state="disabled")

        self.base_button = tk.Button(self.images_frame, text="Base", command=lambda: self.show_image("Base"),
                                     width=self.image_buttons_width)
        self.base_button.pack(side="top", fill="x")
        self.base_button.config(state="disabled")

        self.mask_button = tk.Button(self.images_frame, text="Mask", command=lambda: self.show_image("Mask"),
                                     width=self.image_buttons_width)
        self.mask_button.pack(side="top", fill="x")
        self.mask_button.config(state="disabled")

        self.base_with_mask_button = tk.Button(self.images_frame, text="Base with mask",
                                               command=lambda: self.show_image(
                                                   "Base_mask"),
                                               width=self.image_buttons_width)
        self.base_with_mask_button.pack(side="top", fill="x")
        self.base_with_mask_button.config(state="disabled")

        self.fluor_button = tk.Button(self.images_frame, text="Fluorescence",
                                      command=lambda: self.show_image("Fluor"), width=self.image_buttons_width)
        self.fluor_button.pack(side="top", fill="x")
        self.fluor_button.config(state="disabled")

        self.fluor_with_mask_button = tk.Button(self.images_frame, text="Fluorescence with mask",
                                                command=lambda: self.show_image(
                                                    "Fluor_mask"),
                                                width=self.image_buttons_width)
        self.fluor_with_mask_button.pack(side="top", fill="x")
        self.fluor_with_mask_button.config(state="disabled")

        self.optional_button = tk.Button(self.images_frame, text="Secondary Channel",
                                         command=lambda: self.show_image(
                                             "Optional"),
                                         width=self.image_buttons_width)
        self.optional_button.pack(side="top", fill="x")
        self.optional_button.config(state="disabled")

        self.current_image_label = tk.Label(self.images_frame, text="")
        self.current_image_label.pack(side="top")

        self.status = tk.StringVar()
        self.status.set("Load Base Image")
        self.status_bar = tk.Label(
            self.parameters_panel, textvariable=self.status, wraplength=self.status_length)
        self.status_bar.pack(side="bottom")

        self.config_gui(self.main_window)
        self.base_parameters_label.config(font=self.gui_font_bold)
        self.mask_parameters_label.config(font=self.gui_font_bold)
        self.load_fluor_label.config(font=self.gui_font_bold)
        self.pixel_size_section.config(font=self.gui_font_bold)

    def compute_features(self):
        """Calls the compute_segments method from ehooke"""
        self.ehooke.parameters.imageprocessingparams.peak_min_distance = self.peak_min_distance_value.get()
        self.ehooke.parameters.imageprocessingparams.peak_min_height = self.peak_min_height_value.get()
        self.ehooke.parameters.imageprocessingparams.peak_min_distance_from_edge = self.peak_min_distance_edge_value.get()
        self.ehooke.parameters.imageprocessingparams.max_peaks = self.max_peaks_value.get()
        self.ehooke.parameters.imageprocessingparams.outline_use_base_mask = self.use_base_mask_value.get()
        self.ehooke.compute_segments()
        self.images[
            "Base_features"] = self.ehooke.segments_manager.base_w_features
        self.images[
            "Fluor_features"] = self.ehooke.segments_manager.fluor_w_features
        self.images["Labels"] = mark_boundaries(rescale_intensity(
            self.ehooke.image_manager.fluor_image), (self.ehooke.segments_manager.labels > 0), color=(1, 0, 0))
        self.show_image("Base_features")
        self.next_button.config(state="active")
        self.base_features_button.config(state="active")
        self.fluor_features_button.config(state="active")
        self.save_labels_button.config(state="active")
        self.labels_button.config(state="active")
        self.status.set(
            "Computation of the features finished. Proceed to the next step")

    def save_labels(self):
        self.ehooke.save_labels()

    def set_segmentscomputation(self):
        """Method used to change the interface to the Segments Computation
        Step"""

        self.current_step = "SegmentComputation"

        self.ehooke.parameters.imageloaderparams.pixel_size = self.pixel_size_value.get()
        self.ehooke.parameters.imageloaderparams.units = self.units_value.get()

        if self.ehooke.parameters.imageloaderparams.units == "um":
            self.ehooke.parameters.imageloaderparams.units = "\u03BC" + "m"
            self.ehooke.parameters.cellprocessingparams.cell_force_merge_below = 0.065

        self.ax.axis("off")
        self.show_image("Fluor_mask")
        self.ax.format_coord = self.remove_coord
        self.canvas.draw()

        for w in self.top_frame.winfo_children():
            w.destroy()

        for w in self.parameters_panel.winfo_children():
            w.destroy()

        for w in self.images_frame.winfo_children():
            w.destroy()

        self.status = tk.StringVar()
        self.status_bar = tk.Label(self.parameters_panel,
                                   textvariable=self.status,
                                   wraplength=self.status_length)
        self.status_bar.pack(side="bottom")
        self.status.set("Waiting for features computation")

        self.compute_features_button = tk.Button(self.top_frame,
                                                 text="Compute Features",
                                                 command=self.compute_features)
        self.compute_features_button.pack(side="left")

        self.next_button = tk.Button(
            self.top_frame, text="Next", command=self.set_cellcomputation)
        self.next_button.pack(side="right")
        self.next_button.config(state="disabled")

        self.back_button = tk.Button(
            self.top_frame, text="Back", command=self.new_analysis)
        self.back_button.pack(side="right")

        self.save_labels_button = tk.Button(self.top_frame, text="Save Labels", command=self.save_labels)
        self.save_labels_button.pack(side="right")
        self.save_labels_button.config(state="disabled")

        self.new_analysis_button = tk.Button(
            self.top_frame, text="New", command=self.new_analysis)
        self.new_analysis_button.pack(side="right")

        self.segments_parameters_label = tk.Label(self.parameters_panel,
                                                  text="Segments Computation Parameters:")
        self.segments_parameters_label.config(width=self.parameters_panel_width)
        self.segments_parameters_label.pack(side="top", fill="x", pady=self.pady)

        self.peak_min_distance_frame = tk.Frame(self.parameters_panel)
        self.peak_min_distance_frame.pack(side="top", fill="x")
        self.peak_min_distance_label = tk.Label(self.peak_min_distance_frame,
                                                text="Peak Min Distance: ")
        self.peak_min_distance_label.pack(side="left")
        self.peak_min_distance_value = tk.IntVar()
        self.peak_min_distance_entry = tk.Entry(self.peak_min_distance_frame,
                                                textvariable=self.peak_min_distance_value, width=4)
        self.peak_min_distance_entry.pack(side="left")
        self.peak_min_distance_value.set(
            int(self.ehooke.parameters.imageprocessingparams.peak_min_distance))

        self.peak_min_height_frame = tk.Frame(self.parameters_panel)
        self.peak_min_height_frame.pack(side="top", fill="x")
        self.peak_min_height_label = tk.Label(
            self.peak_min_height_frame, text="Peak Min Height: ")
        self.peak_min_height_label.pack(side="left")
        self.peak_min_height_value = tk.IntVar()
        self.peak_min_height_entry = tk.Entry(self.peak_min_height_frame,
                                              textvariable=self.peak_min_height_value, width=4)
        self.peak_min_height_entry.pack(side="left")
        self.peak_min_height_value.set(
            int(self.ehooke.parameters.imageprocessingparams.peak_min_height))

        self.peak_min_distance_edge_frame = tk.Frame(self.parameters_panel)
        self.peak_min_distance_edge_frame.pack(side="top", fill="x")
        self.peak_min_distance_edge_label = tk.Label(
            self.peak_min_distance_edge_frame, text="Peak Min Margin: ")
        self.peak_min_distance_edge_label.pack(side="left")
        self.peak_min_distance_edge_value = tk.IntVar()
        self.peak_min_distance_edge_entry = tk.Entry(self.peak_min_distance_edge_frame,
                                                     textvariable=self.peak_min_distance_edge_value, width=4)
        self.peak_min_distance_edge_entry.pack(side="left")
        self.peak_min_distance_edge_value.set(int(
            self.ehooke.parameters.imageprocessingparams.peak_min_distance_from_edge))

        self.max_peaks_frame = tk.Frame(self.parameters_panel)
        self.max_peaks_frame.pack(side="top", fill="x")
        self.max_peaks_label = tk.Label(
            self.max_peaks_frame, text="Max Peaks: ")
        self.max_peaks_label.pack(side="left")
        self.max_peaks_value = tk.IntVar()
        self.max_peaks_entry = tk.Entry(self.max_peaks_frame,
                                        textvariable=self.max_peaks_value, width=5)
        self.max_peaks_entry.pack(side="left")
        self.max_peaks_value.set(
            int(self.ehooke.parameters.imageprocessingparams.max_peaks))

        self.use_base_mask_frame = tk.Frame(self.parameters_panel)
        self.use_base_mask_frame.pack(side="top", fill="x")
        self.use_base_mask_label = tk.Label(
            self.use_base_mask_frame, text="Use Base Mask: ")
        self.use_base_mask_label.pack(side="left")
        self.use_base_mask_value = tk.BooleanVar()
        self.use_base_mask_checkbox = tk.Checkbutton(self.use_base_mask_frame, variable=self.use_base_mask_value,
                                                     onvalue=True, offvalue=False)
        self.use_base_mask_checkbox.pack(side="left")
        self.use_base_mask_value.set(
            self.ehooke.parameters.imageprocessingparams.outline_use_base_mask)

        self.features_default_button = tk.Button(self.parameters_panel, text="Default Parameters",
                                                 command=self.load_default_params_segments)
        self.features_default_button.pack(side="top", fill="x")

        self.fluor_with_mask_button = tk.Button(self.images_frame, text="Fluorescence with Mask",
                                                command=lambda: self.show_image(
                                                    "Fluor_mask"),
                                                width=self.image_buttons_width)
        self.fluor_with_mask_button.pack(side="top", fill="x")

        self.optional_button = tk.Button(self.images_frame, text="Secondary Channel",
                                         command=lambda: self.show_image(
                                             "Optional"),
                                         width=self.image_buttons_width)
        self.optional_button.pack(side="top", fill="x")
        if self.ehooke.image_manager.optional_image is None:
            self.optional_button.config(state="disabled")
        else:
            self.optional_button.config(state="active")

        self.fluor_features_button = tk.Button(self.images_frame, text="Fluor with Features",
                                               command=lambda: self.show_image(
                                                   "Fluor_features"),
                                               width=self.image_buttons_width)
        self.fluor_features_button.pack(side="top", fill="x")
        self.fluor_features_button.config(state="disabled")

        self.base_features_button = tk.Button(self.images_frame, text="Base with Features",
                                              command=lambda: self.show_image(
                                                  "Base_features"),
                                              width=self.image_buttons_width)
        self.base_features_button.pack(side="top", fill="x")
        self.base_features_button.config(state="disabled")

        self.labels_button = tk.Button(self.images_frame,
                                       text="Watershed Labels",
                                       command=lambda: self.show_image(
                                           "Labels"),
                                       width=self.image_buttons_width)
        # self.labels_button.pack(side="top", fill="x")
        # self.labels_button.config(state="disabled")

        self.current_image_label = tk.Label(self.images_frame, text="")
        self.current_image_label.pack(side="top")

        self.config_gui(self.main_window)

        self.segments_parameters_label.config(font=self.gui_font_bold)

        self.show_image(self.current_image)

    def show_cell_info_cellcomputation(self, x, y):
        """Shows the stats of each cell on the side panel"""
        label = int(self.ehooke.cell_manager.merged_labels[int(y), int(x)])

        if 0 < label:
            stats = self.ehooke.cell_manager.cells[str(label)].stats

            self.cellid_value.set(label)
            self.merged_with_value.set(self.ehooke.cell_manager.cells[
                                           str(label)].merged_with)
            self.marked_as_noise_value.set(self.ehooke.cell_manager.cells[
                                               str(label)].marked_as_noise)
            self.area_value.set(float(str(stats["Area"])[0:7]))
            self.perimeter_value.set(float(str(stats["Perimeter"])[0:7]))
            self.length_value.set(float(str(stats["Length"])[0:7]))
            self.width_value.set(float(str(stats["Width"])[0:7]))
            self.eccentricity_value.set(float(str(stats["Eccentricity"])[0:7]))
            self.irregularity_value.set(float(str(stats["Irregularity"])[0:7]))
            self.neighbours_value.set(stats["Neighbours"])

        else:
            self.cellid_value.set(0)
            self.merged_with_value.set("No")
            self.marked_as_noise_value.set("No")
            self.area_value.set(0)
            self.perimeter_value.set(0)
            self.length_value.set(0)
            self.width_value.set(0)
            self.eccentricity_value.set(0)
            self.irregularity_value.set(0)
            self.neighbours_value.set(0)

        lum = self.ehooke.image_manager.original_fluor_image[int(y), int(x)]
        return "Luminance: " + str(lum)

    def compute_cells(self):
        """Method used to compute the cells"""
        self.ehooke.parameters.imageprocessingparams.axial_step = self.axial_step_value.get()
        self.ehooke.parameters.cellprocessingparams.cell_force_merge_below = self.force_merge_below_value.get()
        self.ehooke.parameters.cellprocessingparams.merge_length_tolerance = self.merge_length_tolerance_value.get()
        self.ehooke.parameters.cellprocessingparams.merge_dividing_cells = self.merge_dividing_value.get()
        self.ehooke.parameters.cellprocessingparams.merge_min_interface = self.merge_min_interface_value.get()
        self.status.set("Computing cells...")
        self.ehooke.compute_cells()
        self.ax.format_coord = self.show_cell_info_cellcomputation

        self.current_step = "CellsComputed"

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells

        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

        self.show_image("Fluor_cells_outlined")

        self.next_button.config(state="active")
        self.merge_from_file_button.config(state="active")
        self.force_merge_button.config(state="active")
        self.split_cell_button.config(state="active")
        self.declare_as_noise_button.config(state="active")
        self.undo_as_noise_button.config(state="active")
        self.base_w_cells_button.config(state="active")
        self.fluor_cells_out_button.config(state="active")
        self.status.set("Cell Computation Finished. Proceed to the next step")

    def merge_on_press(self, event):

        if event.button == 3:
            label = int(self.ehooke.cell_manager.merged_labels[
                            int(event.ydata), int(event.xdata)])

            if label > 0:

                if len(self.merge_list) < 1:
                    self.merge_list.append(label)
                    self.status.set("Waiting for second cell")

                elif len(self.merge_list) == 1:
                    if label != self.merge_list[0]:
                        self.status.set("Merging cells...")

                        self.ehooke.merge_cells(self.merge_list[0], label)
                        self.images[
                            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
                        self.images[
                            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

                        self.show_image(self.current_image)

                        self.canvas.mpl_disconnect(self.cid)
                        self.event_connected = False
                        self.merge_list = []

                        self.status.set("Merge Finished")
                    else:
                        self.canvas.mpl_disconnect(self.cid)
                        self.event_connected = False
                        self.status.set("Same Cell. Repeat Merge.")

            else:
                self.canvas.mpl_disconnect(self.cid)
                self.event_connected = False
                self.status.set("Not a Cell. Repeat Merge")

    def force_merge(self):
        """Method used to force the merge of two cells"""
        if self.event_connected:
            self.canvas.mpl_disconnect(self.cid)
        self.status.set("Right-click on two cells")
        self.merge_list = []
        self.cid = self.canvas.mpl_connect(
            'button_release_event', self.merge_on_press)
        self.event_connected = True

    def splitting_on_press(self, event):
        if event.button == 3:
            self.status.set("Splitting Cells")
            label = int(self.ehooke.cell_manager.merged_labels[
                            int(event.ydata), int(event.xdata)])

            if label > 0:
                if self.ehooke.cell_manager.cells[str(label)].merged_with != "No":
                    self.ehooke.split_cells(label)
                    self.images[
                        "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
                    self.images[
                        "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

                    self.show_image(self.current_image)

            self.canvas.mpl_disconnect(self.cid)
            self.event_connected = False
            self.status.set("Splitting Finished")

    def split_cell(self):
        """Method used to split a previously merged cell"""
        if self.event_connected:
            self.canvas.mpl_disconnect(self.cid)
        self.status.set("Right-click on cell")
        self.cid = self.canvas.mpl_connect(
            'button_release_event', self.splitting_on_press)
        self.event_connected = True

    def noise_on_press(self, event):
        if event.button == 3:
            self.status.set("Removing Cell")
            label = int(self.ehooke.cell_manager.merged_labels[
                            int(event.ydata), int(event.xdata)])

            if label > 0:
                if self.ehooke.cell_manager.cells[str(label)].selection_state != 0:
                    self.ehooke.define_as_noise(label, True)
                    self.images[
                        "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
                    self.images[
                        "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

                    self.show_image(self.current_image)

            self.canvas.mpl_disconnect(self.cid)
            self.event_connected = False
            self.status.set("May Proceed to Cell Processing")

    def declare_as_noise(self):
        """Method used to define a cell object as noise"""
        if self.event_connected:
            self.canvas.mpl_disconnect(self.cid)
        self.status.set("Righ-click noise object")
        self.cid = self.canvas.mpl_connect(
            'button_release_event', self.noise_on_press)
        self.event_connected = True

    def undo_noise_on_press(self, event):
        if event.button == 3:
            self.status.set("Adding Cell")
            label = int(self.ehooke.cell_manager.merged_labels[
                            int(event.ydata), int(event.xdata)])

            if label > 0:
                if self.ehooke.cell_manager.cells[str(label)].selection_state == 0:
                    self.ehooke.define_as_noise(label, False)
                    self.images[
                        "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
                    self.images[
                        "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

                    self.show_image(self.current_image)

            self.canvas.mpl_disconnect(self.cid)
            self.event_connected = False
            self.status.set("May Proceed to Cell Processing")

    def undo_as_noise(self):
        """Method used to define a a cell from an object that was previously
        defined as noise"""
        if self.event_connected:
            self.canvas.mpl_disconnect(self.cid)
        self.status.set("Righ-click cell object")
        self.cid = self.canvas.mpl_connect(
            'button_release_event', self.undo_noise_on_press)
        self.event_connected = True

    def merge_from_file(self):

        self.ehooke.merge_from_file()
        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

        self.show_image(self.current_image)

    def set_cellcomputation(self):
        """Method used to change the interface to the Cell Computation
        Step"""
        self.ax.axis("off")
        self.show_image("Base_features")
        self.ax.format_coord = self.remove_coord
        self.canvas.draw()

        for w in self.top_frame.winfo_children():
            w.destroy()

        for w in self.parameters_panel.winfo_children():
            w.destroy()

        for w in self.images_frame.winfo_children():
            w.destroy()

        self.current_step = "CellComputation"

        self.status = tk.StringVar()
        self.status_bar = tk.Label(
            self.parameters_panel, textvariable=self.status, wraplength=self.status_length)
        self.status_bar.pack(side="bottom")
        self.status.set("Waiting for cell computation")

        self.compute_cells_button = tk.Button(
            self.top_frame, text="Compute Cells", command=self.compute_cells)
        self.compute_cells_button.pack(side="left")

        self.next_button = tk.Button(
            self.top_frame, text="Next", command=self.set_cellprocessing)
        self.next_button.pack(side="right")
        self.next_button.config(state="disabled")
        self.back_button = tk.Button(
            self.top_frame, text="Back", command=self.set_segmentscomputation)
        self.back_button.pack(side="right")

        self.new_analysis_button = tk.Button(
            self.top_frame, text="New", command=self.new_analysis)
        self.new_analysis_button.pack(side="right")

        self.cellcomputation_parameters_label = tk.Label(self.parameters_panel,
                                                         text="Cell Computation Parameters:")
        self.cellcomputation_parameters_label.config(width=self.parameters_panel_width)
        self.cellcomputation_parameters_label.pack(side="top", fill="x", pady=self.pady)

        self.axial_step_frame = tk.Frame(self.parameters_panel)
        self.axial_step_frame.pack(side="top", fill="x")
        self.axial_step_label = tk.Label(
            self.axial_step_frame, text="Axial Step: ")
        self.axial_step_label.pack(side="left")
        self.axial_step_value = tk.IntVar()
        self.axial_step_entry = tk.Entry(
            self.axial_step_frame, textvariable=self.axial_step_value, width=4)
        self.axial_step_entry.pack(side="left")
        self.axial_step_value.set(
            self.ehooke.parameters.cellprocessingparams.axial_step)

        self.force_merge_below_frame = tk.Frame(self.parameters_panel)
        self.force_merge_below_frame.pack(side="top", fill="x")
        self.force_merge_below_label = tk.Label(
            self.force_merge_below_frame, text="Force Merge If Area Below: ")
        self.force_merge_below_label.pack(side="left")
        self.force_merge_below_value = tk.DoubleVar()
        self.force_merge_below_entry = tk.Entry(self.force_merge_below_frame,
                                                textvariable=self.force_merge_below_value, width=10)
        self.force_merge_below_entry.pack(side="left")
        self.force_merge_below_value.set(
            self.ehooke.parameters.cellprocessingparams.cell_force_merge_below)

        self.merge_dividing_frame = tk.Frame(self.parameters_panel)
        self.merge_dividing_frame.pack(side="top", fill="x")
        self.merge_dividing_label = tk.Label(
            self.merge_dividing_frame, text="Merge Dividing Cells: ")
        self.merge_dividing_label.pack(side="left")
        self.merge_dividing_value = tk.BooleanVar()
        self.merge_dividing_checkbox = tk.Checkbutton(self.merge_dividing_frame, variable=self.merge_dividing_value,
                                                      onvalue=True, offvalue=False)
        self.merge_dividing_checkbox.pack(side="left")
        self.merge_dividing_value.set(
            self.ehooke.parameters.cellprocessingparams.merge_dividing_cells)

        self.merge_length_tolerance_frame = tk.Frame(self.parameters_panel)
        self.merge_length_tolerance_frame.pack(side="top", fill="x")
        self.merge_length_tolerance_label = tk.Label(self.merge_length_tolerance_frame,
                                                     text="Length Tolerance on Merge: ")
        self.merge_length_tolerance_label.pack(side="left")
        self.merge_length_tolerance_value = tk.DoubleVar()
        self.merge_length_tolerance_entry = tk.Entry(self.merge_length_tolerance_frame,
                                                     textvariable=self.merge_length_tolerance_value, width=4)
        self.merge_length_tolerance_entry.pack(side="left")
        self.merge_length_tolerance_value.set(
            self.ehooke.parameters.cellprocessingparams.merge_length_tolerance)

        self.merge_min_interface_frame = tk.Frame(self.parameters_panel)
        self.merge_min_interface_frame.pack(side="top", fill="x")
        self.merge_min_interface_label = tk.Label(
            self.merge_min_interface_frame, text="Min Interface for Merge: ")
        self.merge_min_interface_label.pack(side="left")
        self.merge_min_interface_value = tk.IntVar()
        self.merge_min_interface_entry = tk.Entry(self.merge_min_interface_frame,
                                                  textvariable=self.merge_min_interface_value, width=4)
        self.merge_min_interface_entry.pack(side="left")
        self.merge_min_interface_value.set(
            self.ehooke.parameters.cellprocessingparams.merge_min_interface)

        self.merge_from_file_button = tk.Button(self.parameters_panel, text="Load Merge List",
                                                command=self.merge_from_file)
        self.merge_from_file_button.pack(side="top", fill="x")
        self.merge_from_file_button.config(state="disabled")

        self.force_merge_button = tk.Button(self.parameters_panel, text="Force Merge (M)",
                                            command=self.force_merge)
        self.force_merge_button.pack(side="top", fill="x")
        self.force_merge_button.config(state="disabled")

        self.split_cell_button = tk.Button(
            self.parameters_panel, text="Undo merge (S)", command=self.split_cell)
        self.split_cell_button.pack(side="top", fill="x")
        self.split_cell_button.config(state="disabled")

        self.declare_as_noise_button = tk.Button(self.parameters_panel, text="Define as Noise (N)",
                                                 command=self.declare_as_noise)
        self.declare_as_noise_button.config(state="disabled")
        self.declare_as_noise_button.pack(side="top", fill="x")

        self.undo_as_noise_button = tk.Button(self.parameters_panel, text="Undo as Noise (U)",
                                              command=self.undo_as_noise)
        self.undo_as_noise_button.config(state="disabled")
        self.undo_as_noise_button.pack(side="top", fill="x")

        self.cellprocessing_default_button = tk.Button(self.parameters_panel, text="Default Parameters",
                                                       command=self.load_default_params_cell_computation)
        self.cellprocessing_default_button.pack(side="top", fill="x")

        self.cell_info_frame = tk.Frame(self.images_frame)
        self.cell_info_frame.pack(side="bottom", fill="x")

        self.cellid_frame = tk.Frame(self.cell_info_frame)
        self.cellid_frame.pack(side="top", fill="x")
        self.cellid_label = tk.Label(self.cellid_frame, text="Cell ID: ")
        self.cellid_label.pack(side="left")
        self.cellid_value = tk.IntVar()
        self.cellid_value_label = tk.Label(
            self.cellid_frame, textvariable=self.cellid_value)
        self.cellid_value_label.pack(side="left")

        self.merged_with_frame = tk.Frame(self.cell_info_frame)
        self.merged_with_frame.pack(side="top", fill="x")
        self.merged_with_label = tk.Label(
            self.merged_with_frame, text="Merged Cell: ")
        self.merged_with_label.pack(side="left")
        self.merged_with_value = tk.StringVar()
        self.merged_with_value_label = tk.Label(
            self.merged_with_frame, textvariable=self.merged_with_value)
        self.merged_with_value_label.pack(side="left")

        self.marked_as_noise_frame = tk.Frame(self.cell_info_frame)
        self.marked_as_noise_frame.pack(side="top", fill="x")
        self.marked_as_noise_label = tk.Label(
            self.marked_as_noise_frame, text="Marked as Noise: ")
        self.marked_as_noise_label.pack(side="left")
        self.marked_as_noise_value = tk.StringVar()
        self.marked_as_noise_value_label = tk.Label(
            self.marked_as_noise_frame, textvariable=self.marked_as_noise_value)
        self.marked_as_noise_value_label.pack(side="left")

        units = self.ehooke.parameters.imageloaderparams.units

        self.area_frame = tk.Frame(self.cell_info_frame)
        self.area_frame.pack(side="top", fill="x")
        self.area_label = tk.Label(self.area_frame, text="Area (" + units + "\u00b2): ")
        self.area_label.pack(side="left")
        self.area_value = tk.DoubleVar()
        self.area_value_label = tk.Label(
            self.area_frame, textvariable=self.area_value)
        self.area_value_label.pack(side="left")

        self.perimeter_frame = tk.Frame(self.cell_info_frame)
        self.perimeter_frame.pack(side="top", fill="x")
        self.perimeter_label = tk.Label(
            self.perimeter_frame, text="Perimeter (" + units + "): ")
        self.perimeter_label.pack(side="left")
        self.perimeter_value = tk.DoubleVar()
        self.perimeter_value_label = tk.Label(
            self.perimeter_frame, textvariable=self.perimeter_value)
        self.perimeter_value_label.pack(side="left")

        self.length_frame = tk.Frame(self.cell_info_frame)
        self.length_frame.pack(side="top", fill="x")
        self.length_label = tk.Label(self.length_frame, text="Length (" + units + "): ")
        self.length_label.pack(side="left")
        self.length_value = tk.DoubleVar()
        self.length_value_label = tk.Label(
            self.length_frame, textvariable=self.length_value)
        self.length_value_label.pack(side="left")

        self.width_frame = tk.Frame(self.cell_info_frame)
        self.width_frame.pack(side="top", fill="x")
        self.width_label = tk.Label(self.width_frame, text="Width (" + units + "): ")
        self.width_label.pack(side="left")
        self.width_value = tk.DoubleVar()
        self.width_value_label = tk.Label(
            self.width_frame, textvariable=self.width_value)
        self.width_value_label.pack(side="left")

        self.eccentricity_frame = tk.Frame(self.cell_info_frame)
        self.eccentricity_frame.pack(side="top", fill="x")
        self.eccentricity_label = tk.Label(
            self.eccentricity_frame, text="Eccentricity (" + units + "): ")
        self.eccentricity_label.pack(side="left")
        self.eccentricity_value = tk.DoubleVar()
        self.eccentricity_value_label = tk.Label(
            self.eccentricity_frame, textvariable=self.eccentricity_value)
        self.eccentricity_value_label.pack(side="left")

        self.irregularity_frame = tk.Frame(self.cell_info_frame)
        self.irregularity_frame.pack(side="top", fill="x")
        self.irregularity_label = tk.Label(
            self.irregularity_frame, text="Irregularity (" + units + "): ")
        self.irregularity_label.pack(side="left")
        self.irregularity_value = tk.DoubleVar()
        self.irregularity_value_label = tk.Label(
            self.irregularity_frame, textvariable=self.irregularity_value)
        self.irregularity_value_label.pack(side="left")

        self.neighbours_frame = tk.Frame(self.cell_info_frame)
        self.neighbours_frame.pack(side="top", fill="x")
        self.neighbours_label = tk.Label(
            self.neighbours_frame, text="Neighbours: ")
        self.neighbours_label.pack(side="left")
        self.neighbours_value = tk.IntVar()
        self.neighbours_value_label = tk.Label(
            self.neighbours_frame, textvariable=self.neighbours_value)
        self.neighbours_value_label.pack(side="left")

        self.empty_label = tk.Label(self.cell_info_frame)
        self.empty_label.pack(side="top")

        self.base_button = tk.Button(self.images_frame, text="Base", command=lambda: self.show_image("Base"),
                                     width=self.image_buttons_width)
        self.base_button.pack(side="top", fill="x")
        self.base_button.config(state="active")

        self.base_features_button = tk.Button(self.images_frame, text="Base with Features",
                                              command=lambda: self.show_image(
                                                  "Base_features"),
                                              width=self.image_buttons_width)
        self.base_features_button.pack(side="top", fill="x")

        self.base_w_cells_button = tk.Button(self.images_frame, text="Base Outlined",
                                             command=lambda: self.show_image("Base_cells_outlined"),
                                             width=self.image_buttons_width)
        self.base_w_cells_button.pack(side="top", fill="x")
        self.base_w_cells_button.config(state="disabled")

        self.fluor_button = tk.Button(self.images_frame, text="Fluorescence", command=lambda: self.show_image("Fluor"),
                                      width=self.image_buttons_width)
        self.fluor_button.pack(side="top", fill="x")

        self.fluor_cells_out_button = tk.Button(self.images_frame, text="Fluorescence Outlined",
                                                command=lambda: self.show_image(
                                                    "Fluor_cells_outlined"),
                                                width=self.image_buttons_width)
        self.fluor_cells_out_button.pack(side="top", fill="x")
        self.fluor_cells_out_button.config(state="disabled")

        self.optional_button = tk.Button(self.images_frame, text="Secondary Channel",
                                         command=lambda: self.show_image(
                                             "Optional"),
                                         width=self.image_buttons_width)
        self.optional_button.pack(side="top", fill="x")
        if self.ehooke.image_manager.optional_image is None:
            self.optional_button.config(state="disabled")
        else:
            self.optional_button.config(state="active")

        self.current_image_label = tk.Label(self.images_frame, text="")
        self.current_image_label.pack(side="top")

        self.config_gui(self.main_window)

        self.cellcomputation_parameters_label.config(font=self.gui_font_bold)

        for w in self.cell_info_frame.winfo_children():
            for w1 in w.winfo_children():
                w1.config(font=("Verdana", 8, "normal"))

        self.show_image(self.current_image)

    def set_cellcomputation_from_cellprocessing(self):
        """Method to go back to cell computation"""
        self.canvas.mpl_disconnect(self.cid)
        self.event_connected = False

        self.set_cellcomputation()

    def on_press(self, event):
        if event.button == 3:

            label = str(int(self.ehooke.cell_manager.merged_labels[
                                int(event.ydata), int(event.xdata)]))

            if int(label) > 0:

                self.ehooke.cell_manager.cells[label].selection_state *= -1
                self.ehooke.cell_manager.overlay_cells(
                    self.ehooke.image_manager)

                self.images[
                    "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells

                self.images[
                    "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

                if self.ehooke.image_manager.optional_image is not None:
                    self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

                self.show_image(self.current_image)

    def show_cell_info_cellprocessing(self, x, y):
        """Shows the stats of each cell (including fluor stats) on the side
        panel"""
        """Shows the stats of each cell on the side panel"""
        label = int(self.ehooke.cell_manager.merged_labels[int(y), int(x)])

        if 0 < label:
            stats = self.ehooke.cell_manager.cells[str(label)].stats

            self.cellid_value.set(label)
            self.merged_with_value.set(self.ehooke.cell_manager.cells[
                                           str(label)].merged_with)
            self.cell_cycle_phase_value.set(str(stats["Cell Cycle Phase"]))
            self.marked_as_noise_value.set(self.ehooke.cell_manager.cells[
                                               str(label)].marked_as_noise)
            self.area_value.set(float(str(stats["Area"])[0:7]))
            self.perimeter_value.set(float(str(stats["Perimeter"])[0:7]))
            self.length_value.set(float(str(stats["Length"])[0:7]))
            self.width_value.set(float(str(stats["Width"])[0:7]))
            self.eccentricity_value.set(float(str(stats["Eccentricity"])[0:7]))
            self.irregularity_value.set(float(str(stats["Irregularity"])[0:7]))
            self.neighbours_value.set(stats["Neighbours"])
            self.baseline_value.set(float(str(stats["Baseline"])[0:7]))
            self.cellmedian_value.set(float(str(stats["Cell Median"])[0:7]))
            self.permedian_value.set(float(str(stats["Membrane Median"])[0:7]))
            self.septmedian_value.set(float(str(stats["Septum Median"])[0:7]))
            self.cytomedian_value.set(
                float(str(stats["Cytoplasm Median"])[0:7]))
            self.fr_value.set(float(str(stats["Fluor Ratio"])[0:7]))
            self.fr75_value.set(float(str(stats["Fluor Ratio 75%"])[0:7]))
            self.fr25_value.set(float(str(stats["Fluor Ratio 25%"])[0:7]))
            self.fr10_value.set(float(str(stats["Fluor Ratio 10%"])[0:7]))

        else:
            self.cellid_value.set(0)
            self.merged_with_value.set("No")
            self.cell_cycle_phase_value.set("0")
            self.marked_as_noise_value.set("No")
            self.area_value.set(0)
            self.perimeter_value.set(0)
            self.length_value.set(0)
            self.width_value.set(0)
            self.eccentricity_value.set(0)
            self.irregularity_value.set(0)
            self.neighbours_value.set(0)
            self.baseline_value.set(0)
            self.cellmedian_value.set(0)
            self.permedian_value.set(0)
            self.septmedian_value.set(0)
            self.cytomedian_value.set(0)
            self.fr_value.set(0)
            self.fr75_value.set(0)
            self.fr25_value.set(0)
            self.fr10_value.set(0)

        lum = self.ehooke.image_manager.original_fluor_image[int(y), int(x)]
        return "Luminance: " + str(lum)

    def process_cells(self):
        """Method used to process the individual regions of each cell
        aswell as their fluor stats"""
        self.ehooke.parameters.cellprocessingparams.classify_cells = self.classify_cells_checkbox_value.get()
        self.ehooke.parameters.cellprocessingparams.secondary_channel = self.secondary_channel_checkbox_value.get()
        self.ehooke.parameters.cellprocessingparams.microscope = self.microscope_value.get()
        self.ehooke.parameters.cellprocessingparams.heatmap = self.heatmap_checkbox_value.get()
        septum_option = self.find_septum_menu_value.get()
        if septum_option == "Closed":
            self.ehooke.parameters.cellprocessingparams.find_septum = True
        elif septum_option == "Closed+Open":
            self.ehooke.parameters.cellprocessingparams.find_openseptum = True
            self.ehooke.parameters.cellprocessingparams.find_septum = False
        elif septum_option == "No":
            self.ehooke.parameters.cellprocessingparams.find_openseptum = False
            self.ehooke.parameters.cellprocessingparams.find_septum = False
        look_for_septum_in = self.look_for_septum_in_menu_value.get()
        if look_for_septum_in == "Base":
            self.ehooke.parameters.cellprocessingparams.look_for_septum_in_base = True
        elif look_for_septum_in == "Secondary":
            self.ehooke.parameters.cellprocessingparams.look_for_septum_in_optional = True

        self.ehooke.parameters.cellprocessingparams.septum_algorithm = self.septum_algorithm_value.get()
        self.ehooke.parameters.cellprocessingparams.inner_mask_thickness = self.membrane_thickness_value.get()
        self.ehooke.parameters.cellprocessingparams.signal_ratio = self.optional_signal_ratio_value.get()
        self.status.set("Processing cells...")
        self.ehooke.process_cells()

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells

        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells

        if self.ehooke.image_manager.optional_image is not None:
            self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells
            self.optional_w_cells_button.config(state="active")
            self.select_optional_button.config(state="active")

        self.images["Fluor_with_lines"] = self.ehooke.image_manager.fluor_image

        self.show_image(self.current_image)

        self.filter_cells_button.config(state="active")
        self.generate_report_button.config(state="active")
        self.compute_coloc_button.config(state="active")
        self.select_all_button.config(state="active")
        self.unselect_all_button.config(state="active")
        self.invert_selection_button.config(state="active")
        self.select_from_file_button.config(state="active")
        self.add_line_button.config(state="active")
        self.remove_line_button.config(state="active")
        self.status.set("Cell Processing Finished. Use right-click to select/unselect cells")

        if self.classify_cells_checkbox_value.get():
            self.select_phase1_button.config(state="active")
            self.select_phase2_button.config(state="active")
            self.select_phase3_button.config(state="active")
            self.assign_phase1_button.config(state="active")
            self.assign_phase2_button.config(state="active")
            self.assign_phase3_button.config(state="active")
        else:
            self.select_phase1_button.config(state="disabled")
            self.select_phase2_button.config(state="disabled")
            self.select_phase3_button.config(state="disabled")
            self.assign_phase1_button.config(state="disabled")
            self.assign_phase2_button.config(state="disabled")
            self.assign_phase3_button.config(state="disabled")

        self.current_step = "CellsProcessed"

    def select_all_cells(self):
        """Method used to mark all cells as selected"""
        self.ehooke.select_all_cells()

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells
        self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

        self.show_image(self.current_image)

    def reject_all_cells(self):
        """Method used to mark all cells as rejected"""
        self.ehooke.reject_all_cells()

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells
        self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

        self.show_image(self.current_image)

    def invert_selection(self):
        self.ehooke.invert_selection()

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells
        self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

        self.show_image(self.current_image)

    def select_from_file(self):
        """Loads a file generated by cyphID with a list of cells to be selected"""

        self.ehooke.select_from_file()

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells
        self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

        self.show_image(self.current_image)

    def select_cells_phase(self, phase):

        self.ehooke.select_cells_phase(phase)

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells
        self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

        self.show_image(self.current_image)

    def change_cell_stat(self, label_c1):
        """Method used to change the state of a cell"""
        self.ehooke.cell_manager.cells[str(label_c1)].selection_state *= -1

    def filter_cells(self):
        """Method used to filter the cells based on a set of params"""
        filters = []

        if self.areafilter_checkbox_value.get():
            filters.extend(
                [("Area", self.areafilter_min_value.get(), self.areafilter_max_value.get())])

        if self.perimeterfilter_checkbox_value.get():
            filters.extend([("Perimeter", self.perimeterfilter_min_value.get(
            ), self.perimeterfilter_max_value.get())])

        if self.eccentricityfilter_checkbox_value.get():
            filters.extend([("Eccentricity", self.eccentricityfilter_min_value.get(),
                             self.eccentricityfilter_max_value.get())])

        if self.irregularityfilter_checkbox_value.get():
            filters.extend([("Irregularity", self.irregularityfilter_min_value.get(),
                             self.irregularityfilter_max_value.get())])

        if self.neighboursfilter_checkbox_value.get():
            filters.extend([("Neighbours", self.neighboursfilter_min_value.get(),
                             self.neighboursfilter_max_value.get())])

        self.ehooke.parameters.cellprocessingparams.cell_filters = filters
        self.ehooke.filter_cells()

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells
        self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

        self.show_image(self.current_image)

    def generate_report(self):
        """Method used to save a report with the cell stats"""
        self.ehooke.generate_reports()

    def compute_pcc(self):
        self.ehooke.compute_coloc()

    def draw_line(self, event):
        if event.button == 3:
            if len(self.points) < 2:
                self.points.append((int(event.ydata), int(event.xdata)))
            else:
                self.points.append((int(event.ydata), int(event.xdata)))

                self.canvas.mpl_disconnect(self.cid)
                self.cid = self.canvas.mpl_connect(
                    'button_release_event', self.on_press)
                self.ehooke.linescan_manager.add_line(self.points[0],
                                                      self.points[1],
                                                      self.points[2])

                self.ehooke.linescan_manager.overlay_lines_on_image(
                    self.ehooke.image_manager.fluor_image)
                self.images[
                    "Fluor_with_lines"] = self.ehooke.linescan_manager.fluor_w_lines
                self.show_image("Fluor_with_lines")

    def add_line_linescan(self):
        # add line code

        self.points = []
        self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect(
            'button_release_event', self.draw_line)

    def remove_line_linescan(self):
        # remove line code

        self.ehooke.linescan_manager.remove_line()

        self.ehooke.linescan_manager.overlay_lines_on_image(
            self.ehooke.image_manager.fluor_image)
        self.images[
            "Fluor_with_lines"] = self.ehooke.linescan_manager.fluor_w_lines
        self.show_image("Fluor_with_lines")

    def select_optional_signal(self):

        self.ehooke.parameters.cellprocessingparams.signal_ratio = self.optional_signal_ratio_value.get()
        signal_ratio = self.ehooke.parameters.cellprocessingparams.signal_ratio
        self.ehooke.select_cells_optional(signal_ratio)

        self.images[
            "Fluor_cells_outlined"] = self.ehooke.cell_manager.fluor_w_cells
        self.images[
            "Base_cells_outlined"] = self.ehooke.cell_manager.base_w_cells
        self.images["Optional_cells_outlined"] = self.ehooke.cell_manager.optional_w_cells

        self.show_image(self.current_image)

    def phase1_on_press(self, event):
        if event.button == 3:
            label = int(self.ehooke.cell_manager.merged_labels[
                            int(event.ydata), int(event.xdata)])

            if label > 0:
                self.ehooke.cell_manager.cells[str(label)].stats["Cell Cycle Phase"] = 1
            self.canvas.mpl_disconnect(self.cid)
            self.cid = self.canvas.mpl_connect('button_release_event',
                                               self.on_press)

    def phase2_on_press(self, event):
        if event.button == 3:
            label = int(self.ehooke.cell_manager.merged_labels[
                            int(event.ydata), int(event.xdata)])

            if label > 0:
                self.ehooke.cell_manager.cells[str(label)].stats["Cell Cycle Phase"] = 2
            self.canvas.mpl_disconnect(self.cid)
            self.cid = self.canvas.mpl_connect('button_release_event',
                                               self.on_press)

    def phase3_on_press(self, event):
        if event.button == 3:
            label = int(self.ehooke.cell_manager.merged_labels[
                            int(event.ydata), int(event.xdata)])

            if label > 0:
                self.ehooke.cell_manager.cells[str(label)].stats["Cell Cycle Phase"] = 3
            self.canvas.mpl_disconnect(self.cid)
            self.cid = self.canvas.mpl_connect('button_release_event',
                                               self.on_press)

    def assign_cell_cycle_phase(self, phase):
        if self.event_connected:
            self.canvas.mpl_disconnect(self.cid)
        self.status.set("Select Cell for cell cycle phase assignment")
        if phase == 1:
            self.cid = self.canvas.mpl_connect('button_release_event',
                                               self.phase1_on_press)
        elif phase == 2:
            self.cid = self.canvas.mpl_connect('button_release_event',
                                               self.phase2_on_press)
        elif phase == 3:
            self.cid = self.canvas.mpl_connect('button_release_event',
                                               self.phase3_on_press)
        self.event_connected = True

    def check_filter_params(self):

        for flt in self.ehooke.parameters.cellprocessingparams.cell_filters:
            if flt[0] == "Area":
                self.areafilter_checkbox_value.set(True)
                self.areafilter_min_value.set(int(flt[1]))
                self.areafilter_max_value.set(int(flt[2]))
            elif flt[0] == "Perimeter":
                self.perimeterfilter_checkbox_value.set(True)
                self.perimeterfilter_min_value.set(int(flt[1]))
                self.perimeterfilter_max_value.set(int(flt[2]))
            elif flt[0] == "Eccentricity":
                self.eccentricityfilter_checkbox_value.set(True)
                self.eccentricityfilter_min_value.set(float(flt[1]))
                self.eccentricityfilter_max_value.set(float(flt[2]))
            elif flt[0] == "Irregularity":
                self.irregularityfilter_checkbox_value.set(True)
                self.irregularityfilter_min_value.set(float(flt[1]))
                self.irregularityfilter_max_value.set(float(flt[2]))
            elif flt[0] == "Neighbours":
                self.neighboursfilter_checkbox_value.set(True)
                self.neighboursfilter_min_value.set(int(flt[1]))
                self.neighboursfilter_max_value.set(int(flt[2]))

    def set_cellprocessing(self):
        """Method used to change the interface to the Cell Processing
        Step"""
        self.ax.axis("off")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        self.show_image("Fluor_cells_outlined")
        self.ax.format_coord = self.show_cell_info_cellprocessing
        self.canvas.draw()

        self.current_step = "CellProcessing"

        self.cid = self.canvas.mpl_connect(
            'button_release_event', self.on_press)
        self.event_connected = True

        for w in self.top_frame.winfo_children():
            w.destroy()

        for w in self.parameters_panel.winfo_children():
            w.destroy()

        for w in self.images_frame.winfo_children():
            w.destroy()

        self.status = tk.StringVar()
        self.status.set("Load Phase Image")
        self.status_bar = tk.Label(
            self.parameters_panel, textvariable=self.status, wraplength=self.status_length)
        self.status_bar.pack(side="bottom")
        self.status.set("Waiting Cell Processing")

        self.process_cells_button = tk.Button(
            self.top_frame, text="Process", command=self.process_cells)
        self.process_cells_button.pack(side="left")

        self.compute_coloc_button = tk.Button(
            self.top_frame, text="PCC", command=self.compute_pcc)
        self.compute_coloc_button.pack(side="left")
        self.compute_coloc_button.config(state="disabled")

        self.select_all_button = tk.Button(self.top_frame, text="Select All",
                                           command=self.select_all_cells)
        self.select_all_button.pack(side="left", fill="x")
        self.select_all_button.config(state="disabled")

        self.unselect_all_button = tk.Button(self.top_frame, text="Reject All",
                                             command=self.reject_all_cells)
        self.unselect_all_button.pack(side="left", fill="x")
        self.unselect_all_button.config(state="disabled")

        self.filter_cells_button = tk.Button(
            self.top_frame, text="Apply Filters", command=self.filter_cells)
        self.filter_cells_button.pack(side="left")
        self.filter_cells_button.config(state="disabled")

        self.invert_selection_button = tk.Button(self.top_frame, text="Invert Selection",
                                                 command=self.invert_selection)
        self.invert_selection_button.pack(side="left", fill="x")
        self.invert_selection_button.config(state="disabled")

        self.select_from_file_button = tk.Button(
            self.top_frame, text="Load Selection", command=self.select_from_file)
        self.select_from_file_button.pack(side="left", fill="x")
        self.select_from_file_button.config(state="disabled")

        self.select_optional_button = tk.Button(self.top_frame, text="Select Cells with Secondary Signal",
                                                command=self.select_optional_signal)
        self.select_optional_button.pack(side="left", fill="x")
        self.select_optional_button.config(state="disabled")

        self.optional_ratio_label = tk.Label(self.top_frame, text="Secondary Signal Ratio:")
        self.optional_ratio_label.pack(side="left", fill="x")

        self.optional_signal_ratio_value = tk.DoubleVar()
        self.optionalfilter_min_entry = tk.Entry(
            self.top_frame, textvariable=self.optional_signal_ratio_value, width=5)
        self.optionalfilter_min_entry.pack(side="left")
        self.optional_signal_ratio_value.set(self.default_params.cellprocessingparams.signal_ratio)

        self.generate_report_button = tk.Button(
            self.top_frame, text="Save Report", command=self.generate_report)
        self.generate_report_button.pack(side="right")
        self.generate_report_button.config(state="disabled")

        self.back_button = tk.Button(
            self.top_frame, text="Back", command=self.set_cellcomputation_from_cellprocessing)
        self.back_button.pack(side="right")

        self.new_analysis_button = tk.Button(
            self.top_frame, text="New", command=self.new_analysis)
        self.new_analysis_button.pack(side="right")

        self.cellprocessing_label = tk.Label(
            self.parameters_panel, text="Cell Processing Parameters: ")
        self.cellprocessing_label.config(width=self.parameters_panel_width)
        self.cellprocessing_label.pack(side="top", pady=self.pady)

        self.microscope_frame = tk.Frame(self.parameters_panel)
        self.microscope_frame.pack(side="top", fill="x")
        self.microscope_label = tk.Label(
            self.microscope_frame, text="Microscope: ")
        self.microscope_label.pack(side="left")
        self.microscope_value = tk.StringVar()
        self.microscope_menu = tk.OptionMenu(self.microscope_frame, self.microscope_value,
                                             'Epifluorescence', 'SIM')
        self.microscope_menu.pack(side="left")
        self.microscope_value.set(self.ehooke.parameters.cellprocessingparams.microscope)

        self.classify_cells_frame = tk.Frame(self.parameters_panel)
        self.classify_cells_frame.pack(side="top", fill="x")
        self.classify_cells_label = tk.Label(
            self.classify_cells_frame, text="Classify Cell Cycle Phase: ")
        self.classify_cells_label.pack(side="left")
        self.classify_cells_checkbox_value = tk.BooleanVar()
        self.classify_cells_checkbox = tk.Checkbutton(self.classify_cells_frame,
                                                      variable=self.classify_cells_checkbox_value,
                                                      onvalue=True, offvalue=False)
        self.classify_cells_checkbox_value.set(self.ehooke.parameters.cellprocessingparams.classify_cells)
        self.classify_cells_checkbox.pack(side="left")

        self.secondary_channel_frame = tk.Frame(self.parameters_panel)
        self.secondary_channel_frame.pack(side="top", fill="x")
        self.secondary_channel_label = tk.Label(self.secondary_channel_frame, text="Use Secondary to classify: ")
        self.secondary_channel_label.pack(side="left")
        self.secondary_channel_checkbox_value = tk.BooleanVar()
        self.secondary_channel_checkbox = tk.Checkbutton(self.secondary_channel_frame,
                                                         variable=self.secondary_channel_checkbox_value,
                                                         onvalue=True, offvalue=False)
        self.secondary_channel_checkbox_value.set(self.ehooke.parameters.cellprocessingparams.secondary_channel)
        self.secondary_channel_checkbox.pack(side="left")

        self.heatmap_frame = tk.Frame(self.parameters_panel)
        self.heatmap_frame.pack(side="top", fill="x")
        self.heatmap_label = tk.Label(self.heatmap_frame, text="Compute fluorescence heatmap: ")
        self.heatmap_label.pack(side="left")
        self.heatmap_checkbox_value = tk.BooleanVar()
        self.heatmap_checkbox = tk.Checkbutton(self.heatmap_frame, variable=self.heatmap_checkbox_value,
                                               onvalue=True, offvalue=False)
        self.heatmap_checkbox_value.set(self.ehooke.parameters.cellprocessingparams.heatmap)
        self.heatmap_checkbox.pack(side="left")

        self.find_septum_frame = tk.Frame(self.parameters_panel)
        self.find_septum_frame.pack(side="top", fill="x")
        self.find_septum_label = tk.Label(
            self.find_septum_frame, text="Find Septa: ")
        self.find_septum_label.pack(side="left")
        self.find_septum_menu_value = tk.StringVar()
        self.find_septum_menu = tk.OptionMenu(self.find_septum_frame, self.find_septum_menu_value,
                                              "No", "Closed", "Closed+Open")
        if self.ehooke.parameters.cellprocessingparams.find_septum:
            self.find_septum_menu_value.set("Closed")
        elif self.ehooke.parameters.cellprocessingparams.find_openseptum:
            self.find_septum_menu_value.set("Closed+Open")
        else:
            self.find_septum_menu_value.set("No")
        self.find_septum_menu.pack(side="left")

        self.look_for_septum_in_frame = tk.Frame(self.parameters_panel)
        self.look_for_septum_in_frame.pack(side="top", fill="x")
        self.look_for_septum_in_label = tk.Label(
            self.look_for_septum_in_frame, text="Look for Septa in: ")
        self.look_for_septum_in_label.pack(side="left")
        self.look_for_septum_in_menu_value = tk.StringVar()
        if self.ehooke.image_manager.optional_image is None:
            self.look_for_septum_in_menu = tk.OptionMenu(self.look_for_septum_in_frame,
                                                         self.look_for_septum_in_menu_value,
                                                         "Fluorescence", "Base")
        else:
            self.look_for_septum_in_menu = tk.OptionMenu(self.look_for_septum_in_frame,
                                                         self.look_for_septum_in_menu_value,
                                                         "Fluorescence", "Base", "Secondary")

        if self.ehooke.parameters.cellprocessingparams.look_for_septum_in_base:
            self.look_for_septum_in_menu_value.set("Base")
        elif self.ehooke.parameters.cellprocessingparams.look_for_septum_in_optional:
            self.look_for_septum_in_menu_value.set("Secondary")
        else:
            self.look_for_septum_in_menu_value.set("Fluorescence")
        self.look_for_septum_in_menu.pack(side="left")

        self.septum_algorithm_frame = tk.Frame(self.parameters_panel)
        self.septum_algorithm_frame.pack(side="top", fill="x")
        self.septum_algorithm_label = tk.Label(
            self.septum_algorithm_frame, text="Septum Algorithm: ")
        self.septum_algorithm_label.pack(side="left")
        self.septum_algorithm_value = tk.StringVar()
        self.septum_algorithm_menu = tk.OptionMenu(self.septum_algorithm_frame, self.septum_algorithm_value,
                                                   'Box', 'Isodata')
        self.septum_algorithm_menu.pack(side="left")
        self.septum_algorithm_value.set(self.ehooke.parameters.cellprocessingparams.septum_algorithm)

        self.membrane_thickness_frame = tk.Frame(self.parameters_panel)
        self.membrane_thickness_frame.pack(side="top", fill="x")
        self.membrane_thickness_label = tk.Label(
            self.membrane_thickness_frame, text="Membrane Thickness: ")
        self.membrane_thickness_label.pack(side="top")

        self.membrane_thickness_value = tk.Scale(self.membrane_thickness_frame, from_=4, to=8, orient="horizontal")
        self.membrane_thickness_value.pack()
        self.membrane_thickness_value.set(self.ehooke.parameters.cellprocessingparams.inner_mask_thickness)

        self.filters_label = tk.Label(
            self.parameters_panel, text="Cell Filters: ")
        self.filters_label.pack(side="top", pady=self.pady)

        self.areafilter_frame = tk.Frame(self.parameters_panel)
        self.areafilter_frame.pack(side="top", fill="x")
        self.areafilter_checkbox_value = tk.BooleanVar()
        self.areafilter_checkbox = tk.Checkbutton(self.areafilter_frame, variable=self.areafilter_checkbox_value,
                                                  onvalue=True, offvalue=False)
        self.areafilter_checkbox_value.set(False)
        self.areafilter_checkbox.pack(side="left")
        self.areafilter_label = tk.Label(self.areafilter_frame, text="Area: ")
        self.areafilter_label.pack(side="left")
        self.areafilter_min_value = tk.IntVar()
        self.areafilter_max_value = tk.IntVar()
        self.areafilter_min_entry = tk.Entry(
            self.areafilter_frame, textvariable=self.areafilter_min_value, width=5)
        self.areafilter_min_entry.pack(side="left")
        self.areafilter_max_entry = tk.Entry(
            self.areafilter_frame, textvariable=self.areafilter_max_value, width=5)
        self.areafilter_max_entry.pack(side="left")
        self.areafilter_min_value.set(0)
        self.areafilter_max_value.set(1000)

        self.perimeterfilter_frame = tk.Frame(self.parameters_panel)
        self.perimeterfilter_frame.pack(side="top", fill="x")
        self.perimeterfilter_checkbox_value = tk.BooleanVar()
        self.perimeterfilter_checkbox = tk.Checkbutton(self.perimeterfilter_frame,
                                                       variable=self.perimeterfilter_checkbox_value,
                                                       onvalue=True, offvalue=False)
        self.perimeterfilter_checkbox_value.set(False)
        self.perimeterfilter_checkbox.pack(side="left")
        self.perimeterfilter_label = tk.Label(
            self.perimeterfilter_frame, text="Perimeter: ")
        self.perimeterfilter_label.pack(side="left")
        self.perimeterfilter_min_value = tk.IntVar()
        self.perimeterfilter_max_value = tk.IntVar()
        self.perimeterfilter_min_entry = tk.Entry(self.perimeterfilter_frame,
                                                  textvariable=self.perimeterfilter_min_value, width=5)
        self.perimeterfilter_min_entry.pack(side="left")
        self.perimeterfilter_max_entry = tk.Entry(self.perimeterfilter_frame,
                                                  textvariable=self.perimeterfilter_max_value, width=5)
        self.perimeterfilter_max_entry.pack(side="left")
        self.perimeterfilter_min_value.set(0)
        self.perimeterfilter_max_value.set(500)

        self.eccentricityfilter_frame = tk.Frame(self.parameters_panel)
        self.eccentricityfilter_frame.pack(side="top", fill="x")
        self.eccentricityfilter_checkbox_value = tk.BooleanVar()
        self.eccentricityfilter_checkbox = tk.Checkbutton(self.eccentricityfilter_frame,
                                                          variable=self.eccentricityfilter_checkbox_value,
                                                          onvalue=True, offvalue=False)
        self.eccentricityfilter_checkbox_value.set(False)
        self.eccentricityfilter_checkbox.pack(side="left")
        self.eccentricityfilter_label = tk.Label(
            self.eccentricityfilter_frame, text="Eccentricity: ")
        self.eccentricityfilter_label.pack(side="left")
        self.eccentricityfilter_min_value = tk.DoubleVar()
        self.eccentricityfilter_max_value = tk.DoubleVar()
        self.eccentricityfilter_min_entry = tk.Entry(self.eccentricityfilter_frame,
                                                     textvariable=self.eccentricityfilter_min_value, width=5)
        self.eccentricityfilter_min_entry.pack(side="left")
        self.eccentricityfilter_max_entry = tk.Entry(self.eccentricityfilter_frame,
                                                     textvariable=self.eccentricityfilter_max_value, width=5)
        self.eccentricityfilter_max_entry.pack(side="left")
        self.eccentricityfilter_min_value.set(0)
        self.eccentricityfilter_max_value.set(1)

        self.irregularityfilter_frame = tk.Frame(self.parameters_panel)
        self.irregularityfilter_frame.pack(side="top", fill="x")
        self.irregularityfilter_checkbox_value = tk.BooleanVar()
        self.irregularityfilter_checkbox = tk.Checkbutton(self.irregularityfilter_frame,
                                                          variable=self.irregularityfilter_checkbox_value,
                                                          onvalue=True, offvalue=False)
        self.irregularityfilter_checkbox_value.set(False)
        self.irregularityfilter_checkbox.pack(side="left")
        self.irregularityfilter_label = tk.Label(
            self.irregularityfilter_frame, text="Irregularity: ")
        self.irregularityfilter_label.pack(side="left")
        self.irregularityfilter_min_value = tk.DoubleVar()
        self.irregularityfilter_max_value = tk.DoubleVar()
        self.irregularityfilter_min_entry = tk.Entry(self.irregularityfilter_frame,
                                                     textvariable=self.irregularityfilter_min_value, width=5)
        self.irregularityfilter_min_entry.pack(side="left")
        self.irregularityfilter_max_entry = tk.Entry(self.irregularityfilter_frame,
                                                     textvariable=self.irregularityfilter_max_value, width=5)
        self.irregularityfilter_max_entry.pack(side="left")
        self.irregularityfilter_min_value.set(0)
        self.irregularityfilter_max_value.set(20)

        self.neighboursfilter_frame = tk.Frame(self.parameters_panel)
        self.neighboursfilter_frame.pack(side="top", fill="x")
        self.neighboursfilter_checkbox_value = tk.BooleanVar()
        self.neighboursfilter_checkbox = tk.Checkbutton(self.neighboursfilter_frame,
                                                        variable=self.neighboursfilter_checkbox_value,
                                                        onvalue=True, offvalue=False)
        self.neighboursfilter_checkbox_value.set(False)
        self.neighboursfilter_checkbox.pack(side="left")
        self.neighboursfilter_label = tk.Label(
            self.neighboursfilter_frame, text="Neighbours: ")
        self.neighboursfilter_label.pack(side="left")
        self.neighboursfilter_min_value = tk.IntVar()
        self.neighboursfilter_max_value = tk.IntVar()
        self.neighboursfilter_min_entry = tk.Entry(self.neighboursfilter_frame,
                                                   textvariable=self.neighboursfilter_min_value, width=5)
        self.neighboursfilter_min_entry.pack(side="left")
        self.neighboursfilter_max_entry = tk.Entry(self.neighboursfilter_frame,
                                                   textvariable=self.neighboursfilter_max_value, width=5)
        self.neighboursfilter_max_entry.pack(side="left")
        self.neighboursfilter_min_value.set(0)
        self.neighboursfilter_max_value.set(10)

        self.check_filter_params()

        self.cellprocessing_default_button = tk.Button(self.parameters_panel, text="Default Parameters",
                                                       command=self.load_default_params_cell_processing)
        self.cellprocessing_default_button.pack(side="top", fill="x", pady=(5, 5))

        self.cell_cycle_label = tk.Label(self.parameters_panel, text="Cell Cycle Phase:")
        self.cell_cycle_label.pack(side="top", pady=self.pady)

        self.phase1_frame = tk.Frame(self.parameters_panel)
        self.phase1_frame.pack(side="top")
        self.select_phase1_button = tk.Button(
            self.phase1_frame, text="Select Phase 1 cells", command=lambda: self.select_cells_phase(1))
        self.select_phase1_button.pack(side="left", fill="x")
        self.select_phase1_button.config(state="disabled")
        self.assign_phase1_button = tk.Button(
            self.phase1_frame, text="Assign Phase 1", command=lambda: self.assign_cell_cycle_phase(1))
        self.assign_phase1_button.pack(side="left", fill="x")
        self.assign_phase1_button.config(state="disabled")

        self.phase2_frame = tk.Frame(self.parameters_panel)
        self.phase2_frame.pack(side="top")
        self.select_phase2_button = tk.Button(
            self.phase2_frame, text="Select Phase 2 cells", command=lambda: self.select_cells_phase(2))
        self.select_phase2_button.pack(side="left", fill="x")
        self.select_phase2_button.config(state="disabled")
        self.assign_phase2_button = tk.Button(
            self.phase2_frame, text="Assign Phase 2", command=lambda: self.assign_cell_cycle_phase(2))
        self.assign_phase2_button.pack(side="left", fill="x")
        self.assign_phase2_button.config(state="disabled")

        self.phase3_frame = tk.Frame(self.parameters_panel)
        self.phase3_frame.pack(side="top")
        self.select_phase3_button = tk.Button(
            self.phase3_frame, text="Select Phase 3 cells", command=lambda: self.select_cells_phase(3))
        self.select_phase3_button.pack(side="left", fill="x")
        self.select_phase3_button.config(state="disabled")
        self.assign_phase3_button = tk.Button(
            self.phase3_frame, text="Assign Phase 3", command=lambda: self.assign_cell_cycle_phase(3))
        self.assign_phase3_button.pack(side="left", fill="x")
        self.assign_phase3_button.config(state="disabled")

        self.linescan_label = tk.Label(self.parameters_panel, text="Linescan:")
        self.linescan_label.pack(side="top", pady=self.pady)

        self.add_line_button = tk.Button(self.parameters_panel, text="Add Line (L)",
                                         command=self.add_line_linescan)
        self.add_line_button.pack(side="top", fill="x")
        self.add_line_button.config(state="disabled")

        self.remove_line_button = tk.Button(self.parameters_panel, text="Undo Last Line (K)",
                                            command=self.remove_line_linescan)
        self.remove_line_button.pack(side="top", fill="x")
        self.remove_line_button.config(state="disabled")

        self.cell_info_frame = tk.Frame(self.images_frame)
        self.cell_info_frame.pack(side="bottom", fill="x")

        self.cellid_frame = tk.Frame(self.cell_info_frame)
        self.cellid_frame.pack(side="top", fill="x")
        self.cellid_label = tk.Label(self.cellid_frame, text="Cell ID: ")
        self.cellid_label.pack(side="left")
        self.cellid_value = tk.IntVar()
        self.cellid_value_label = tk.Label(
            self.cellid_frame, textvariable=self.cellid_value)
        self.cellid_value_label.pack(side="left")

        self.merged_with_frame = tk.Frame(self.cell_info_frame)
        self.merged_with_frame.pack(side="top", fill="x")
        self.merged_with_label = tk.Label(
            self.merged_with_frame, text="Merged Cell: ")
        self.merged_with_label.pack(side="left")
        self.merged_with_value = tk.StringVar()
        self.merged_with_value_label = tk.Label(
            self.merged_with_frame, textvariable=self.merged_with_value)
        self.merged_with_value_label.pack(side="left")

        self.marked_as_noise_frame = tk.Frame(self.cell_info_frame)
        self.marked_as_noise_frame.pack(side="top", fill="x")
        self.marked_as_noise_label = tk.Label(
            self.marked_as_noise_frame, text="Marked as Noise: ")
        self.marked_as_noise_label.pack(side="left")
        self.marked_as_noise_value = tk.StringVar()
        self.marked_as_noise_value_label = tk.Label(
            self.marked_as_noise_frame, textvariable=self.marked_as_noise_value)
        self.marked_as_noise_value_label.pack(side="left")

        self.cell_cycle_phase_frame = tk.Frame(self.cell_info_frame)
        self.cell_cycle_phase_frame.pack(side="top", fill="x")
        self.cell_cycle_phase_label = tk.Label(
            self.cell_cycle_phase_frame, text="Cell Cycle Phase: ")
        self.cell_cycle_phase_label.pack(side="left")
        self.cell_cycle_phase_value = tk.StringVar()
        self.cell_cycle_phase_value_label = tk.Label(
            self.cell_cycle_phase_frame, textvariable=self.cell_cycle_phase_value)
        self.cell_cycle_phase_value_label.pack(side="left")

        units = self.ehooke.parameters.imageloaderparams.units

        self.area_frame = tk.Frame(self.cell_info_frame)
        self.area_frame.pack(side="top", fill="x")
        self.area_label = tk.Label(self.area_frame, text="Area (" + units + "\u00b2): ")
        self.area_label.pack(side="left")
        self.area_value = tk.IntVar()
        self.area_value_label = tk.Label(
            self.area_frame, textvariable=self.area_value)
        self.area_value_label.pack(side="left")

        self.perimeter_frame = tk.Frame(self.cell_info_frame)
        self.perimeter_frame.pack(side="top", fill="x")
        self.perimeter_label = tk.Label(
            self.perimeter_frame, text="Perimeter (" + units + "): ")
        self.perimeter_label.pack(side="left")
        self.perimeter_value = tk.IntVar()
        self.perimeter_value_label = tk.Label(
            self.perimeter_frame, textvariable=self.perimeter_value)
        self.perimeter_value_label.pack(side="left")

        self.length_frame = tk.Frame(self.cell_info_frame)
        self.length_frame.pack(side="top", fill="x")
        self.length_label = tk.Label(self.length_frame, text="Length (" + units + "): ")
        self.length_label.pack(side="left")
        self.length_value = tk.IntVar()
        self.length_value_label = tk.Label(
            self.length_frame, textvariable=self.length_value)
        self.length_value_label.pack(side="left")

        self.width_frame = tk.Frame(self.cell_info_frame)
        self.width_frame.pack(side="top", fill="x")
        self.width_label = tk.Label(self.width_frame, text="Width (" + units + "): ")
        self.width_label.pack(side="left")
        self.width_value = tk.IntVar()
        self.width_value_label = tk.Label(
            self.width_frame, textvariable=self.width_value)
        self.width_value_label.pack(side="left")

        self.eccentricity_frame = tk.Frame(self.cell_info_frame)
        self.eccentricity_frame.pack(side="top", fill="x")
        self.eccentricity_label = tk.Label(
            self.eccentricity_frame, text="Eccentricity (" + units + "): ")
        self.eccentricity_label.pack(side="left")
        self.eccentricity_value = tk.IntVar()
        self.eccentricity_value_label = tk.Label(
            self.eccentricity_frame, textvariable=self.eccentricity_value)
        self.eccentricity_value_label.pack(side="left")

        self.irregularity_frame = tk.Frame(self.cell_info_frame)
        self.irregularity_frame.pack(side="top", fill="x")
        self.irregularity_label = tk.Label(
            self.irregularity_frame, text="Irregularity (" + units + "): ")
        self.irregularity_label.pack(side="left")
        self.irregularity_value = tk.IntVar()
        self.irregularity_value_label = tk.Label(
            self.irregularity_frame, textvariable=self.irregularity_value)
        self.irregularity_value_label.pack(side="left")

        self.neighbours_frame = tk.Frame(self.cell_info_frame)
        self.neighbours_frame.pack(side="top", fill="x")
        self.neighbours_label = tk.Label(
            self.neighbours_frame, text="Neighbours: ")
        self.neighbours_label.pack(side="left")
        self.neighbours_value = tk.IntVar()
        self.neighbours_value_label = tk.Label(
            self.neighbours_frame, textvariable=self.neighbours_value)
        self.neighbours_value_label.pack(side="left")

        self.baseline_frame = tk.Frame(self.cell_info_frame)
        self.baseline_frame.pack(side="top", fill="x")
        self.baseline_label = tk.Label(self.baseline_frame, text="Baseline: ")
        self.baseline_label.pack(side="left")
        self.baseline_value = tk.IntVar()
        self.baseline_value_label = tk.Label(
            self.baseline_frame, textvariable=self.baseline_value)
        self.baseline_value_label.pack(side="left")

        self.cellmedian_frame = tk.Frame(self.cell_info_frame)
        self.cellmedian_frame.pack(side="top", fill="x")
        self.cellmedian_label = tk.Label(
            self.cellmedian_frame, text="Cell Median: ")
        self.cellmedian_label.pack(side="left")
        self.cellmedian_value = tk.IntVar()
        self.cellmedian_value_label = tk.Label(
            self.cellmedian_frame, textvariable=self.cellmedian_value)
        self.cellmedian_value_label.pack(side="left")

        self.permedian_frame = tk.Frame(self.cell_info_frame)
        self.permedian_frame.pack(side="top", fill="x")
        self.permedian_label = tk.Label(
            self.permedian_frame, text="Perimeter Median: ")
        self.permedian_label.pack(side="left")
        self.permedian_value = tk.IntVar()
        self.permedian_value_label = tk.Label(
            self.permedian_frame, textvariable=self.permedian_value)
        self.permedian_value_label.pack(side="left")

        self.septmedian_frame = tk.Frame(self.cell_info_frame)
        self.septmedian_frame.pack(side="top", fill="x")
        self.septmedian_label = tk.Label(
            self.septmedian_frame, text="Septum Median: ")
        self.septmedian_label.pack(side="left")
        self.septmedian_value = tk.IntVar()
        self.septmedian_value_label = tk.Label(
            self.septmedian_frame, textvariable=self.septmedian_value)
        self.septmedian_value_label.pack(side="left")

        self.cytomedian_frame = tk.Frame(self.cell_info_frame)
        self.cytomedian_frame.pack(side="top", fill="x")
        self.cytomedian_label = tk.Label(
            self.cytomedian_frame, text="Cyto Median: ")
        self.cytomedian_label.pack(side="left")
        self.cytomedian_value = tk.IntVar()
        self.cytomedian_value_label = tk.Label(
            self.cytomedian_frame, textvariable=self.cytomedian_value)
        self.cytomedian_value_label.pack(side="left")

        self.fr_frame = tk.Frame(self.cell_info_frame)
        self.fr_frame.pack(side="top", fill="x")
        self.fr_label = tk.Label(self.fr_frame, text="FR: ")
        self.fr_label.pack(side="left")
        self.fr_value = tk.IntVar()
        self.fr_value_label = tk.Label(
            self.fr_frame, textvariable=self.fr_value)
        self.fr_value_label.pack(side="left")

        self.fr75_frame = tk.Frame(self.cell_info_frame)
        self.fr75_frame.pack(side="top", fill="x")
        self.fr75_label = tk.Label(self.fr75_frame, text="FR 75%: ")
        self.fr75_label.pack(side="left")
        self.fr75_value = tk.IntVar()
        self.fr75_value_label = tk.Label(
            self.fr75_frame, textvariable=self.fr75_value)
        self.fr75_value_label.pack(side="left")

        self.fr25_frame = tk.Frame(self.cell_info_frame)
        self.fr25_frame.pack(side="top", fill="x")
        self.fr25_label = tk.Label(self.fr25_frame, text="FR 25%: ")
        self.fr25_label.pack(side="left")
        self.fr25_value = tk.IntVar()
        self.fr25_value_label = tk.Label(
            self.fr25_frame, textvariable=self.fr25_value)
        self.fr25_value_label.pack(side="left")

        self.fr10_frame = tk.Frame(self.cell_info_frame)
        self.fr10_frame.pack(side="top", fill="x")
        self.fr10_label = tk.Label(self.fr10_frame, text="FR 10%: ")
        self.fr10_label.pack(side="left")
        self.fr10_value = tk.IntVar()
        self.fr10_value_label = tk.Label(
            self.fr10_frame, textvariable=self.fr10_value)
        self.fr10_value_label.pack(side="left")

        self.empty_label = tk.Label(self.cell_info_frame)
        self.empty_label.pack(side="top")

        self.base_button = tk.Button(self.images_frame, text="Base", command=lambda: self.show_image("Base"),
                                     width=self.image_buttons_width)
        self.base_button.pack(side="top", fill="x")
        self.base_button.config(state="active")

        self.base_features_button = tk.Button(self.images_frame, text="Base with Features",
                                              command=lambda: self.show_image(
                                                  "Base_features"),
                                              width=self.image_buttons_width)
        # self.base_features_button.pack(side="top", fill="x")

        self.base_w_cells_button = tk.Button(self.images_frame, text="Base Outlined",
                                             command=lambda: self.show_image("Base_cells_outlined"),
                                             width=self.image_buttons_width)
        self.base_w_cells_button.pack(side="top", fill="x")
        self.base_w_cells_button.config(state="active")

        self.fluor_button = tk.Button(self.images_frame, text="Fluorescence", command=lambda: self.show_image("Fluor"),
                                      width=self.image_buttons_width)
        self.fluor_button.pack(side="top", fill="x")

        self.fluor_cells_out_button = tk.Button(self.images_frame, text="Fluorescence Outlined",
                                                command=lambda: self.show_image(
                                                    "Fluor_cells_outlined"),
                                                width=self.image_buttons_width)
        self.fluor_cells_out_button.pack(side="top", fill="x")
        self.fluor_cells_out_button.config(state="active")

        self.optional_button = tk.Button(self.images_frame, text="Secondary Channel",
                                         command=lambda: self.show_image(
                                             "Optional"),
                                         width=self.image_buttons_width)
        self.optional_button.pack(side="top", fill="x")
        if self.ehooke.image_manager.optional_image is None:
            self.optional_button.config(state="disabled")
        else:
            self.optional_button.config(state="active")

        self.optional_w_cells_button = tk.Button(self.images_frame, text="Secondary Outlined",
                                                 command=lambda: self.show_image("Optional_cells_outlined"),
                                                 width=self.image_buttons_width)
        self.optional_w_cells_button.pack(side="top", fill="x")
        self.optional_w_cells_button.config(state="disabled")

        self.fluor_lines_button = tk.Button(self.images_frame, text="Linescan", command=lambda: self.show_image(
            "Fluor_with_lines"), width=self.image_buttons_width)
        self.fluor_lines_button.pack(side="top", fill="x")
        self.fluor_lines_button.config(state="active")

        self.current_image_label = tk.Label(self.images_frame, text="")
        self.current_image_label.pack(side="top")

        self.config_gui(self.main_window)

        self.cellprocessing_label.config(font=self.gui_font_bold)
        self.membrane_thickness_label.config(font=self.gui_font_bold)
        self.filters_label.config(font=self.gui_font_bold)
        self.cell_cycle_label.config(font=self.gui_font_bold)
        self.linescan_label.config(font=self.gui_font_bold)

        for w in self.cell_info_frame.winfo_children():
            for w1 in w.winfo_children():
                w1.config(font=("Verdana", 8, "normal"))

        self.show_image(self.current_image)

    def new_analysis(self):
        """Restarts ehooke to conduct a new analysis"""
        if self.event_connected:
            self.canvas.mpl_disconnect(self.cid)

        working_dir = self.ehooke.working_dir
        self.ehooke = EHooke()
        self.ehooke.working_dir = working_dir
        self.default_params = self.ehooke.parameters
        self.images = {}
        self.current_image = None
        self.set_imageloader()

    def on_closing(self):
        """Creates a prompt when trying to close the main windows"""
        if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
            self.main_window.destroy()
            self.main_window.quit()

    def config_gui_loop(self, widget):
        for w in widget.winfo_children():
            widget_class = w.winfo_class()

            if widget_class == "Frame":
                w.config(background=self.frames_background)
                self.config_gui_loop(w)

            elif widget_class == "Label":
                w.config(background=self.frames_background,
                         foreground=self.text_color,
                         font=self.gui_font)
            elif widget_class == "Button":
                w.config(background=self.button_background,
                         activebackground=self.button_background,
                         activeforeground=self.text_color,
                         highlightbackground=self.button_background,
                         highlightcolor=self.button_background,
                         foreground=self.text_color,
                         font=self.gui_font)
            elif widget_class == "Menubutton":
                w.config(background=self.button_background,
                         activebackground=self.button_background,
                         activeforeground=self.text_color,
                         foreground=self.text_color,
                         highlightbackground=self.button_background,
                         highlightcolor=self.button_background,
                         font=self.gui_font)
            elif widget_class == "Entry":
                w.config(background=self.button_background,
                         foreground=self.text_color,
                         font=self.gui_font)
            elif widget_class == "Checkbutton":
                w.config(background=self.button_background,
                         font=self.gui_font)
            elif widget_class == "Scale":
                w.config(background=self.button_background,
                         foreground=self.text_color,
                         highlightbackground=self.button_background,
                         highlightcolor=self.button_background,
                         font=self.gui_font)

    def config_gui(self, widget):

        if self.dark_mode:
            self.frames_background = "#102027"
            self.button_background = "#37474F"
            self.text_color = "#CFD8DC"
            self.fig.patch.set_facecolor('#4F5B62')

        else:
            self.frames_background = "gray99"
            self.button_background = "gray70"
            self.text_color = "gray5"
            self.fig.patch.set_facecolor('#E0E0E0')

        self.config_gui_loop(widget)


if __name__ == "__main__":
    interface = Interface()
    interface.main_window.protocol("WM_DELETE_WINDOW", interface.on_closing)
    interface.main_window.mainloop()
