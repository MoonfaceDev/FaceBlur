# imports

from pickle import dumps, loads
import tkinter as tk
from functools import partial
from multiprocessing import Queue, Process, freeze_support
from queue import Empty
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.ttk import Frame, Button, Label, Separator, Spinbox, Combobox

import cv2
import numpy as np
from PIL import ImageTk, Image

import blur_methods
import detection
import models
import recognition
import tracking
import video_utils


class App(Frame):
    def __init__(self, window):
        Frame.__init__(self, window, name="frame")
        # Import icons
        self.visibility_icon = ImageTk.PhotoImage(Image.open("../images/visibility.png").resize((25, 25)))
        self.visibility_off_icon = ImageTk.PhotoImage(Image.open("../images/visibility_off.png").resize((25, 25)))
        self.back_icon = ImageTk.PhotoImage(Image.open("../images/arrow_back.png").resize((25, 25)))
        self.forward_icon = ImageTk.PhotoImage(Image.open("../images/arrow_forward.png").resize((25, 25)))
        self.done_icon = ImageTk.PhotoImage(Image.open("../images/done.png").resize((25, 25)))
        self.close_icon = ImageTk.PhotoImage(Image.open("../images/close.png").resize((25, 25)))
        self.settings_icon = ImageTk.PhotoImage(Image.open("../images/settings.png").resize((25, 25)))
        self.logo_image = ImageTk.PhotoImage(Image.open("../images/logo.png").resize((100, 100)))
        # Initialize window
        self.window = window
        self.window.geometry("400x300+100+100")
        self.window.title("FaceBlur")
        self.window.resizable(False, False)
        self.window.iconphoto(True, self.logo_image)
        self.pack(fill=tk.BOTH, expand=True)
        self.initialize_video_selection_ui()
        # Initialize class fields
        self.index = 0  # index of current showing unique face image
        self.faces_display = []  # list of PhotoImage objects of frames including unique faces
        self.video_file = ""  # path to input video
        self.destination = ""  # path to output video
        self.queue = Queue()  # multiprocessing queue for sending data back from different processes
        self.photo_img = None  # PhotoImage object of current frame
        self.blur_toggles = None  # ndarray where each element indicates whether the matching face will be blurred
        self.settings = {  # dictionary holding settings value from settings window
            "resamples": 1,
            "tolerance": 0.5,
            "track_period": 10,
            "blur_method": "pixelate",
            "blur_intensity": 20,
            "display_output": True
        }

    def initialize_video_selection_ui(self):
        # Initialize logo picture
        logo = Label(self, image=self.logo_image)
        logo.image = self.logo_image
        logo.place(y=125, x=200, anchor=tk.CENTER)
        # Initialize settings button
        settings_button = Button(self, command=self.open_settings, image=self.settings_icon)
        settings_button.place(y=275, x=25, anchor=tk.CENTER)
        # Initialize select video button
        select_video_button = Button(self, text="Select Video")
        select_video_button.config(command=partial(self.select_video, [logo, select_video_button, settings_button], [select_video_button, settings_button]))
        select_video_button.place(y=200, x=200, anchor=tk.CENTER)

    def initialize_face_toggle_ui(self, face_locations, face_encodings, unique_encodings):
        # Initialize canvas
        canvas = tk.Canvas(self, width=400, height=250)
        canvas.place(y=0, relx=0.5, anchor=tk.N)
        # Initialize the strip below canvas
        strip = tk.Canvas(self, width=400, height=50, bg="#222222")
        strip.place(y=250, relx=0.5, anchor=tk.N)
        # Initialize blur toggle button
        blur_toggle_button = Button(self)
        blur_toggle_button.config(command=partial(self.toggle_blur, canvas, blur_toggle_button))
        blur_toggle_button.place(y=275, x=200, anchor=tk.CENTER)
        # Initialize back arrow button
        back_button = Button(self, image=self.back_icon)
        back_button.config(command=partial(self.back, canvas, blur_toggle_button))
        back_button.place(y=275, x=150, anchor=tk.CENTER)
        # Initialize forward arrow  button
        forward_button = Button(self, image=self.forward_icon)
        forward_button.config(command=partial(self.forward, canvas, blur_toggle_button))
        forward_button.place(y=275, x=250, anchor=tk.CENTER)
        # Initialize close button
        close_button = Button(self, command=self.cancel, image=self.close_icon)
        close_button.place(y=275, x=25, anchor=tk.CENTER)
        # Initialize done button
        done_button = Button(self, image=self.done_icon)
        done_button.config(command=partial(self.done, [back_button, forward_button, done_button, close_button, blur_toggle_button], face_locations, face_encodings, unique_encodings))
        done_button.place(y=275, x=375, anchor=tk.CENTER)
        # Display first face image
        self.display_face(canvas, blur_toggle_button)

    def process_queue(self, data_size=-1, on_complete=None):
        if data_size == -1:
            if self.queue.qsize() == 1:  # Check if process has finished
                try:
                    self.queue.get()
                    self.close()
                except Empty:
                    self.window.after(1000, self.process_queue, data_size, on_complete)
            else:
                self.window.after(1000, self.process_queue, data_size, on_complete)
        else:
            if self.queue.qsize() == data_size:  # Check if process has finished
                try:
                    data = get_data(self.queue, data_size)
                    on_complete(data)
                except Empty:
                    self.window.after(1000, self.process_queue, data_size, on_complete)
            else:
                self.window.after(1000, self.process_queue, data_size, on_complete)

    def on_analyze_completed(self, widgets_to_destroy, data):
        # Unpack data from video analysis
        (face_locations,
         face_encodings,
         unique_frames,
         unique_faces,
         unique_encodings) = data
        # Change to face toggle UI
        for widget in widgets_to_destroy:
            widget.destroy()
        for i in range(len(unique_faces)):
            focused_image = blur_methods.focused(unique_frames[i].copy(), unique_faces[i], 20)  # image with blurred background
            display_img, face = fit_to_window(focused_image, unique_faces[i])  # cropped image, new face coordinates
            cv2.rectangle(img=display_img, pt1=(face[0], face[1]), pt2=(face[2], face[3]), color=(0, 0, 0), thickness=2)
            self.faces_display.append(display_img)
        self.blur_toggles = np.zeros(len(self.faces_display), dtype=int)
        self.initialize_face_toggle_ui(face_locations, face_encodings, unique_encodings)

    def display_face(self, canvas, blur_toggle_button):
        # Display current image
        self.photo_img = ImageTk.PhotoImage(Image.fromarray(self.faces_display[self.index]))
        canvas.create_image((0, 0), image=self.photo_img, anchor=tk.NW)
        # Decide blur toggle icon color
        if self.blur_toggles[self.index]:
            blur_toggle_button.config(image=self.visibility_off_icon)
        else:
            blur_toggle_button.config(image=self.visibility_icon)

    def toggle_blur(self, canvas, blur_toggle_button):
        # Changes the blur toggle
        self.blur_toggles[self.index] = not self.blur_toggles[self.index]
        self.display_face(canvas, blur_toggle_button)

    def back(self, canvas, blur_toggle_button):
        # Moves backwards and displays the matching image
        if self.index == 0:
            self.index = len(self.blur_toggles) - 1
        else:
            self.index = self.index - 1
        self.display_face(canvas, blur_toggle_button)

    def forward(self, canvas, blur_toggle_button):
        # Moves forward and displays the matching image
        if self.index == len(self.blur_toggles) - 1:
            self.index = 0
        else:
            self.index = self.index + 1
        self.display_face(canvas, blur_toggle_button)

    def done(self, widgets_to_disable, face_locations, face_encodings, unique_encodings):
        # Spawns the video saving destination dialog
        self.select_destination(face_locations, face_encodings, unique_encodings, widgets_to_disable)

    def close(self):
        # Closes the program after video was created and saved
        self.window.destroy()

    def cancel(self):
        # Restarts the program
        self.window.destroy()
        window = tk.Tk()
        App(window)
        window.mainloop()

    def open_settings(self):
        # Opens settings as a new sub-window
        settings_window = tk.Toplevel(self.window)
        settings_window.transient(self.window)
        settings_window.grab_set()
        Settings(settings_window, self.settings)

    def select_video(self, widgets_to_remove, widgets_to_disable):
        # Disables UI
        for widget in widgets_to_disable:
            widget.config(state=tk.DISABLED)
        # Spawns video opening dialog
        self.video_file = askopenfilename(title="Open", filetypes=[
            ("All Video Files", ".mp4"),
            ("All Video Files", ".flv"),
            ("All Video Files", ".avi"),
        ])
        if self.video_file:  # Pressed 'Open'
            # Starts video analysis process
            process = Process(target=analyze_video, args=(self.queue, self.video_file, self.settings))  # analysis process
            process.start()
            self.window.after(1000, self.process_queue, 5, partial(self.on_analyze_completed, widgets_to_remove))
        else:  # Pressed 'Cancel'
            # Enables UI back
            for widget in widgets_to_disable:
                widget.config(state=tk.NORMAL)

    def select_destination(self, face_locations, face_encodings, unique_encodings, widgets_to_disable):
        # Disables UI
        for widget in widgets_to_disable:
            widget.config(state=tk.DISABLED)
        # Spawns video saving dialog
        self.destination = asksaveasfilename(title="Select Destination", defaultextension=".avi", filetypes=[("AVI", "*.avi")])
        if self.destination:  # Pressed 'Save'
            # Creates list of chosen faces' encoding arrays
            blur_encodings = []  # encodings of chosen faces
            for i in range(len(unique_encodings)):
                if self.blur_toggles[i]:
                    blur_encodings.append(unique_encodings[i])
            # Starts video making process
            process = Process(target=make_video, args=(self.queue, self.video_file, self.destination, face_locations, face_encodings, blur_encodings, self.settings))  # video making process
            process.start()
            self.window.after(1000, self.process_queue)
        else:  # Pressed 'Cancel'
            # Enables UI back
            for widget in widgets_to_disable:
                widget.config(state=tk.NORMAL)


class Settings(Frame):
    def __init__(self, window, settings):
        Frame.__init__(self, window, name="settings")
        # Initialize window
        self.logo_image = ImageTk.PhotoImage(Image.open("../images/logo.png").resize((100, 100)))
        self.window = window
        self.window.geometry("400x331+200+200")
        self.window.title("Settings")
        self.window.resizable(False, False)
        self.window.iconphoto(False, self.logo_image)
        self.pack(fill=tk.BOTH, expand=True)
        self.settings = settings
        self.initialize_settings_ui()

    def initialize_settings_ui(self):
        # Initializes resamples field
        resamples_label = Label(self, text="Resamples (affects recognition accuracy)")
        resamples_label.place(y=12.5, x=12.5)
        resamples_text = tk.StringVar()
        resamples_text.set(self.settings["resamples"])
        resamples_spin = Spinbox(self, from_=1, to=10, textvariable=resamples_text)
        resamples_spin.config(command=partial(self.save_settings, "resamples", lambda: int(resamples_text.get())))
        resamples_spin.place(y=37.5, x=12.5)
        separator1 = Separator(self, orient='horizontal')
        separator1.place(y=62.5, x=12.5, width=375, height=1)

        # Initializes tolerance field
        tolerance_label = Label(self, text="Face matching tolerance (lower is more strict)")
        tolerance_label.place(y=68.75, x=12.5)
        tolerance_text = tk.StringVar()
        tolerance_text.set(self.settings["tolerance"])
        tolerance_spin = Spinbox(self, from_=0, to=1, increment=0.1, textvariable=tolerance_text)
        tolerance_spin.config(command=partial(self.save_settings, "tolerance", lambda: float(tolerance_text.get())))
        tolerance_spin.place(y=93.75, x=12.5)
        separator2 = Separator(self, orient='horizontal')
        separator2.place(y=118.75, x=12.5, width=375, height=1)

        # Initializes track period field
        track_period_label = Label(self, text="Track period (the number of frames between each recognition)")
        track_period_label.place(y=125, x=12.5)
        track_period_text = tk.StringVar()
        track_period_text.set(self.settings["track_period"])
        track_period_spin = Spinbox(self, from_=1, to=30, textvariable=track_period_text)
        track_period_spin.config(command=partial(self.save_settings, "track_period", lambda: int(track_period_text.get())))
        track_period_spin.place(y=150, x=12.5)
        separator3 = Separator(self, orient='horizontal')
        separator3.place(y=175, x=12.5, width=375, height=1)

        # Initializes blur method field
        blur_method_label = Label(self, text="Blur method")
        blur_method_label.place(y=181.25, x=12.5)
        blur_method_text = tk.StringVar()
        blur_method_text.set(self.settings["blur_method"])
        blur_method_menu = Combobox(self, textvariable=blur_method_text, values=("pixelate", "blur", "blacken"))
        blur_method_text.trace('w', partial(self.save_settings, "blur_method", lambda: blur_method_text.get()))
        blur_method_menu.place(y=206.25, x=12.5)
        separator4 = Separator(self, orient='horizontal')
        separator4.place(y=231.25, x=12.5, width=375, height=1)

        # Initializes blur intensity field
        blur_intensity_label = Label(self, text="Blur intensity (filter size)")
        blur_intensity_label.place(y=237.5, x=12.5)
        blur_intensity_text = tk.StringVar()
        blur_intensity_text.set(self.settings["blur_intensity"])
        blur_intensity_spin = Spinbox(self, from_=1, to=30, textvariable=blur_intensity_text)
        blur_intensity_spin.config(command=partial(self.save_settings, "blur_intensity", lambda: int(blur_intensity_text.get())))
        blur_intensity_spin.place(y=262.5, x=12.5)
        separator5 = Separator(self, orient='horizontal')
        separator5.place(y=287.5, x=12.5, width=375, height=1)

        # Initializes display output field
        display_output_flag = tk.IntVar()
        display_output_flag.set(self.settings["display_output"])
        display_output_checkbox = tk.Checkbutton(self, text='Display output', variable=display_output_flag, onvalue=1, offvalue=0)
        display_output_checkbox.config(command=partial(self.save_settings, "display_output", lambda: display_output_flag.get()))
        display_output_checkbox.place(y=293.75, x=12.5)

    def save_settings(self, name, get_variable, *_):
        # Saves setting to dictionary
        self.settings[name] = get_variable()


def analyze_video(queue, video_file, settings):
    """
    Finds faces' locations and encodings in the desired frames, and finds unique faces
    :param queue: Multiprocessing queue
    :param video_file: Path to input video file
    :param settings: Settings dictionary
    """
    # get video
    video = cv2.VideoCapture(video_file)  # input VideoCapture object
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames in input video
    # initialize models
    net = models.detection_dnn_model()  # detection model
    face_encoder = models.face_encoder()  # face recognition encoding model
    pose_predictor = models.pose_predictor()  # pose predictor model for finding face landmarks
    # get settings
    track_period = settings["track_period"]  # track period from settings
    resamples = settings["resamples"]  # number of resamples from settings
    tolerance = settings["tolerance"]  # face matching tolerance from settings
    # initialize lists
    face_locations = []  # list of face locations for each frame
    face_encodings = []  # list of face encodings for each frame
    unique_frames = []  # list of frame images including unique faces
    unique_face_locations = []  # list of unique faces' locations
    unique_encodings = []  # list of unique faces' encodings
    for i in range(frame_count):
        ret, img = video.read()  # ret indicates if frame was read correctly, img is last read frame
        if i % track_period == 0:  # frame for detection
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # frame in rgb format
            face_locations.append(detection.detect_faces(img, net))
            face_encodings.append(recognition.encode_faces(rgb, face_locations[-1], pose_predictor, face_encoder, resamples))
            new_indices, new_encodings = recognition.exclude_faces(face_encodings[-1], np.array(unique_encodings), tolerance)  # indices and encodings of new faces found
            for k in range(len(new_indices)):  # for each new face found
                unique_frames.append(rgb)
                unique_face_locations.append(face_locations[-1][k])
                unique_encodings.append(new_encodings[k])
    send_data(queue, [face_locations, face_encodings, unique_frames, unique_face_locations, unique_encodings])


def send_data(queue, data):
    """
    Pushes all elements into the queue
    :param queue: Multiprocessing queue
    :param data: Data array to be pushed into the queue
    """
    for obj in data:
        queue.put(dumps(obj, protocol=-1))


def get_data(queue, item_count):
    """
    :param queue: Multiprocessing queue
    :param item_count: Number of elements to be pulled
    :return: Array of pulled elements
    """
    return [loads(queue.get()) for _ in range(item_count)]


def make_video(queue, video_file, destination, face_locations, face_encodings, match_encodings, settings):
    """
    Blurs selected faces and generates video
    :param queue: Multiprocessing queue
    :param video_file: Path to input video file
    :param destination: Path to output location
    :param face_locations: List of face locations for each frame in which detection was performed
    :param face_encodings: List of face encodings for each frame in which detection was performed
    :param match_encodings: List of chosen faces' encodings
    :param settings: Settings dictionary
    """
    trackers = []  # list of tracker objects, one for each matched face
    # get video
    video = cv2.VideoCapture(video_file)  # input VideoCapture object
    frame_rate = video.get(cv2.CAP_PROP_FPS)  # frames per second in input video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # width of input video frame
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height of input video frame
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # number of frames in input video
    # get settings
    track_period = settings["track_period"]  # track period from settings
    tolerance = settings["tolerance"]  # face matching tolerance from settings
    blur_method = settings["blur_method"]  # type of blurring from settings
    blur_intensity = settings["blur_intensity"]  # blurring filter size from settings
    display_output = settings["display_output"]  # flag indicating whether to display output video from settings
    # initialize writer
    out = video_utils.initialize_writer(destination, (width, height), frame_rate)  # VideoWriter object
    for i in range(frame_count):
        ret, img = video.read()  # ret indicates if frame was read correctly, img is last read frame
        if i % track_period == 0:  # frame for detection
            current_frame_encodings = np.array(face_encodings[i // track_period])  # array of encodings for faces in current frame
            matched_indices, matched_encodings = recognition.match_faces(current_frame_encodings, np.array(match_encodings), tolerance)  # indices of matched faces from current frame and their encodings
            matched_locations = [face_locations[i // track_period][k] for k in matched_indices]  # locations of matched faces from current frame
            trackers = tracking.start_trackers(img, matched_locations)  # list of tracker objects, one for each matched face
        else:  # frame for tracking
            matched_locations = tracking.update_locations(trackers, img)  # updated locations of matched faces from current frame
        # generate blurred image
        blurred = None  # object holding image with blurred faces
        if blur_method == "pixelate":
            blurred = blur_methods.pixelated(img, matched_locations, blur_intensity)
        elif blur_method == "blur":
            blurred = blur_methods.blurred(img, matched_locations, blur_intensity)
        elif blur_method == "blacken":
            blurred = blur_methods.blackened(img, matched_locations)
        out.write(blurred)

    out.release()
    queue.put(0)
    if display_output:
        video_utils.display_video(destination)


def fit_to_window(img, face, target_width=400, target_height=250):
    """
    :param img: Original frame
    :param face: face location in the frame
    :param target_width: Width of cropped frame
    :param target_height: Height of cropped frame
    :return: Frame after being cropped accordingly to face, new face location
    """
    height, width, _ = img.shape  # input image dimensions
    if width / height > target_width / target_height:  # crop horizontally
        ratio = target_height / height  # image sizes ratio
        height = int(ratio * height)  # output image height
        width = int(ratio * width)  # output image width
        face = (face * ratio).astype(int)  # face coordinates in new image
        img = cv2.resize(img, (width, height))  # transformed image
        x_middle = int((face[0] + face[2]) / 2)  # face center x coordinate
        if x_middle < target_width / 2:  # returns the left part of the image
            return img[:, 0:target_width], face - np.array([0, 0, 0, 0])
        if x_middle > width - target_width / 2:  # returns the right part of the image
            return img[:, width - target_width:width], face - np.array([width - target_width, 0, width - target_width, 0])
        # returns the part of the image around the face
        return img[:, x_middle - int(target_width / 2):x_middle + int(target_width / 2)], face - np.array([x_middle - int(target_width / 2), 0, x_middle - int(target_width / 2), 0])
    else:  # crop vertically
        ratio = target_width / width  # image sizes ratio
        height = int(height * ratio)  # output image height
        width = int(width * ratio)  # output image width
        face = (face * ratio).astype(int)  # face coordinates in new image
        img = cv2.resize(img, (width, height))  # transformed image
        y_middle = int((face[1] + face[3]) / 2)  # face center y coordinate
        if y_middle < target_height / 2:   # returns the upper part of the image
            return img[0:target_height, :], face - np.array([0, 0, 0, 0])
        if y_middle > height - target_height / 2:  # returns the lower part of the image
            return img[height - target_height:height, :], face - np.array([0, height - target_height, 0, height - target_height])
        # returns the part of the image around the face
        return img[y_middle - int(target_height / 2):y_middle + int(target_height / 2), :], face - np.array([0, y_middle - int(target_height / 2), 0, y_middle - int(target_height / 2)])


def main():
    window = tk.Tk()
    App(window)
    window.mainloop()


if __name__ == "__main__":
    freeze_support()
    main()
