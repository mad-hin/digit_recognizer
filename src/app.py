from tkinter import *

try :
    from PIL import ImageGrab
except :
    from pyscreenshot import ImageGrab

from image_process.img_process import img_preprocess, predict_digit


class App(Tk):
    def __init__(self, model_name):
        """
        Init the gui layout
        :param model_name: the model using
        """
        super().__init__()
        self.modelName = model_name

        # configure the root window
        self.title("AI digital recognition with " + self.modelName)
        self.geometry("300x400")
        self.resizable(width=0, height=0)

        # create Canvas for handwrite digit
        self.cv = Canvas(self, width=300, height=300, bg='white')
        self.cv.bind('<B1-Motion>', self.paint)
        self.cv.pack()

        # Button for reset
        self.bnt1 = Button(self, text="Reset", width=250, command=self.reset_canvas)
        self.bnt1.pack()

        # Button for prediction
        self.bnt2 = Button(self, text="Predict", width=250, command=self.prediction)
        self.bnt2.pack()

        # Text box for show prediction
        self.t = Text(self)
        self.t.pack()

    def paint(self, event):
        """
        Drawing on the canvas
        :param event:
        """
        # get x1, y1, x2, y2 co-ordinates
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        color = "black"
        # display the mouse movement inside Canvas
        self.cv.create_oval(x1, y1, x2, y2, fill=color, outline=color)

    def prediction(self):
        """
        print the predicted number to Text box
        """
        self.canvas2image()
        img = img_preprocess("images/out.png")
        predict = predict_digit(img, self.modelName)
        self.t.delete(1.0, END)
        self.t.insert(INSERT, str(predict))

    def canvas2image(self):
        """
        convert the canvas to image
        """
        # Get the coordinate of the canvas
        x = self.cv.winfo_rootx()
        print(x)
        y = self.cv.winfo_rooty()
        x1 = x + self.cv.winfo_width()
        y1 = y + + self.cv.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save("images/out.png")
        print("image created")

    def reset_canvas(self):
        """
        reset the canvas
        """
        self.cv.delete("all")


a = App("cnn")
a.mainloop()
