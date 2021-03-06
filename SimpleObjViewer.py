
import glfw
import numpy as np
import scipy.linalg 

from OpenGL.GL import *
from OpenGL.GLU import *
import OglForGlfw 
import TMesh



# class EventManager
# this is a class to impliment all mouse event handler and draw_scene function
class EventManager:

    def __init__(self):
        self.b_Lbtn = False
        self.b_Rbtn = False
        self.b_Mbtn = False
        self.object1 = TMesh.TMesh( init_as = "Cube"  , radi=0.5)
        self.object2 = TMesh.TMesh( init_as = "Sphere", radi=0.8)
        self.object3 = TMesh.TMesh( init_as = "Obj"   , fname="./e3.obj") 
        
    def func_Ldown(self, point, ogl, window) :
        self.b_Lbtn = True
        ogl.mouse_down_trans(point)
    
    def func_Lup(self, point, ogl, window):
        self.b_Lbtn = False
        ogl.mouse_up()

    def func_Rdown(self, point, ogl, window):
        self.b_Rbtn = True
        ogl.mouse_down_rot(point)

    def func_Rup(self, point, ogl, window): 
        self.b_Rbtn = False
        ogl.mouse_up()

    def func_Mdown(self, point, ogl, window):
        self.b_Mbtn = True
        ogl.mouse_down_zoom(point)

    def func_Mup(self, point, ogl, window):
        self.b_Mbtn = False
        ogl.mouse_up()

    def func_mouse_move(self, point, ogl, window):
        if not (self.b_Lbtn or self.b_Rbtn or self.b_Mbtn) :
            return  
        ogl.mouse_move(point)
        window.display()
    
    def func_draw_scene(self, ogl, window):
        glEnable(GL_LIGHTING)
        mate = np.array([[0.2,0.2,0.1,0.5],[0.7,0.7,0.1,0.5],[1.0,1.0,1.0,0.5],[64.0,0,0,0]], dtype=np.float32)
        self.object1.draw_by_VBO(mate[0], mate[1], mate[2], mate[3], "polygon")
        glTranslated(3.0,0,0)
        self.object2.draw_by_VBO(mate[0], mate[1], mate[2], mate[3], "smooth")
        glTranslated(-3.0,0,0)
        self.object3.draw_by_VBO(mate[0], mate[1], mate[2], mate[3], "smooth")



def main():    
    #Howto use GlfwMainWindow and OglForGlfw
    #1. prepare a class with callback functions 
    manager = EventManager()

    #2. generate instance of GlfwMainWindow
    if not glfw.init():
        raise RuntimeError("Fails to initialize glfw")
    mainwindow = OglForGlfw.GlfwMainWindow(  
            "Main Window", 
            [800, 600], 
            [100,100],
            manager.func_Ldown, manager.func_Lup, 
            manager.func_Rdown, manager.func_Rup, 
            manager.func_Mdown, manager.func_Mup, 
            manager.func_mouse_move,
            manager.func_draw_scene)
    
    #3. wait events
    while not ( mainwindow.window_should_close()):
        mainwindow.wait_events_timeout()

    glfw.terminate()

if __name__ == "__main__":
    main()
