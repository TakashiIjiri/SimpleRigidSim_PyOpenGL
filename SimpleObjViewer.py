
import glfw
import numpy as np
import scipy.linalg 

from OpenGL.GL import *
from OpenGL.GLU import *
import GlfwWinManager 
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
        self.pre_pos = (0,0)        
    def func_Ldown(self, point, glfw_manager) :
        self.pre_pos = point
        self.b_Lbtn = True
    
    def func_Lup(self, point, glfw_manager):
        self.pre_pos = point
        self.b_Lbtn = False

    def func_Rdown(self, point, glfw_manager):
        self.pre_pos = point
        self.b_Rbtn = True

    def func_Rup(self, point, glfw_manager): 
        self.pre_pos = point
        self.b_Rbtn = False

    def func_Mdown(self, point, glfw_managerw):
        self.pre_pos = point
        self.b_Mbtn = True

    def func_Mup(self, point, glfw_manager):
        self.pre_pos = point
        self.b_Mbtn = False

    def func_mouse_move(self, point, glfw_manager):
        if not (self.b_Lbtn or self.b_Rbtn or self.b_Mbtn) :
            return  
        dx = point[0] - self.pre_pos[0]
        dy = point[1] - self.pre_pos[1]
        if self.b_Lbtn : glfw_manager.camera_trans(dx, dy)
        if self.b_Mbtn : glfw_manager.camera_zoom (dx, dy)
        if self.b_Rbtn : glfw_manager.camera_rot  (dx, dy)
        self.pre_pos = point
        glfw_manager.display()
    
    def func_draw_scene(self, glfw_manager):
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
    glfw_manager = GlfwWinManager.GlfwWinManager(  
            "Main Window", 
            [800, 600], 
            [100,100],
            manager.func_Ldown, manager.func_Lup, 
            manager.func_Rdown, manager.func_Rup, 
            manager.func_Mdown, manager.func_Mup, 
            manager.func_mouse_move,
            manager.func_draw_scene)
    
    #3. wait events
    while not ( glfw_manager.window_should_close()):
        glfw_manager.wait_events_timeout()

    glfw.terminate()

if __name__ == "__main__":
    main()
