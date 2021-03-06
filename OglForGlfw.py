
import glfw
import numpy as np
import scipy.linalg

from OpenGL.GL import *
from OpenGL.GLU import *
import TMesh


# Rotation matrix along x/y/z axis
def get_xrot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[  1,  0,  0],[  0, c,-s],[  0, s, c]])

def get_yrot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c,  0, s],[  0,  1,  0],[-s,  0, c]])

def get_zrot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c,-s,  0],[ s, c,  0],[  0,  0,  1]])

# Rotation matrix along axis 
def get_axisrot (angle, axis) :
    length = np.linalg.norm(axis)
    if length <= 0.0000001 :
        return np.identity(3)
    axis /= length
    return scipy.linalg.expm( np.cross(np.identity(3), angle*axis) )


# class OglForGlfw 
# This class manages
# + OpenGL camera (Viewpoint and projection) manipulation
# + OpenGL unproject (convert mouse point to 3d ray)
# + Light positions
# for pyOpenGL & glfw environment
class OglForGlfw:

    def __init__(self):
        # camera position/center/Y-direction
        self.cam_pos = np.array([0.0, 0.0, -10.0], dtype=np.float32)
        self.cam_cnt = np.array([0.0, 0.0,   0.0], dtype=np.float32)
        self.cam_up  = np.array([0.0, 1.0,   0.0], dtype=np.float32)

        self.mouse_pos2d = np.array([0.0, 0.0], dtype=np.float32)
        self.mouse_mode  = "NON"
        self.b_rendering = False
        self.clearcolor  = np.array([0.2, 0.2, 0.2, 0.5], dtype=np.float32);


    def draw_begin (
            self,
            glfw_window,
            view_w,
            view_h,
            fov_y = 45.0,
            view_near = 0.02,
            view_far = 700.0) :

        if self.b_rendering : return
        self.b_rendering = True
        glfw.make_context_current(glfw_window)

        #set viewport and projection matrix
        glViewport(0, 0, view_w, view_h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov_y, view_w / np.float32(view_h), view_near, view_far)

        #set ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt( self.cam_pos[0], self.cam_pos[1], self.cam_pos[2],
                   self.cam_cnt[0], self.cam_cnt[1], self.cam_cnt[2],
                   self.cam_up [0], self.cam_up [1], self.cam_up [2])
        glClearColor( self.clearcolor[0], self.clearcolor[1],
                      self.clearcolor[2], self.clearcolor[3])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

        #set light parameters
        self.set_light_params()


    def draw_end(self, glfw_window ):
        glfw.swap_buffers(glfw_window)
        glfw.make_context_current(None)
        self.b_rendering = False


    def set_light_params(self):
        light_pos = np.array([[ 1000, 1000,-1000,1],
                              [-1000, 1000,-1000,1],
                              [ 1000,-1000,-1000,1]], dtype=np.float32)
        light_white = np.array([1.0,1.0,1.0,1.0], dtype=np.float32)
        light_gray  = np.array([0.4,0.4,0.4,1.0], dtype=np.float32)
        light_black = np.array([0.0,0.0,0.0,1.0], dtype=np.float32)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_LIGHT2)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos[0])
        glLightfv(GL_LIGHT1, GL_POSITION, light_pos[1])
        glLightfv(GL_LIGHT2, GL_POSITION, light_pos[2])
        glLightfv(GL_LIGHT0, GL_AMBIENT , light_white )
        glLightfv(GL_LIGHT1, GL_AMBIENT , light_black )
        glLightfv(GL_LIGHT2, GL_AMBIENT , light_black )
        glLightfv(GL_LIGHT0, GL_DIFFUSE , light_white )
        glLightfv(GL_LIGHT1, GL_DIFFUSE , light_gray  )
        glLightfv(GL_LIGHT2, GL_DIFFUSE , light_gray  )
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_white )
        glLightfv(GL_LIGHT1, GL_SPECULAR, light_white )
        glLightfv(GL_LIGHT2, GL_SPECULAR, light_white )


    def mouse_down_rot(self, point):
        self.mouse_pos2d = np.array(point, dtype=np.float32)
        self.mouse_mode = "ROT"

    def mouse_down_trans(self, point):
        self.mouse_pos2d = np.array(point, dtype=np.float32)
        self.mouse_mode = "TRANS"

    def mouse_down_zoom(self, point):
        self.mouse_pos2d = np.array(point, dtype=np.float32)
        self.mouse_mode = "ZOOM"

    def mouse_up(self) :
        self.mouse_mode = "NON"

    def mouse_move(self, point):
        dx = np.float32(point[0]) - self.mouse_pos2d[0]
        dy = np.float32(point[1]) - self.mouse_pos2d[1]

        if self.mouse_mode == "NON":
            return

        elif self.mouse_mode == "ROT":
            theta = -dx / 200.0
            phi   = -dy / 200.0

            rot_theta = get_axisrot(theta, self.cam_up)
            rot_phi   = get_axisrot(phi  , np.cross( self.cam_cnt - self.cam_pos, self.cam_up))
            rot = np.dot(rot_phi, rot_theta)
            self.cam_up  = np.dot(rot, self.cam_up)
            self.cam_pos = np.dot(rot, (self.cam_pos - self.cam_cnt)) + self.cam_cnt

        elif self.mouse_mode == "ZOOM":
            new_pos = self.cam_pos + dy / 80.0 * (self.cam_cnt - self.cam_pos)
            if np.linalg.norm(new_pos - self.cam_cnt) > 0.02 :
                self.cam_pos = new_pos

        elif self.mouse_mode == "TRANS":
            c = np.linalg.norm(self.cam_pos - self.cam_cnt) / 900.0
            x_dir = np.cross(self.cam_pos - self.cam_cnt, self.cam_up)
            x_dir /= np.linalg.norm(x_dir)
            trans = (c * dx) * x_dir + (c * dy) * self.cam_up
            self.cam_pos += trans
            self.cam_cnt += trans
        self.mouse_pos2d = np.array(point, dtype=np.float32)


    def unproject(self, glfw_window, x, y, z):
        if not self.b_rendering :
            glfw.make_context_current(glfw_window)

        model_mat = glGetFloatv(GL_MODELVIEW_MATRIX)
        proj_mat  = glGetFloatv(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)
        pos = gluUnProject( x, vp[3] - y, z, model_mat.astype('d'), proj_mat.astype('d'), vp)

        if not self.b_rendering :
            glfw.make_context_current(None)
        return np.array(pos, dtype=np.float32)


    def get_cursor_ray(self, glfw_window, point):
        pos1 = self.unproject(glfw_window, point[0], point[1], 0.01)
        pos2 = self.unproject(glfw_window, point[0], point[1], 0.2 )
        ray_dir = pos2 - pos1
        ray_dir /= np.linalg.norm(ray_dir)
        return pos1, ray_dir


# Class GlfwMainWindow
# this class 
# + generates glfw window 
# + manages mouse events
#    following call back functions should be set at constractor
#    func_Ldown(x,y,ogl,window), func_Lup(x,y,ogl,window),
#    func_Rdown(x,y,ogl,window), func_Rup(x,y,ogl,window),
#    func_Mdown(x,y,ogl,window), func_Mup(x,y,ogl,window),
#    func_mouse_move(x,y,ogl,window),
#    func_draw_scene(ogl,window)
#    memo (x, y): mouse position, 
#         ogl   : instance of OglForGlfw
#         window: instance of GlfwMainWindow 
class GlfwMainWindow:

    def __init__(
            self,
            window_title,
            window_size,
            window_position,
            func_Ldown, func_Lup,
            func_Rdown, func_Rup,
            func_Mdown, func_Mup,
            func_mouse_move,
            func_draw_scene
        ):

        self.window = glfw.create_window(window_size[0], window_size[1], window_title, None, None)
        self.ogl    = OglForGlfw()

        if not self.window:
            glfw.terminate()
            raise RuntimeError('Could not create an window')

        self.func_Ldown, self.func_Lup = func_Ldown, func_Lup
        self.func_Rdown, self.func_Rup = func_Rdown, func_Rup
        self.func_Mdown, self.func_Mup = func_Mdown, func_Mup
        self.func_mouse_move = func_mouse_move
        self.func_draw_scene = func_draw_scene

        #set callback functions
        glfw.set_cursor_pos_callback    (self.window, self.cursor_pos)
        glfw.set_cursor_enter_callback  (self.window, self.cursor_enter)
        glfw.set_mouse_button_callback  (self.window, self.mouse_button)
        glfw.set_window_refresh_callback(self.window, self.window_refresh)
        glfw.set_window_pos             (self.window, window_position[0], window_position[1])

        #call display from here (TODO check! is this necessary?)
        glfw.make_context_current(self.window)
        glClearColor(0.3, 0.3, 0.3, 1.0)
        self.display()   # necessary only on Windows


    def cursor_pos(self, window, xpos, ypos):
        self.func_mouse_move( (xpos, ypos), self.ogl, self)

    def cursor_enter(self, window, entered):
        #print( 'cursor_enter:',entered, id(self.window), id(window))
        pass

    def mouse_button(self, window, button, action, mods):
        point = glfw.get_cursor_pos(window) #point : 2dtuple
        if   ( button == 0 and action == 1) : self.func_Ldown( point, self.ogl, self)
        elif ( button == 1 and action == 1) : self.func_Rdown( point, self.ogl, self)
        elif ( button == 2 and action == 1) : self.func_Mdown( point, self.ogl, self)
        elif ( button == 0 and action == 0) : self.func_Lup  ( point, self.ogl, self)
        elif ( button == 1 and action == 0) : self.func_Rup  ( point, self.ogl, self)
        elif ( button == 2 and action == 0) : self.func_Mup  ( point, self.ogl, self)

    def window_should_close(self):
        return glfw.window_should_close(self.window)

    def wait_events_timeout(self):
        glfw.make_context_current(self.window)
        glfw.wait_events_timeout(1e-3)

    def window_refresh(self, window):
        self.display()

    def display(self):
        size = glfw.get_window_size(self.window)
        self.ogl.draw_begin(
            self.window, size[0], size[1],
            fov_y = 45.0, view_near = 0.02, view_far = 700.0
        )

        glDisable(GL_LIGHTING)
        glLineWidth(2)
        glBegin(GL_LINES)
        glColor3d ( 1.0, 0.0, 0.0)
        glVertex3d( 0.0, 0.0, 0.0)
        glVertex3d( 1.0, 0.0, 0.0)
        glColor3d ( 0.0, 1.0, 0.0)
        glVertex3d( 0.0, 0.0, 0.0)
        glVertex3d( 0.0, 1.0, 0.0)
        glColor3d ( 0.0, 0.0, 1.0)
        glVertex3d( 0.0, 0.0, 0.0)
        glVertex3d( 0.0, 0.0, 1.0)
        glEnd()

        self.func_draw_scene(self.ogl, self)

        self.ogl.draw_end(self.window)
        #glfw.make_context_current(0)
