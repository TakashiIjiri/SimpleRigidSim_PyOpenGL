
# Simple sample of rigid simulator in Python/OpenGL/Glfw environment
# Render objects with OpenGL and glfw
# Set timer by tkinter 


import glfw
import numpy as np
import scipy.linalg
import time

import tkinter as tk
import tkinter.ttk as ttk

from OpenGL.GL import *
from OpenGL.GLU import *
import OglForGlfw
import TMesh


FLOOR_Y = 0.0
FLOOR_SIZE = (10.0,10.0)


# class RigidBall
# 剛体の球を表すクラス
# 
class RigidBall:
    def __init__(self, radius, init_pos, init_velo ) :
        self.radi = radius
        self.pos   = init_pos
        self.velo  = init_velo
        self.rot   = np.zeros(3,dtype=np.float32)
        self.rot_v = np.zeros(3,dtype=np.float32)

        self.mesh  = TMesh.TMesh(init_as="Sphere", radi = radius )
        self.mate  = np.array([[0.2,0.2,0.2,0.5],[0.2,0.7,0.1,0.5],[1.0,1.0,1.0,0.5],[64.0,0,0,0]], dtype=np.float32)

        self.vis_line = np.zeros((2,3), np.float32)
        self.torque_dir = np.zeros((1,3), np.float32)

    def get_rot_mat(self) :
        length = np.linalg.norm(self.rot)
        if  length > 0.0001:
            rotmat = OglForGlfw.get_axisrot( length, self.rot/length )
        else :
            rotmat = np.identity(3, np.float32)
        return rotmat

    def step(self, dt, drag_const ):
        # 剛体シミュレーション
        # 慣性モーメントテンソル = I と近似
        # その他パラメタはすべて適当に指定
        # drag_const はドラッグ中の制約頂点
        # [const_vtx, target_pos] 

        force  = np.zeros(3, dtype=np.float32)
        torque = np.zeros(3, dtype=np.float32)
        force += np.array([0,-10,0], dtype=np.float32) #gravity

        if drag_const[0] >= 0 :
            rotmat = self.get_rot_mat()
            const_pos = np.dot(rotmat , self.mesh.verts[ drag_const[0] ]) + self.pos
            torque += 5.0 * np.cross(const_pos - self.pos, drag_const[1] - const_pos)
            force  += 3 * (drag_const[1] - const_pos)
            self.vis_line[0,:] = const_pos
            self.vis_line[1,:] = drag_const[1]
        else:
            self.vis_line = np.zeros((2,3), np.float32)
        self.torque_dir = torque


        self.rot_v += dt * torque
        self.rot   += dt * self.rot_v
        self.velo  += dt * force
        self.pos   += dt * self.velo
        self.velo  *= 0.99
        self.rot_v *= 0.94

        # collistion to the floor
        if self.pos[1] - self.radi < FLOOR_Y:
            self.pos[1] = FLOOR_Y + self.radi
            self.velo[1] *= -1


    def draw(self) :
        glEnable(GL_LIGHTING)

        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        rotmat = self.get_rot_mat()
        m = np.identity(4, dtype=np.float32)
        m[0:3,0:3] = rotmat[0:3,0:3]
        glMultMatrixf(m.transpose())
        self.mesh.draw_by_VBO(self.mate[0], self.mate[1], self.mate[2], self.mate[3])
        glPopMatrix()

        #vis const
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        glColor3d(0,0,1)
        glVertex3fv(self.vis_line[0])
        glVertex3fv(self.vis_line[1])

        glColor3d(1,1,0)
        glVertex3fv(self.pos)
        glVertex3fv(self.pos + 0,3 * self.torque_dir)
        glEnd()

    # return picked position and closest vertex index
    def pick(self, ray_pos, ray_dir) :
        rotmat = self.get_rot_mat()
        return self.mesh.pick(ray_pos, ray_dir, rotmat, self.pos)



# class EventManager
# this class manages mouse events
# このクラスにマウスイベント処理・描画処理を集約
class EventManager:

    def __init__(self):
        self.b_Lbtn = False
        self.b_Rbtn = False
        self.b_Mbtn = False
        # obj_idx, vtx_idx, pick_pos, draged_pos
        self.b_drag_object = [-1,-1, np.zeros(3, np.float32), np.zeros(3, np.float32)]

        self.balls = [RigidBall(1.5, np.array([0.,2.,5.]), np.array([0.,1.,0.])),
                      RigidBall(1.2, np.array([2.,6.,0.]), np.array([0.,1.,1.])),
                      RigidBall(1.8, np.array([0.,2.,2.]), np.array([-1.,1.,0.]))]

    def func_Ldown(self, point, ogl, window) :
        self.b_Lbtn = True
        ray_pos, ray_dir = ogl.get_cursor_ray(window.window, point)
        for i, b in enumerate(self.balls) :
            pick_vid, pick_pos = b.pick(ray_pos, ray_dir)
            if  pick_vid != -1 :
                self.b_drag_object = [i, pick_vid, pick_pos, pick_pos]

        if self.b_drag_object[0] < 0 :
            ogl.mouse_down_trans(point)


    def func_Lup(self, point, ogl, window):
        self.b_Lbtn = False
        self.b_drag_object = [-1,-1, np.zeros(3, np.float32)] # obj_idx, vtx_idx, pick_pos
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

        if self.b_drag_object[0] >= 0 :
            ray_pos, ray_dir = ogl.get_cursor_ray(window.window, point)
            depth = np.linalg.norm(ray_pos- self.b_drag_object[2])
            self.b_drag_object[3] = ray_pos + depth * ray_dir
        else:
            ogl.mouse_move(point)
            #window.display()

    def draw_floor(self):
        mate  = np.array([[0.2,0.2,0.2,0.5],
                          [0.2,0.2,0.2,0.5],[0.2,0.2,0.2,0.5],[1.0,0,0,0]], dtype=np.float32)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT  , mate[0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE  , mate[1])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR , mate[2])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mate[3])

        glBegin(GL_QUADS)
        glNormal3d(0.,1.,0.)
        glVertex3d(-FLOOR_SIZE[0], 0., -FLOOR_SIZE[0])
        glVertex3d( FLOOR_SIZE[0], 0., -FLOOR_SIZE[0])
        glVertex3d( FLOOR_SIZE[0], 0.,  FLOOR_SIZE[0])
        glVertex3d(-FLOOR_SIZE[0], 0.,  FLOOR_SIZE[0])
        glEnd()


    def func_draw_scene(self, ogl, window):
        glEnable(GL_LIGHTING)
        self.draw_floor()
        for b in self.balls:
            b.draw()


    def step(self) :
        for i, b in enumerate(self.balls):
            pick_const = [-1, np.zeros(3, dtype=np.float32)]
            if self.b_drag_object[0] == i :
                pick_const = [self.b_drag_object[1], self.b_drag_object[3]]
            elif i == 0 :
                pick_const = [10, np.array([1,6,1], dtype=np.float32)]
            elif i == 1 :
                pick_const = [20, np.array([1,8,4], dtype=np.float32)]
            elif i == 2 :
                pick_const = [30, np.array([6,8,1], dtype=np.float32)]

            b.step(0.04, pick_const)



# class MainDialog
# tkinter のダイアログ
# このクラスのインスタンスとして glfwを持たせる
# tkinterのself.root.after 関数を利用してtimer機能を実装
#
# memo 
# tkinter の tk.mainloop() 中でも glfwのイベントを受け取れるので
# tkinter / glfw の両方を利用することができる
#
class MainDialog(ttk.Frame):

    def __init__(self, root_):
        super().__init__(root_)

        #initialize glfw frames
        self.manager = EventManager()

        #generate instance of GlfwMainWindow
        self.mainwindow = OglForGlfw.GlfwMainWindow(
            "Main Window", [800, 600], [100,100],
            self.manager.func_Ldown, self.manager.func_Lup,
            self.manager.func_Rdown, self.manager.func_Rup,
            self.manager.func_Mdown, self.manager.func_Mup,
            self.manager.func_mouse_move,
            self.manager.func_draw_scene)

        # memo :  通常はイベント待ちをするけど それはtkinterのmainloopに任せる
        # memo : (今回はsimulatorなので，mainloopではなくon_timer)
        #while not ( mainwindow.window_should_close()):
        #    mainwindow.wait_events_timeout()

        #initialize tkinter Frame
        self.root = root_
        self.pack()
        self.param = tk.StringVar()
        self.label1 = ttk.Label(self,text="------Simple Simulator-----")
        self.label1.pack(side="top")

        self.label2 = ttk.Label(self,text="time")
        self.label2.pack(side="top", anchor=tk.W)

        button = ttk.Button(self,text="Quit",command = self.quit_simulator )
        button.pack(side="top")

        #timerを起動
        self.on_timer()

    def on_timer(self):
        tmp = str(time.monotonic())
        self.label2.configure(text=time.strftime("%H:%M:%S  ") + tmp)

        self.manager.step()
        self.mainwindow.display()

        self.root.after(10, self.on_timer)

    #パラメータを入力するモーダルダイアログを開く
    def quit_simulator(self):
        exit()






def main():

    if not glfw.init():
        raise RuntimeError("Fails to initialize glfw")

    app  = tk.Tk()
    app.title("Simple Rigid Sim dlg")
    app.geometry("200x100")
    dialog = MainDialog(app)

    tk.mainloop()

    print("finish and terminate glfw")
    glfw.terminate()

if __name__ == "__main__":
    main()
