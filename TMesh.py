
import glfw
import numpy as np
from OpenGL.GL import *


# cals intersection between 
# ray (ray_pos + t * ray_dir) with triangle (x0, x1, x2)
# return True_or_False , intersect_pos
def t_intersect_ray_and_triangle ( ray_pos, ray_dir, x0, x1, x2 ) :
    A = np.array( [ [x1[0] - x0[0], x2[0] - x0[0], -ray_dir[0]],
                    [x1[1] - x0[1], x2[1] - x0[1], -ray_dir[1]],
                    [x1[2] - x0[2], x2[2] - x0[2], -ray_dir[2]] ], dtype=np.float32)
    stu = np.dot( np.linalg.inv(A), (ray_pos - x0) )

    if  0 <= stu[0] and stu[0] <= 1 and \
        0 <= stu[1] and stu[1] <= 1 and \
        0 <= stu[0] + stu[1] and stu[0] + stu[1] <= 1 :
        return True, ray_pos + stu[2] * ray_dir
    return False, np.zeros(3, dtype=np.float32)


# -- class TMesh -- 
# 3D triangle mesh model 
class TMesh:

    # init_as should be "Non", "Cube", "Sphere", "Obj"
    # radi is radius         used only when init_as == "Cube" or "Sphere"
    # fname is obj file name used only when nnit_as == "Obj"
    def __init__(self, init_as = "NON", radi = 1.0, fname = ""):
        # vertices/normals/textureCoord
        self.verts = np.array((1,3))
        self.norms = np.array((1,3))
        self.tex2d = np.array((1,2))

        #Triangle vertex_ids/texcd_ids/normals
        self.pvids = np.array((1,3)) # polygon vertex_idx (i0,i1,i2)
        self.ptids = np.array((1,3)) # polygon texcd_ idx (t0,t1,t2)
        self.pnrms = np.array((1,3))

        # OpenGL VBP name : vertex/normal1/norma2/texcd/index
        # normal:smooth rendering, normal2:visualize polygon edge
        self.gl_buffers = [0,0,0,0]

        if init_as == "Cube" :
            self.init_as_cube(radi)
        elif  init_as == "Sphere" :
            self.init_as_sphere(radi)
        elif  init_as == "Obj" :
            self.init_from_obj(fname)


    def init_as_cube(self, r):
        print("init_as_cube", r)
        r = np.float32(r)
        self.verts = np.array([[0.,0.,0.], [r,0.,0.], [r,r,0], [0,r,0],
                               [0.,0.,r ], [r,0.,r ], [r,r,r], [0,r,r]], dtype = np.float32)
        self.tex2d = np.array([[0.,0.], [1.,0.], [1.,1.], [0.,1.],
                               [0.,0.], [1.,0.], [1.,1.], [0.,1.]], dtype = np.float32)
        self.pvids = np.array([[0,2,1],[0,3,2],[0,1,5],[0,5,4],[1,2,6],[1,6,5],
                               [2,7,6],[2,3,7],[3,4,7],[3,0,4],[4,5,6],[4,6,7]], dtype = np.uint32)
        self.ptids = self.pvids

        self.update_normals()


    def init_as_sphere(self, r=1.0, reso_verti = 20, reso_hori = 20):
        vs = []
        ps = []
        step_phi   =   np.pi / (reso_verti + 1.0)
        step_theta = 2*np.pi / (reso_hori       )

        vs.append([0.,0.,-1.] ) #南極
        for phi_i in range(reso_verti):
            for theta_i in range(reso_hori):
                phi   = step_phi   * (phi_i + 1) - 0.5 * np.pi
                theta = step_theta * theta_i
                vs.append([ np.cos(theta)*np.cos(phi),  np.sin(theta)*np.cos(phi), np.sin(phi) ])
        vs.append([0.,0.,1]) #北極

        #Buttom(南極) / body / Top (北極) 
        for theta_i in range(reso_hori):
            ps.append([ 0, 1+(theta_i + 1)%reso_hori, 1+theta_i])

        for phi_i in range(reso_verti - 1) :
            for theta_i in range( reso_hori ) :
                i0 = 1+  phi_i    * reso_hori +  theta_i
                i1 = 1+  phi_i    * reso_hori + (theta_i + 1)%reso_hori
                i2 = 1+ (phi_i+1) * reso_hori + (theta_i + 1)%reso_hori
                i3 = 1+ (phi_i+1) * reso_hori +  theta_i
                ps.append([i0,i1,i2])
                ps.append([i0,i2,i3])

        n = len(vs) - 1
        for theta_i in range(reso_hori) :
            ps.append( [n, n-reso_hori+theta_i, n-reso_hori+(theta_i+1)%reso_hori] )

        self.verts = r * np.array(vs, dtype = np.float32)
        self.tex2d = np.zeros((self.verts.shape[0], 2), dtype = np.float32)
        self.pvids = np.array(ps, dtype = np.uint32)
        self.ptids = self.pvids
        self.update_normals()


    def init_from_obj(self, fname):
        vs, uvs,  = [], [] #vertex (3d), uvs (3d)
        ps, puvs  = [], [] #polygon_verts_ids(i0,i1,i2), polygon_uv_ids

        with open(fname) as fp:
            lines = fp.readlines()
            for line in lines:
                t = line.split()

                if len(t) < 2 :
                    print(t)
                elif t[0] == "vt":
                    uvs.append([np.float32(t[1]), np.float32(t[2])])
                elif t[0] == "v" :
                    vs.append([np.float32(t[1]), np.float32(t[2]), np.float32(t[3])])
                elif t[0] == "f" :
                    # f vidx/tidx/nidx vidx/tidx/nidx vidx/tidx/nidx (vert/tex/norm)
                    if len(t) < 4 : continue
                    t1, t2, t3 = t[1].split("/"), t[2].split("/"), t[3].split("/")
                    if len(t1) == 1 :
                        ps  .append([int(t1[0])-1, int(t2[0])-1, int(t3[0])-1])
                        puvs.append([           0,            0,            0])
                    elif len(t1) >= 2  :
                        # [f a/b a/b a/b] or [f a/b/c a/b/c a/b/c]
                        ps  .append([int(t1[0])-1, int(t2[0])-1, int(t3[0])-1])
                        if len(t1[1]) > 0 :
                            puvs.append([int(t1[1])-1, int(t2[1])-1, int(t3[1])-1])

        self.verts = np.array(vs  , dtype = np.float32)
        self.tex2d = np.array(uvs , dtype = np.float32)
        self.pvids = np.array(ps  , dtype = np.uint32)
        self.ptids = np.array(puvs, dtype = np.uint32)

        if len(uvs) == 0 :
            self.tex2d = np.zeros((self.verts.shape[0], 2), dtype = np.float32)
        if len(puvs) == 0 :
            self.ptids = self.pvids

        self.update_normals()
        print("loaded object file info : ", self.verts.shape[0], self.tex2d.shape[0], self.pvids.shape[0], self.ptids.shape[0] )


    def update_normals(self):
        num_verts = self.verts.shape[0]
        num_polys = self.pvids.shape[0]
        self.norms = np.zeros((num_verts, 3), dtype = np.float32)
        self.pnrms = np.zeros((num_polys, 3), dtype = np.float32)

        for p in range(num_polys):
            i0 = self.pvids[p,0]
            i1 = self.pvids[p,1]
            i2 = self.pvids[p,2]
            self.pnrms[p] = np.cross( self.verts[i1] - self.verts[i0],
                                      self.verts[i2] - self.verts[i0])
            norm_len = np.linalg.norm(self.pnrms[p])
            if norm_len != 0 :
                self.pnrms[p] /= norm_len

            self.norms[i0] = self.norms[i0] + self.pnrms[p]
            self.norms[i1] = self.norms[i1] + self.pnrms[p]
            self.norms[i2] = self.norms[i2] + self.pnrms[p]

        for v in range(num_verts) :
            n = np.linalg.norm( self.norms[v] );
            if n > 0 : self.norms[v] /= n


    #norm_mode should be "smooth" or "polygon"
    def draw_by_glfuncs(self, ambi, diff, spec, shin, norm_mode = "smooth"):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT  , ambi)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE  , diff)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR , spec)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin)

        num_polys = self.pvids.shape[0]
        glBegin(GL_TRIANGLES)
        if norm_mode == "smooth" :
            for p in range(num_polys):
                i0, i1, i2 = self.pvids[p,0], self.pvids[p,1], self.pvids[p,2]
                glNormal3fv(self.norms[i0])
                glVertex3fv(self.verts[i0])
                glNormal3fv(self.norms[i1])
                glVertex3fv(self.verts[i1])
                glNormal3fv(self.norms[i2])
                glVertex3fv(self.verts[i2])
        else :
            for p in range(num_polys):
                i0, i1, i2 = self.pvids[p,0], self.pvids[p,1], self.pvids[p,2]
                glNormal3fv(self.pnrms[p])
                glVertex3fv(self.verts[i0])
                glVertex3fv(self.verts[i1])
                glVertex3fv(self.verts[i2])
        glEnd()


    # rendering with VBO
    # when modify the mesh, call "clear_VBO()" befor calling this function
    def draw_by_VBO(self, ambi, diff, spec, shin, norm_mode = "smooth", use_uv = False):

        num_polys = self.pvids.shape[0]
        if self.gl_buffers[0] == 0 :
            #initialize VBOs
            self.gl_buffers = glGenBuffers(5)
            print(self.gl_buffers)
            vs  = np.zeros(num_polys*9, dtype=np.float32)
            ns1 = np.zeros(num_polys*9, dtype=np.float32) # smooth
            ns2 = np.zeros(num_polys*9, dtype=np.float32) # polygon
            ts  = np.zeros(num_polys*9, dtype=np.float32)
            ids = np.zeros(num_polys*3, dtype=np.uint32)
            for p in range(num_polys):
                i0, i1, i2, piv = self.pvids[p,0], self.pvids[p,1], self.pvids[p,2], 9*p
                vs[piv+0:piv+3], ns1[piv+0:piv+3], ns2[piv+0:piv+3] = self.verts[i0,:], self.norms[i0,:], self.pnrms[p]
                vs[piv+3:piv+6], ns1[piv+3:piv+6], ns2[piv+3:piv+6] = self.verts[i1,:], self.norms[i1,:], self.pnrms[p]
                vs[piv+6:piv+9], ns1[piv+6:piv+9], ns2[piv+6:piv+9] = self.verts[i2,:], self.norms[i2,:], self.pnrms[p]
                i0, i1, i2, piv = self.ptids[p,0], self.ptids[p,1], self.ptids[p,2], 6*p
                ts[piv+0:piv+2] = self.tex2d[i0,:]
                ts[piv+2:piv+4] = self.tex2d[i1,:]
                ts[piv+4:piv+6] = self.tex2d[i2,:]
            ids = np.arange(num_polys*3, dtype=np.uint32)

            #send verts/norms/uvs/ids
            glBindBuffer(GL_ARRAY_BUFFER, self.gl_buffers[0])
            glBufferData(GL_ARRAY_BUFFER, 4*num_polys*9, vs, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.gl_buffers[1])
            glBufferData(GL_ARRAY_BUFFER, 4*num_polys*9, ns1, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.gl_buffers[2])
            glBufferData(GL_ARRAY_BUFFER, 4*num_polys*9, ns2, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, self.gl_buffers[3])
            glBufferData(GL_ARRAY_BUFFER, 4*num_polys*6, ts, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gl_buffers[4])
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*num_polys*3, ids, GL_STATIC_DRAW)

        #rendering with VBO
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT  , ambi)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE  , diff)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR , spec)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        if use_uv :
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        #vertices/normal/texcd/indices
        glBindBuffer(GL_ARRAY_BUFFER, self.gl_buffers[0])
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.gl_buffers[1] if norm_mode == "smooth" else self.gl_buffers[2] )
        glNormalPointer(GL_FLOAT, 0, None)
        if use_uv :
            glBindBuffer(GL_ARRAY_BUFFER, self.gl_buffers[3])
            glTexCoordPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.gl_buffers[4])
        glDrawElements(GL_TRIANGLES, num_polys * 3, GL_UNSIGNED_INT, None)


    def clear_VBO(self):
        if self.gl_buffers[0] != 0 :
            glDeleteBuffers(5, self.gl_buffers)
        self.gl_buffers = np.zeros(5, dtype=int)

    # calc intersection between ray (ray_pos + t*ray_dir) with this mesh
    # return (1) closest_vertex_idx (-1 when ray doesnot intersect) 
    #        (2) intersection point
    def pick(self, ray_pos, ray_dir, rot, trans) :
        pick_vid = -1
        pick_pos = np.zeros(3, dtype=np.float32)
        pick_depth = 1000000

        for p in self.pvids :
            x0 = np.dot(rot, self.verts[p[0]]) + trans
            x1 = np.dot(rot, self.verts[p[1]]) + trans
            x2 = np.dot(rot, self.verts[p[2]]) + trans
            tf, pos = t_intersect_ray_and_triangle(ray_pos, ray_dir, x0,x1,x2)
            if not tf : continue
            d = np.linalg.norm(ray_pos - pos)
            if d < pick_depth :
                pick_depth = d
                pick_pos = pos
                d1, d2, d3 = np.linalg.norm(x0 - pos), np.linalg.norm(x1 - pos), np.linalg.norm(x2 - pos)
                if d1 <= d2 and d1 <= d3 : pick_vid = p[0]
                elif d2 <= d3 : pick_vid = p[1]
                else : pick_vid = p[2]

        return pick_vid, pick_pos
