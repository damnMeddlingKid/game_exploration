import time

import numpy as np
import pygame
from PIL import Image   
from numba import njit

WIDTH, HEIGHT = 960, 540
FPS = 60
TITLE = "Game Exploration"
BG_COLOR = (20, 24, 32)

def load_obj(file_name):
    vertices = []
    indices = []
    tex_idx = []
    uv = []
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.split(' ')
            if parts[0] == 'v':
                vertices.append((float(parts[1]), float(parts[2]), -float(parts[3])))
            elif parts[0] == 'vt':
                uv.append((float(parts[1]), float(parts[2])))
            elif parts[0] == 'f':
                temp = []
                tex_temp = []
                for index in parts[1:]:
                    index_parts = index.split('/')
                    temp.append(int(index_parts[0]) - 1)
                    tex_temp.append(int(index_parts[1]) -1)
                    
                # swap two indices to correct winding order
                temptemp = temp[0]
                temp[0] = temp[1]
                temp[1] = temptemp
                
                indices.append(temp)

                temptex = tex_temp[0]
                tex_temp[0] = tex_temp[1]
                tex_temp[1] = temptex
                tex_idx.append(tex_temp)
                

    vertices = np.array(vertices, dtype=np.float64)
    uv = np.array(uv, dtype=np.float64)
    tris = vertices[indices]
    point1, point2, point3 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1 = point2 - point1
    e2 = point3 - point1
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals/lengths
    uv = uv[tex_idx]
    return vertices, normals, indices, uv

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.running = True
        self.frame_start = time.perf_counter()
        self.work_ms = 0.0

        vertices, normals, indices, uv = load_obj("garen/source/garentri.obj")
        self.triangle_vertices = vertices
        self.triangle_normals = normals
        self.triangle_indices = np.array(indices)
        self.triangle_uv = uv
        img = Image.open("garen/textures/garen_tex.jpg")
        self.triangle_texture = np.array(img)
        self.light_direction = np.array([1,-1,0])
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)

        self.triangle_position = np.array([0,-0.5,3], dtype=np.float64)
        self.triangle_y_rot = 0
        
        fov = np.deg2rad(60)
        # tan(fov/2) = HEIGHT/2 / f
        # f = (HEIGHT/2)/tan(fov/2)
        # viewx/f = x/z, # viewy/f = y/z
        # viewx = f*(x/z), viewy = f*(y/z)
        self.focal_z = (HEIGHT/2)/np.tan(fov/2)
        self.camera_position = np.array([0,0,-1], dtype=np.float64)
        self.camera_dir = np.array([0,0,0], dtype=np.float64)
        self.camera_rot = np.array([0,0,0], dtype=np.float64)
        self.camera_rotv = np.array([0,0,0], dtype=np.float64)
        self.look_dir = np.array([0, 0, 1], dtype=np.float64)
        self.depth_buffer = np.array([0]*WIDTH*HEIGHT, dtype=np.float64)
        self.move_velocity = 4.0

    def world_to_view(self, points):
        view_points = points - self.camera_position
        view_points = self.rotatey(view_points, -self.camera_rot[0])
        view_points = self.rotatex(view_points, -self.camera_rot[1])
        return view_points

    def project(self, view_points):
        normalized_points = view_points.copy()
        normalized_points[:, :2] /= view_points[:, [-1]]
        return self.focal_z * normalized_points

    def view_to_screen(self, projected_points):
        screen_points = np.multiply(np.array([1, -1, 1]), projected_points)
        return screen_points + np.array([WIDTH/2, HEIGHT/2, 0])

    def rotatey(self, points, rad):
        """
        rotate in y axis
        a' = a cos θ − b sin θ
        b' = a sin θ + b cos θ
        """
        rot_matrix = np.array([
            [np.cos(rad),0,-np.sin(rad)],
            [0,1,0],
            [np.sin(rad),0,np.cos(rad)],
        ])
        # TODO fix the matrix instead of the transpose.
        return points @ rot_matrix

    def rotatex(self, points, rad):
        """
        rotate in y axis
        a' = a cos θ − b sin θ
        b' = a sin θ + b cos θ
        """
        rot_matrix = np.array([
            [1,0,0],
            [0,np.cos(rad),np.sin(rad)],
            [0,-np.sin(rad),np.cos(rad)],
        ])
        return points @ rot_matrix
        
    def handle_events(self) -> None:
        keys = pygame.key.get_pressed()
        
        self.look_dir = np.array([0, 0, 1], dtype=np.float64)
        self.camera_dir = np.array([0, 0, 0], dtype=np.float64)
        mouse_pos = np.array(pygame.mouse.get_rel())
        #mouse_pos = np.array(pygame.mouse.get_pos()) - np.array((WIDTH/2, HEIGHT/2))
        normalized_pos = mouse_pos / np.array((WIDTH/2, HEIGHT/2))
        angle_max =  np.pi/2
        look_angle = normalized_pos * angle_max
        self.camera_rot[0]+=3*(mouse_pos[0]/WIDTH/2)
        self.camera_rot[1]+=3*(mouse_pos[1]/WIDTH/2)
        self.camera_rot[1]=np.clip(self.camera_rot[1], a_min=-angle_max, a_max=angle_max)
        self.camera_rot[0]=np.clip(self.camera_rot[0], a_min=-angle_max, a_max=angle_max)

        self.look_dir = self.rotatex(self.look_dir, self.camera_rot[1])
        self.look_dir = self.rotatey(self.look_dir, self.camera_rot[0])

        if keys[pygame.K_w]:
            self.camera_dir += self.look_dir
        if keys[pygame.K_s]:
            self.camera_dir += -self.look_dir

        camera_dir_norm = np.linalg.norm(self.camera_dir)
        if camera_dir_norm > 0:
            self.camera_dir = self.camera_dir / camera_dir_norm
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

    def update(self, dt: float) -> None:
        self.triangle_y_rot += 0.5*dt
        self.camera_position += self.camera_dir*self.move_velocity*dt

    def draw(self) -> None:
        self.screen.fill(BG_COLOR)
        rotated_tris = self.rotatey(self.triangle_vertices, self.triangle_y_rot)
        rotated_normals = self.rotatey(self.triangle_normals, self.triangle_y_rot)
        translated_tris = rotated_tris + self.triangle_position
        view_tris = self.world_to_view(translated_tris)
        projected_tris = self.project(view_tris)
        screen_vertices = self.view_to_screen(projected_tris)

        face_brightness = np.clip(np.dot(rotated_normals, self.light_direction), a_min=0.0, a_max=None)
        ambient = 0.5
        intensity = ambient + (1 - ambient) * face_brightness
        #base = np.array([90, 200, 250])
        base = np.array([250, 250, 250])
        colors = intensity[:, None] * base
        colors = colors.astype(np.uint8)

        # N X 3 X 3
        triangles = view_tris[self.triangle_indices]
        face_depth = np.average(triangles[:, :, 2], axis=1)
        draw_index = np.argsort(face_depth)

        # for idx in draw_index:
        #     tri_idx = self.triangle_indices[idx]
        #     vertices = screen_vertices[tri_idx]
        #     pygame.draw.polygon(self.screen, colors[idx], vertices)
        #     # pygame.draw.circle(self.screen, color="black", center=vertices[0], radius=1)
        #     # pygame.draw.circle(self.screen, color="black", center=vertices[1], radius=1)
        #     # pygame.draw.circle(self.screen, color="black", center=vertices[2], radius=1)
        # self.screen.fill((255, 0, 0))
        self.depth_buffer.fill(0)
        pixels = pygame.surfarray.pixels3d(self.screen)
        Game.jitted_render_triangles(pixels, draw_index, self.triangle_indices, screen_vertices, intensity, self.triangle_uv, self.triangle_texture, self.depth_buffer)
        #pygame.draw.circle(self.screen, "red", (WIDTH//2, HEIGHT//2), 100)
        del pixels

        work_ms = (time.perf_counter() - self.frame_start) * 1000
        self.work_ms = 0.95 * self.work_ms + 0.05 * work_ms  # smooth it so it's readable

        effective_fps = 1000 / self.work_ms if self.work_ms > 0 else 0.0
        fps_text = self.font.render(
            f"FPS: {self.clock.get_fps():.0f}   work: {self.work_ms:.1f} ms"
            f"   effective: {effective_fps:.0f} FPS",
            True,
            (200, 200, 200),
        )
        self.screen.blit(fps_text, (10, 10))

        pygame.display.flip()
        #import ipdb;ipdb.set_trace()

    @njit(cache=True)
    def jitted_render_triangles(px_array, draw_index, triangle_indices, screen_vertices,  intensity, texture_uv, texture, depth_buffer):        
        for idx in draw_index:
            tri_idx = triangle_indices[idx]
            vertices = screen_vertices[tri_idx]
            uv = texture_uv[idx]
            
            v1 = vertices[2] - vertices[0]
            v2 = vertices[1] - vertices[2]
            v3 = vertices[0] - vertices[1]
            
            min_x_px = int(np.min(vertices[:,0]))
            max_x_px = int(np.max(vertices[:,0]))
            min_y_px = int(np.min(vertices[:,1]))
            max_y_px = int(np.max(vertices[:,1]))

            # min_x = max(int(min(ax, bx, cx)), 0)
            # max_x = min(int(max(ax, bx, cx)), width - 1)
            # min_y = max(int(min(ay, by, cy)), 0)
            # max_y = min(int(max(ay, by, cy)), height - 1)

            
            tri_area = v1[1] * v2[0] + -v1[0] * v2[1]

            if tri_area <= 0: continue 

            uv0_u = uv[0][0]
            uv0_v = uv[0][1]

            uv1_u = uv[1][0]
            uv1_v = uv[1][1]

            uv2_u = uv[2][0]
            uv2_v = uv[2][1]

            inv_z0 = 1/vertices[0][2]
            inv_z1 = 1/vertices[1][2]
            inv_z2 = 1/vertices[2][2]
            

            for x in range(min_x_px, max_x_px+1):
                for y in range(min_y_px, max_y_px+1):
                    if x <0 or y <0 or x > WIDTH-1 or y > HEIGHT-1:
                        continue

                    n1 = v1[1]*(x-vertices[0][0]) -v1[0]*(y-vertices[0][1])
                    n2 = v2[1]*(x-vertices[2][0]) -v2[0]*(y-vertices[2][1])
                    n3 = v3[1]*(x-vertices[1][0]) -v3[0]*(y-vertices[1][1])

                    if (n1 >= 0 and n2 >= 0 and n3 >= 0):# or (n1 <= 0 and n2 <= 0 and n3 <= 0):
                        inv_z = (n1*inv_z1 + n2*inv_z0+ n3*inv_z2)/tri_area
                        
                        if  inv_z < depth_buffer[x + y*WIDTH]:
                            continue

                        depth_buffer[x + y*WIDTH] = inv_z
                        
                        i_uv_u = ((n1*uv1_u*inv_z1 + n2*uv0_u*inv_z0 + n3*uv2_u*inv_z2)/tri_area)/inv_z
                        i_uv_v = ((n1*uv1_v*inv_z1 + n2*uv0_v*inv_z0+ n3*uv2_v*inv_z2)/tri_area)/inv_z

                        tex_cord_row = int((texture.shape[0]-1) - i_uv_v*(texture.shape[0]-1))
                        tex_cord_col = int(i_uv_u*(texture.shape[1]-1))
                        color = texture[(tex_cord_row, tex_cord_col)]
                        px_array[x, y][0] = color[0] * intensity[idx]
                        px_array[x, y][1] = color[1] * intensity[idx]
                        px_array[x, y][2] = color[2] * intensity[idx]

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000  # seconds since last frame
            self.frame_start = time.perf_counter()
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
