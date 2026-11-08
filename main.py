import numpy as np
import pygame

WIDTH, HEIGHT = 960, 540
FPS = 60
TITLE = "Game Exploration"
BG_COLOR = (20, 24, 32)

def load_obj(file_name):
    vertices = []
    indices = []
    with open(file_name, 'r') as f:
        for line in f:
            parts = line.split(' ')
            if parts[0] == 'v':
                vertices.append((float(parts[1]), float(parts[2]), -float(parts[3])))
            elif parts[0] == 'f':
                temp = []
                for index in parts[1:]:
                    index_parts = index.split('/')
                    temp.append(int(index_parts[0]) - 1)
                # swap two indices to correct winding order
                temptemp = temp[0]
                temp[0] = temp[1]
                temp[1] = temptemp
                indices.append(temp)

    vertices = np.array(vertices, dtype=np.float64)
    tris = vertices[indices]
    point1, point2, point3 = tris[:, 0], tris[:, 1], tris[:, 2]
    e1 = point2 - point1
    e2 = point3 - point1
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals/lengths
    return vertices, normals, indices

class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.running = True

        vertices, normals, indices = load_obj("suzanneg.obj")
        self.triangle_vertices = vertices
        self.triangle_normals = normals
        self.triangle_indices = indices
        self.light_direction = np.array([1,-1,0])
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)

        self.triangle_position = np.array([0,0,3])
        self.triangle_y_rot = 0
        
        fov = np.deg2rad(60)
        # tan(fov/2) = HEIGHT/2 / f
        # f = (HEIGHT/2)/tan(fov/2)
        # viewx/f = x/z, # viewy/f = y/z
        # viewx = f*(x/z), viewy = f*(y/z)
        self.focal_z = (HEIGHT/2)/np.tan(fov/2)
        self.camera_z = 0

    def world_to_view(self, points):
        return points + np.array([0,0, -self.camera_z])

    def project(self, view_points):
        normalized_points = view_points / view_points[:, [-1]]
        return self.focal_z * normalized_points

    def view_to_screen(self, projected_points):
        screen_points = np.multiply(np.array([1, -1, 0]), projected_points)
        return screen_points[:,:2] + np.array([WIDTH/2, HEIGHT/2])

    def rotate(self, points, rad):
        """
        rotate in y axis
        a' = a cos θ − b sin θ
        b' = a sin θ + b cos θ
        NX3 3x3
        [a,b,c][
            [cos(θ) ,0,sin θ],
            [0      ,1,0    ],
            [-sin(θ),0,cos θ],
        ]
        """
        rot_matrix = np.array([
            [np.cos(rad),0,np.sin(rad)],
            [0,1,0],
            [-np.sin(rad),0,np.cos(rad)],
        ])
        # TODO fix the matrix instead of the transpose.
        return points @ rot_matrix.T
        
    def handle_events(self) -> None:
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            self.camera_z += 0.1
        if keys[pygame.K_DOWN]:
            self.camera_z -= 0.1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False

    def update(self, dt: float) -> None:
        self.triangle_y_rot += 0.5*dt

    def draw(self) -> None:
        self.screen.fill(BG_COLOR)
        rotated_tris = self.rotate(self.triangle_vertices, self.triangle_y_rot)
        rotated_normals = self.rotate(self.triangle_normals, self.triangle_y_rot)
        translated_tris = rotated_tris + self.triangle_position
        view_tris = self.world_to_view(translated_tris)
        projected_tris = self.project(view_tris)
        screen_vertices = self.view_to_screen(projected_tris)

        face_brightness = np.clip(np.dot(rotated_normals, self.light_direction), a_min=0.0, a_max=None)
        ambient = 0.2
        intensity = ambient + (1 - ambient) * face_brightness
        #base = np.array([90, 200, 250])
        base = np.array([250, 250, 250])
        colors = intensity[:, None] * base
        colors = colors.astype(np.uint8)



        # N X 3 X 3
        triangles = view_tris[self.triangle_indices]
        face_depth = np.average(triangles[:, :, 2], axis=1)
        tri_indices = np.argsort(-face_depth)

        for idx in tri_indices:
            tri_idx = self.triangle_indices[idx]
            vertices = screen_vertices[tri_idx]
            pygame.draw.polygon(self.screen, colors[idx], vertices)
            # pygame.draw.circle(self.screen, color="black", center=vertices[0], radius=1)
            # pygame.draw.circle(self.screen, color="black", center=vertices[1], radius=1)
            # pygame.draw.circle(self.screen, color="black", center=vertices[2], radius=1)
            

        fps_text = self.font.render(f"FPS: {self.clock.get_fps():.0f}", True, (200, 200, 200))
        self.screen.blit(fps_text, (10, 10))

        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS) / 1000  # seconds since last frame
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
