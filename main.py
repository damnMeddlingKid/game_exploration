import numpy as np
import pygame

WIDTH, HEIGHT = 960, 540
FPS = 60
TITLE = "Game Exploration"
BG_COLOR = (20, 24, 32)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.triangles = np.array([
            (-1,-1,-1), 
            (1,-1,-1), 
            (0,1,-1),
            (-1,-1,1), 
            (1,-1,1), 
            (0,1,1)
        ], dtype=np.float32)

        self.triangles = np.array([
        # front (z = -1)
        (-1,-1,-1), ( 1,-1,-1), ( 1, 1,-1),
        (-1,-1,-1), ( 1, 1,-1), (-1, 1,-1),
        # back (z = +1)
        ( 1,-1, 1), (-1,-1, 1), (-1, 1, 1),
        ( 1,-1, 1), (-1, 1, 1), ( 1, 1, 1),
        # left (x = -1)
        (-1,-1, 1), (-1,-1,-1), (-1, 1,-1),
        (-1,-1, 1), (-1, 1,-1), (-1, 1, 1),
        # right (x = +1)
        ( 1,-1,-1), ( 1,-1, 1), ( 1, 1, 1),
        ( 1,-1,-1), ( 1, 1, 1), ( 1, 1,-1),
        # top (y = +1)
        (-1, 1,-1), ( 1, 1,-1), ( 1, 1, 1),
        (-1, 1,-1), ( 1, 1, 1), (-1, 1, 1),
        # bottom (y = -1)
        (-1,-1, 1), ( 1,-1, 1), ( 1,-1,-1),
        (-1,-1, 1), ( 1,-1,-1), (-1,-1,-1),
        ], dtype=np.float32)

        self.triangle_position = np.array([0,0,5])
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
        rotated_tris = self.rotate(self.triangles, self.triangle_y_rot)
        translated_tris = rotated_tris + self.triangle_position
        view_tris = self.world_to_view(translated_tris)
        projected_tris = self.project(view_tris)
        screen_tris = self.view_to_screen(projected_tris)
        start_tri = None
        
        for (idx, tri) in enumerate(screen_tris):
            pygame.draw.circle(self.screen, color="white", center=pygame.Vector2(tri[0], tri[1]), radius=5)
            if idx % 3 == 0:
                close_tri = screen_tris[idx-3]
                last_tri = screen_tris[idx-1]
                pygame.draw.line(self.screen, "white",pygame.Vector2(last_tri[0], last_tri[1]), pygame.Vector2(close_tri[0], close_tri[1]))
            else:
                pygame.draw.line(self.screen, "white",pygame.Vector2(start_tri[0], start_tri[1]), pygame.Vector2(tri[0], tri[1]))
            start_tri = tri

        close_tri = screen_tris[-3]
        last_tri = screen_tris[-1]

        pygame.draw.line(self.screen, "white",pygame.Vector2(last_tri[0], last_tri[1]), pygame.Vector2(close_tri[0], close_tri[1]))
        
        
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
