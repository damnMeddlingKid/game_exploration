# Learning plan: a 3D wireframe camera (numpy version)

- numpy handles vector math (`+`, `-`, `*`, `np.dot`, `np.cross`, `np.linalg.norm`).
- You write every transform formula yourself, one component at a time. There are no matrices until the final stage.

**Conventions:** y up, x right, camera looks down +z, angles in radians, screen y points down (flipped once, at the very last step).

**Tip:** store a mesh's vertices as one `(N, 3)` float array instead of a list of points. Then each formula works on every vertex at once, for example `x = pts[:, 0]`. When you switch to matrices later, a whole transform becomes `pts @ M.T`.

---

## Stage 1: Points and the perspective projection
**Goal:** the triangle's 3 vertices appear as dots, in perspective.
- Triangle: `np.array([[-1,-1,5], [1,-1,5], [0,1,5]], dtype=float)`
- Perspective from similar triangles: `x' = f·x/z`, `y' = f·y/z`
- Focal length from field of view: `f = (W/2) / tan(fov/2)`. With `f` in pixels, the next step only needs to center the result.
- Map to the screen (the **viewport**): `sx = W/2 + x'`, `sy = H/2 - y'`. The minus sign is the y flip.
- Skip any point with `z <= near` (for example 0.1). Use a boolean mask such as `pts[:, 2] > near`.
- **Check:** the tip of the triangle is at the top. Moving the triangle along z makes it shrink toward the screen center. Changing the FOV zooms.

## Stage 2: Model transforms
**Goal:** spin and place the triangle.
- Scale: `pts * s`. Translate: `pts + t`.
- Write each rotation out by hand, on whole columns:
  - Around Y: `x' = x·cosθ + z·sinθ`, `z' = -x·sinθ + z·cosθ`
  - Around X: `y' = y·cosθ - z·sinθ`, `z' = y·sinθ + z·cosθ`
  - Around Z: `x' = x·cosθ - y·sinθ`, `y' = x·sinθ + y·cosθ`
- Order: scale, then rotate, then translate. Define the triangle around its own center (z = 0), then translate it to z = 5.
- **Check:** the triangle spins in place. Swap the order to translate-then-rotate and it orbits instead.

## Stage 3: Camera position
**Goal:** move the camera instead of the object.
- The camera never moves; you shift the world the opposite way: `p_view = p_world - cam_pos`
- Hook WASD up to `cam_pos`. Moving the camera right should slide the triangle left.

## Stage 4: Camera rotation (yaw and pitch)
**Goal:** look around with the mouse.
- Undo the camera's rotation: subtract `cam_pos`, rotate by **−yaw** around Y, then by **−pitch** around X. Reuse your Stage 2 functions.
- Calculate the movement directions from yaw so W moves you where you're looking:
  `forward = (sin yaw, 0, cos yaw)`, `right = (cos yaw, 0, -sin yaw)`
- Limit pitch to about ±89°. Read mouse movement from `event.rel`, with `pygame.mouse.set_relative_mode(True)` to lock the cursor.
- **Check:** you can walk around the triangle and look at it from behind.

## Stage 5: Basis vectors and look-at (the bridge to matrices)
**Goal:** describe the camera with 3 axis vectors instead of angles.
- Build the camera's axes:
  `forward = normalize(target - eye)`
  `right = normalize(np.cross(world_up, forward))`
  `up = np.cross(forward, right)`
- Convert points into camera space with three dot products: with `d = pts - eye`, compute
  `x_v = d @ right`, `y_v = d @ up`, `z_v = d @ forward`
- **Check:** this gives the same picture as Stage 4. Add an orbit camera that always looks at the triangle.
- **Lesson:** stack `right`, `up`, and `forward` as rows and you have the 3×3 view rotation matrix.

## Stage 6: Wireframe
**Goal:** draw lines instead of dots, then a cube.
- Store a mesh as `vertices` (an `(N, 3)` array) plus `edges` (index pairs). A triangle has 3 edges; a cube has 8 vertices and 12 edges.
- Transform all vertices once, then draw each edge with `pygame.draw.line`.
- **Near-plane clipping:** when one end of an edge is behind the camera, cut the edge where it crosses the near plane:
  `t = (near - z1) / (z2 - z1)`, `p = p1 + t·(p2 - p1)`
- **Check:** you can walk through the cube without lines shooting across the screen.

## Stage 7: Review the pipeline and switch to matrices
```
model → world → view → projection → perspective divide → viewport
```
- Every step except the divide is "multiply by numbers, then add," so each one can become a matrix.
- Translation doesn't fit a 3×3 matrix. Adding a 4th coordinate `w = 1` fixes that, which gives **4×4 matrices**.
- Perspective becomes "copy z into w, then divide by w at the end."
- **Exercise:** build each stage as a 4×4 matrix, chain them as `pts_h @ (P @ V @ M).T`, and check the picture matches your hand-written version.

---

### Common mistakes to watch for
- A missing y flip, which draws everything upside down.
- Rotating the camera by +yaw instead of −yaw.
- Mixing up degrees and radians.
- Applying transforms in the wrong order.
- Dividing by z before skipping or clipping points behind the camera.
- Integer arrays: create arrays with `dtype=float`, or results get rounded to whole numbers.

**Suggested file layout:** `transforms.py` (Stages 2–4), `camera.py` (Stages 3–5), `mesh.py` (Stage 6), plus `main.py` as the game loop.
