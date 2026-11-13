# Learning plan: a 3D camera and software rasterizer (numpy version)

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

# Part 2: Rasterization and texture mapping

The wireframe camera is the core of the course. Part 2 fills triangles yourself, pixel by pixel, which is what makes texturing possible.

## Stage 8: Triangle rasterizer ✅ (done early)
**Goal:** fill triangles yourself instead of with `pygame.draw.polygon`.
- For each triangle, loop over its on-screen bounding box, clamped to the screen.
- **Edge functions:** for edge a → b with perpendicular `n`, a pixel p is on the inside when `n · (p − a)` has the inside sign. A pixel is inside the triangle when all 3 edges agree.
- **Signed area** `v1.x·v2.y − v1.y·v2.x` gives the on-screen winding. Use its sign to skip back faces before touching any pixels, and skip triangles with an area of 0.
- Write pixels into a numpy array (`pygame.surfarray.pixels3d`, indexed `[x, y]`) and compile the loop with numba `@njit`.

## Stage 9: Barycentric coordinates
**Goal:** know *where* inside the triangle each pixel is, not just *whether* it's inside.
- Divide each edge function by the triangle's signed area: `w0, w1, w2 = e_opposite / area`.
  - Each weight belongs to the corner **opposite** its edge.
  - All three are ≥ 0 inside the triangle, and they always add up to 1.
- Any value stored per corner can now be blended at a pixel: `value = w0·value0 + w1·value1 + w2·value2`.
- **Check:** give the corners pure red, green and blue, and blend them per pixel. You should see a smooth color gradient, with each corner's pure color at that corner. These are the vertex colors pygame couldn't do.
- **Experiment:** at a corner pixel, which weight is 1? Along an edge, which weight is 0?

## Stage 10: Texture mapping
**Goal:** paint an image onto triangles.

**10a. Texture data**
- Load the image with `pygame.image.load`, then convert it with `pygame.surfarray.array3d`. That gives a `(tex_w, tex_h, 3)` uint8 array indexed `[x, y]`, which numba can read.
- Start with a **checkerboard** (generate it with numpy, or draw one). Distortion is easy to spot on a checkerboard and hard to spot on a photo.

**10b. UV coordinates from OBJ**
- `vt u v` lines hold texture coordinates, from 0 to 1 across the image. The **second** index in each face group (`v/vt/vn`) chooses one for that corner.
- The `vt` indices are **separate** from the `v` indices: one vertex can have different UVs in different faces, at a *seam*. So store UVs **per triangle corner**, for example a `(T, 3, 2)` array, not per vertex.
- Do the same index swap you use for winding on the UV indices, or the UVs end up on the wrong corners.
- `box.obj` already has UVs. Suzanne needs a UV unwrap in Blender first (UV → Smart UV Project), then export with UVs turned on.

**10c. Looking up a texel**
- Blend the corner UVs with the barycentric weights to get the pixel's (u, v).
- Convert to texel coordinates: `tx = u·(tex_w − 1)`, `ty = (1 − v)·(tex_h − 1)`.
  - **The v flip:** in OBJ files, v = 0 is the **bottom** of the image. In image arrays, row 0 is the **top**.
- **Nearest-neighbour sampling:** round to the nearest whole texel and read it.
- **Out-of-range UVs:** decide between **wrap** (`u mod 1`, which tiles the image) and **clamp** (limit u to 0..1).
- Shade it: multiply the texel color by the face's lighting intensity.

**10d. Affine vs perspective-correct**
- Blending UVs with the screen-space weights (10c) is called **affine** mapping. Look at a textured face up close, at an angle: the checkerboard lines **bend**, especially along the triangle's diagonal. This is the wobbly look of early 3D games.
- The reason: equal steps across the **screen** aren't equal steps across the **surface**. A triangle's far half takes up less of the screen than its near half.
- **Fix:** values divided by z *do* change evenly across the screen. Per pixel:
  1. blend `u/z`, `v/z` and `1/z` with the barycentric weights
  2. divide: `u = (u/z) / (1/z)`, and the same for v
- **Exercise:** derive why 1/z changes evenly across the screen, when z itself doesn't. Start from the projection formula x′ = f·x/z and a straight line in view space.
- **Check:** with the perspective-correct version, checkerboard lines stay straight at any angle.

**10e. Extensions (optional)**
- **Bilinear filtering:** blend the 4 nearest texels using the fractional parts of tx and ty. This smooths blocky textures up close.
- **Mipmaps:** pre-shrunk copies of the texture, to stop distant surfaces shimmering.

## Stage 11: Depth buffer (deferred)
**Goal:** correct overlap for every pixel, without sorting.
- Keep a `(W, H)` float array, set to infinity at the start of each frame.
- Per pixel, blend **1/z** with the barycentric weights. You already do this in Stage 10d.
  - Larger 1/z means **closer**.
  - Only write the pixel if it's closer than the value already stored, then store the new value.
- Painter's sorting is no longer needed, and back-face culling becomes just a speed-up.
- **Check:** Suzanne's ears and nose overlap correctly from every angle.

---

### Common mistakes to watch for
- A missing y flip, which draws everything upside down.
- Rotating the camera by +yaw instead of −yaw.
- Mixing up degrees and radians.
- Applying transforms in the wrong order.
- Dividing by z before skipping or clipping points behind the camera.
- Integer arrays: create arrays with `dtype=float`, or results get rounded to whole numbers.
- Textures: forgetting the v flip (textures upside down), indexing `[y, x]` instead of `[x, y]` on surfarray arrays, and blending u and v instead of u/z and v/z (bending lines).
- uint8 color math: values above 255 wrap around to dark colors. Clip before converting.

**Suggested file layout:** `transforms.py` (Stages 2–4), `camera.py` (Stages 3–5), `mesh.py` (Stage 6), plus `main.py` as the game loop.
