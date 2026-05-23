import numpy as np
import logging
from UI.loupeProperties import VisualElement

logger = logging.getLogger("FFAST")
DEPENDENCIES = ["loupeAtoms"]

_SHAFT_RADIUS = 0.05
_HEAD_RADIUS = 0.12
_HEAD_LENGTH = 0.25   # absolute world-unit cone height, constant regardless of force magnitude
_N_SEGMENTS = 8
_COLOR_TOWARD = np.array([1.0, 0.55, 0.1, 1.0])  # warm orange — arrow tip points at camera
_COLOR_AWAY   = np.array([0.25, 0.5, 1.0, 1.0])  # blue — arrow tail points at camera


def _batch_rotation_z_to(U):
    """(N,3) unit vectors → (N,3,3) rotation matrices mapping +z to each U[i]."""
    N = len(U)
    R = np.tile(np.eye(3, dtype=float), (N, 1, 1))

    parallel = np.abs(U[:, 2]) > 0.9999
    antipar = parallel & (U[:, 2] < 0)
    R[antipar, 1, 1] = -1.0
    R[antipar, 2, 2] = -1.0

    sel = ~parallel
    if not np.any(sel):
        return R

    u = U[sel]
    z = np.zeros_like(u)
    z[:, 2] = 1.0
    axis = np.cross(z, u)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)

    c = u[:, 2, np.newaxis, np.newaxis]
    s = np.sqrt(np.maximum(0.0, 1.0 - c ** 2))

    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    M = len(u)
    K = np.zeros((M, 3, 3))
    K[:, 0, 1] = -kz
    K[:, 0, 2] = ky
    K[:, 1, 0] = kz
    K[:, 1, 2] = -kx
    K[:, 2, 0] = -ky
    K[:, 2, 1] = kx

    I = np.tile(np.eye(3, dtype=float), (M, 1, 1))
    KK = np.einsum("nij,njk->nik", K, K)
    R[sel] = I + s * K + (1.0 - c) * KK
    return R


def _build_arrow_mesh(starts, ends):
    """Batched cylinder+cone mesh. Returns (vertices (V,3), faces (F,3)) or (None, None)."""
    n = _N_SEGMENTS
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    js = np.arange(n)
    js1 = (js + 1) % n

    # Canonical shaft side: z in [0, 1], scaled per-arrow by shaft_length
    shaft_can = np.vstack([
        np.c_[_SHAFT_RADIUS * cos_a, _SHAFT_RADIUS * sin_a, np.zeros(n)],  # bot ring
        np.c_[_SHAFT_RADIUS * cos_a, _SHAFT_RADIUS * sin_a, np.ones(n)],   # top ring
    ])  # (2n, 3)
    shaft_faces = np.vstack([
        np.c_[js, js1, n + js],
        np.c_[js1, n + js1, n + js],
    ])  # (2n, 3)

    # Canonical cone side: z in [0, 1], always scaled by _HEAD_LENGTH
    cone_can = np.vstack([
        np.c_[_HEAD_RADIUS * cos_a, _HEAD_RADIUS * sin_a, np.zeros(n)],  # base ring
        [[0.0, 0.0, 1.0]],                                                # apex
    ])  # (n+1, 3)
    cone_faces = np.c_[js, js1, np.full(n, n)]  # (n, 3)

    # Canonical caps: center + ring, all at z=0 (no z-scaling needed)
    shaft_cap_can = np.vstack([[[0., 0., 0.]],
                                np.c_[_SHAFT_RADIUS * cos_a, _SHAFT_RADIUS * sin_a, np.zeros(n)]])  # (n+1, 3)
    cone_cap_can  = np.vstack([[[0., 0., 0.]],
                                np.c_[_HEAD_RADIUS  * cos_a, _HEAD_RADIUS  * sin_a, np.zeros(n)]])  # (n+1, 3)
    # winding: [center, j1+1, j+1] → outward normal faces -z
    cap_faces = np.c_[np.zeros(n, int), js1 + 1, js + 1]  # (n, 3)

    D = ends - starts
    lengths = np.linalg.norm(D, axis=1)
    mask = lengths > 1e-10
    if not np.any(mask):
        return None, None

    S = starts[mask]
    D = D[mask]
    L = lengths[mask]
    N = len(S)
    U = D / L[:, None]
    R = _batch_rotation_z_to(U)

    shaft_L = np.maximum(0.0, L - _HEAD_LENGTH)  # only shaft length varies
    cone_starts = S + U * shaft_L[:, None]        # cone placed at shaft tip

    def _transform(can, scale_z, origin):
        """Tile canonical verts, scale z, rotate, translate. Returns (N, V, 3)."""
        v = np.tile(can, (N, 1, 1))
        if scale_z is not None:
            v[:, :, 2] *= scale_z[:, None]
        return np.einsum("nij,nkj->nki", R, v) + origin[:, None, :]

    # Caps displaced by tiny epsilon outward (-U direction) so they're
    # geometrically in front of the shaft/cone side faces — polygon offset
    # alone can't guarantee this since flat caps have DZ=0 but slanted sides don't
    _CAP_BIAS = 0.002
    sv  = _transform(shaft_can,     shaft_L,  S)                         # (N, 2n,   3)
    cv  = _transform(cone_can,      np.full(N, _HEAD_LENGTH), cone_starts)  # (N, n+1,  3)
    bsv = _transform(shaft_cap_can, None,     S              - _CAP_BIAS * U)  # shaft bottom cap
    bcv = _transform(cone_cap_can,  None,     cone_starts    - _CAP_BIAS * U)  # cone base cap

    all_verts = np.vstack([sv.reshape(-1, 3), bsv.reshape(-1, 3),
                           cv.reshape(-1, 3), bcv.reshape(-1, 3)])

    # Face index offsets per section (all-arrows-of-one-type layout)
    i = np.arange(N)
    s_off  = (i * 2 * n)                    [:, None, None]
    bs_off = (N * 2*n       + i * (n + 1))  [:, None, None]
    c_off  = (N * (3*n + 1) + i * (n + 1))  [:, None, None]
    bc_off = (N * (4*n + 2) + i * (n + 1))  [:, None, None]

    all_faces = np.vstack([
        (shaft_faces[None] + s_off ).reshape(-1, 3),  # N*2n  — shaft side
        (cap_faces[None]   + bs_off).reshape(-1, 3),  # N*n   — shaft bottom cap
        (cone_faces[None]  + c_off ).reshape(-1, 3),  # N*n   — cone side
        (cap_faces[None]   + bc_off).reshape(-1, 3),  # N*n   — cone base cap
    ])

    return all_verts, all_faces, U  # U: (N,3) unit arrow directions for color computation


def _cam_forward(canvas):
    """Camera forward direction in world space (unit vector)."""
    try:
        cam = canvas.camera
        p0 = cam.transform.map(np.array([[0., 0.,  0., 1.]]))[0, :3]
        p1 = cam.transform.map(np.array([[0., 0., -1., 1.]]))[0, :3]
        d = p1 - p0
        norm = np.linalg.norm(d)
        if norm > 1e-10:
            return d / norm
    except Exception:
        pass
    return np.array([0., 0., -1.])


def _face_colors_from_dirs(unit_dirs, fwd):
    """Per-face colors: each arrow gets one color lerped by dot(arrow_dir, -fwd).
    dot=-1 (away from camera) → _COLOR_AWAY, dot=+1 (toward) → _COLOR_TOWARD.
    Face layout: N*(2n + n + n + n) = N*5n, section-major order.
    """
    n = _N_SEGMENTS
    t = np.clip(-unit_dirs @ fwd, -1.0, 1.0)   # (N,) in [-1, 1]
    alpha = (t + 1.0) * 0.5                      # 0=away, 1=toward
    colors = (_COLOR_AWAY[None] * (1 - alpha[:, None])
              + _COLOR_TOWARD[None] * alpha[:, None])  # (N, 4)
    # sections have 2n, n, n, n faces each — all in arrow-major order within each section
    return np.vstack([
        np.repeat(colors, 2 * n, axis=0),
        np.repeat(colors, n,     axis=0),
        np.repeat(colors, n,     axis=0),
        np.repeat(colors, n,     axis=0),
    ])


class ForceVectorsElement(VisualElement):
    _starts = None
    _ends = None

    def __init__(self, *args, parent=None, **kwargs):
        from vispy import scene

        self.mesh = scene.visuals.Mesh(
            vertices=np.zeros((3, 3)),
            faces=np.array([[0, 1, 2]]),
            parent=parent,
            color=(0.95, 0.95, 0.95, 1.0),
            shading="flat",
        )
        self.mesh.set_gl_state(
            depth_test=True,
            polygon_offset_fill=True,
            polygon_offset=(-50.0, -50.0),  # shift toward camera so atoms don't occlude arrows
        )
        super().__init__(*args, **kwargs, singleElement=None)

    def show(self):
        self.hidden = False
        self.mesh.visible = True

    def hide(self):
        self.hidden = True
        self.mesh.visible = False

    def onNewGeometry(self):
        self.update()

    def _get_forces(self):
        """Return (N_atoms, 3) forces for current frame, or None if unavailable."""
        settings = self.canvas.settings
        model_key = settings.get("forceVectorsModelKey")
        dataset = self.canvas.dataset
        window = settings.get("forceVectorsAvgWindow")
        index = self.canvas.index

        if model_key is not None:
            env = self.canvas.loupe.env
            model = env.getModel(model_key)
            if model is None:
                return None, "no_prediction"
            data = env.getData("forces", model=model, dataset=dataset)
            if data is None:
                return None, "no_prediction"
            forces_all = data.get("forces")
            if forces_all is None:
                return None, "no_prediction"
            if window > 0:
                n = dataset.getN()
                indices = np.arange(-window, window + 1) + index
                indices = indices[(indices >= 0) & (indices < n)]
                if forces_all.ndim == 3:
                    F = np.mean(forces_all[indices], axis=0)
                else:
                    F = forces_all
            else:
                if forces_all.ndim == 3:
                    F = forces_all[index]
                else:
                    F = forces_all
            return F, None

        if window > 0:
            n = dataset.getN()
            indices = np.arange(-window, window + 1) + index
            indices = indices[(indices >= 0) & (indices < n)]
            F = dataset.getForces(indices=indices)
            F = np.mean(F, axis=0)
        else:
            F = dataset.getForces(indices=index)
        return F, None

    def update(self):
        settings = self.canvas.settings
        show = settings.get("showForceVectors")
        status_label = getattr(self.canvas.loupe, "_forceVectorsStatusLabel", None)

        if not show:
            self.hide()
            if status_label:
                status_label.setVisible(False)
            return

        self.show()

        F, err = self._get_forces()

        if err == "no_prediction":
            self._starts = None
            self._ends = None
            if status_label:
                status_label.setText("No predictions computed for this model")
                status_label.setVisible(True)
            self.queueVisualRefresh()
            return

        if status_label:
            status_label.setVisible(False)

        lengthFactor = settings.get("forceVectorsLength")
        normalised = settings.get("forceVectorsNormalised")
        R = self.canvas.getCurrentR()

        for vOrM in self.canvas.currentTransformations:
            if vOrM.ndim == 2:
                F = F @ vOrM

        if normalised:
            normF = F / np.max(np.linalg.norm(F, axis=1)) * lengthFactor / 5
        else:
            normF = F * lengthFactor / 500

        self._starts = R
        self._ends = R + normF
        self.queueVisualRefresh()

    def onCameraChange(self):
        self.queueVisualRefresh()

    def _draw(self, **kwargs):
        show = self.canvas.settings.get("showForceVectors")

        if self._starts is None or not show:
            self.hide()
            return

        self.show()
        verts, faces, unit_dirs = _build_arrow_mesh(self._starts, self._ends)

        if verts is None:
            self.hide()
            return

        fwd = _cam_forward(self.canvas)
        face_colors = _face_colors_from_dirs(unit_dirs, fwd)
        self.mesh.set_data(vertices=verts, faces=faces, face_colors=face_colors)


def loadLoupe(UIHandler, loupe):
    from UI.Templates import SettingsPane, ObjectComboBox
    from PySide6.QtWidgets import QLabel

    loupe.addVisualElement(ForceVectorsElement, "ForceVectorsElement")

    settings = loupe.settings
    settings.addParameters(
        **{
            "showForceVectors": [False, "updateGeometry"],
            "forceVectorsModelKey": [None, "updateGeometry"],
            "forceVectorsLength": [5, "updateGeometry"],
            "forceVectorsAvgWindow": [0, "updateGeometry"],
            "forceVectorsNormalised": [False, "updateGeometry"],
        }
    )
    settings.markAsPerDataset("forceVectorsModelKey")

    # SETTINGS PANE
    pane = SettingsPane(UIHandler, loupe.settings, parent=loupe)
    loupe.addSidebarPane("FORCE VECTORS", pane)

    pane.addSetting(
        "CheckBox",
        "Enable",
        settingsKey="showForceVectors",
        toolTip="Show a vector field corresponding to the forces",
    )

    # SOURCE SELECTOR
    class _ForceSourceComboBox(ObjectComboBox):
        def updateList(self, *args):
            self.currentlyUpdatingList = True
            model_keys = self.env.getAllModelKeys()
            self.currentKeyList = [None] + model_keys
            self.clear()
            self.addItems(
                ["Ground Truth"]
                + [self.env.getModelOrDataset(k).getDisplayName() for k in model_keys]
            )
            if self.selectedKey in self.currentKeyList:
                self.setCurrentIndex(self.currentKeyList.index(self.selectedKey))
                self.currentlyUpdatingList = False
            elif self.currentKeyList:
                self.setCurrentIndex(0)
                self.currentlyUpdatingList = False
                self.forceUpdate()
            else:
                self.currentlyUpdatingList = False

    sourceCombo = _ForceSourceComboBox(UIHandler, hasDatasets=False)
    sourceCombo.setOnIndexChanged(
        lambda key: settings.setParameter("forceVectorsModelKey", key)
    )
    pane.layout.addWidget(sourceCombo)

    # STATUS LABEL (shown when no predictions available)
    statusLabel = QLabel("No predictions computed for this model")
    statusLabel.setWordWrap(True)
    statusLabel.setStyleSheet("color: orange;")
    statusLabel.setVisible(False)
    pane.layout.addWidget(statusLabel)
    loupe._forceVectorsStatusLabel = statusLabel

    pane.addSetting(
        "Slider",
        "Length",
        settingsKey="forceVectorsLength",
        toolTip="Change the length of the force vectors",
        nMin=1,
        nMax=50,
    )
    pane.addSetting(
        "Slider",
        "Avg. window",
        settingsKey="forceVectorsAvgWindow",
        toolTip="Set the number of points to average around for a smoother result.",
        nMin=0,
        nMax=10000,
    )
    pane.addSetting(
        "CheckBox",
        "Normalised",
        settingsKey="forceVectorsNormalised",
        toolTip="If enabled, set the longest vector for every frame to the same length",
    )
