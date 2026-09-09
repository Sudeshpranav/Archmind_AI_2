import io
import math
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import trimesh
from PIL import Image
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")


# ============================================================
# ARCHMIND PRO V2.5.1
# AI-POWERED CONSTRUCTION MATERIAL INTELLIGENCE
# ============================================================

APP_VERSION = "V2.7.0"

HF_REPO_ID = "AloneMrY/archmind-pro-v2-2-models"

TARGET_FILES = {
    "cement_bags": "cement_bags_model.joblib",
    "steel_tonnes": "steel_tonnes_model.joblib",
    "brick_count": "brick_count_model.joblib",
    "aac_block_count": "aac_block_count_model.joblib",
}

FEATURES = [
    "floor_area_sqft",
    "built_up_area_sqft",
    "num_floors",
    "num_rooms",
    "wall_length_ft",
    "wall_thickness_in",
    "wall_height_ft",
    "foundation_depth_ft",
    "foundation_area_sqft",
    "door_area_sqft",
    "window_area_sqft",
    "slab_area_sqft",
    "roof_area_sqft",
    "concrete_volume_m3",
    "slab_thickness_in",
    "soil_bearing_capacity_kpa",
    "structural_intensity",
]

SOIL_CAPACITY = {
    "weak": 100,
    "normal": 180,
    "good": 300,
}

MASONRY_TYPES = [
    "brick",
    "aac",
    "mixed",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ArchMind Pro",
    page_icon="🏗️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🏗️ ArchMind Pro")

st.caption(
    "AI-powered construction material intelligence "
    "from 3D structural geometry"
)

st.success(
    f"🟢 System Online — {APP_VERSION}"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Project Configuration")

building_type = st.sidebar.selectbox(
    "Building Type",
    [
        "Small Residential",
        "Medium Residential",
        "Large Residential",
        "Small Commercial",
        "Medium Commercial",
        "Large Commercial",
        "Industrial",
        "Institutional",
    ],
)

num_floors_input = st.sidebar.number_input(
    "Selected Number of Floors",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
)

floor_height = st.sidebar.number_input(
    "Floor Height (ft)",
    min_value=7.0,
    max_value=30.0,
    value=10.0,
    step=0.5,
)

foundation_depth = st.sidebar.number_input(
    "Foundation Depth (ft)",
    min_value=1.0,
    max_value=30.0,
    value=5.0,
    step=0.5,
)

wall_thickness = st.sidebar.number_input(
    "Wall Thickness (in)",
    min_value=3.0,
    max_value=24.0,
    value=9.0,
    step=0.5,
)

soil_condition = st.sidebar.selectbox(
    "Soil Condition",
    ["weak", "normal", "good"],
    index=1,
)

masonry_type = st.sidebar.selectbox(
    "Masonry Type",
    MASONRY_TYPES,
)

structural_intensity = st.sidebar.slider(
    "Structural Intensity",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.05,
)

st.sidebar.divider()

uploaded_stl = st.sidebar.file_uploader(
    "Upload STL Building Model",
    type=["stl"],
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Blueprint → 3D",
        "STL Material Prediction",
        "Geometry Analysis",
        "Feature Preview",
        "About",
    ],
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading ArchMind ML models...")
def load_models_from_huggingface():

    models = {}
    errors = []

    for target, filename in TARGET_FILES.items():

        try:

            model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
            )

            models[target] = joblib.load(
                model_path
            )

        except Exception as e:

            errors.append(
                f"{filename}: {str(e)}"
            )

    return models, errors


models, model_errors = (
    load_models_from_huggingface()
)


# ============================================================
# STL LOADING
# ============================================================

def load_stl_mesh(uploaded_file):

    if uploaded_file is None:
        raise ValueError(
            "No STL file provided."
        )

    uploaded_file.seek(0)

    mesh = trimesh.load(
        uploaded_file,
        file_type="stl",
        force="mesh",
    )

    if isinstance(mesh, trimesh.Scene):

        if len(mesh.geometry) == 0:
            raise ValueError(
                "STL scene contains no geometry."
            )

        mesh = trimesh.util.concatenate(
            tuple(
                mesh.geometry.values()
            )
        )

    if mesh is None:
        raise ValueError(
            "Unable to load STL."
        )

    if len(mesh.vertices) == 0:
        raise ValueError(
            "STL contains no vertices."
        )

    if len(mesh.faces) == 0:
        raise ValueError(
            "STL contains no faces."
        )

    return mesh


# ============================================================
# HELPERS
# ============================================================

def safe_div(a, b):

    if b is None or abs(b) < 1e-9:
        return 0.0

    return float(a / b)


def estimate_floor_count(
    height_ft,
    floor_height,
):

    if height_ft <= 0:
        return 1

    if floor_height <= 0:
        return 1

    estimated = int(
        round(
            height_ft /
            floor_height
        )
    )

    return max(
        1,
        min(
            estimated,
            100,
        )
    )


# ============================================================
# VERTICAL PROFILE ANALYSIS
# ============================================================

def analyze_vertical_profile(mesh):

    vertices = np.asarray(
        mesh.vertices
    )

    if len(vertices) < 10:
        return {
            "profile_area_cv": 1.0,
            "profile_width_cv": 1.0,
            "profile_depth_cv": 1.0,
            "top_bottom_area_ratio": 0.0,
            "profile_consistency": 0.0,
            "severe_taper": True,
        }

    mins = vertices.min(
        axis=0
    )

    maxs = vertices.max(
        axis=0
    )

    dims = maxs - mins

    vertical_axis = int(
        np.argmax(dims)
    )

    horizontal_axes = [
        i for i in range(3)
        if i != vertical_axis
    ]

    z_values = vertices[
        :, vertical_axis
    ]

    h1 = float(
        z_values.min()
    )

    h2 = float(
        z_values.max()
    )

    height = h2 - h1

    if height <= 0:
        return {
            "profile_area_cv": 1.0,
            "profile_width_cv": 1.0,
            "profile_depth_cv": 1.0,
            "top_bottom_area_ratio": 0.0,
            "profile_consistency": 0.0,
            "severe_taper": True,
        }

    # --------------------------------------------------------
    # Divide object vertically into slices.
    # --------------------------------------------------------

    num_slices = 12

    ratios = []

    widths = []

    depths = []

    for i in range(num_slices):

        lower = (
            h1 +
            height *
            i /
            num_slices
        )

        upper = (
            h1 +
            height *
            (i + 1) /
            num_slices
        )

        mask = (
            (z_values >= lower) &
            (z_values <= upper)
        )

        pts = vertices[mask]

        if len(pts) < 5:
            continue

        x_values = pts[
            :, horizontal_axes[0]
        ]

        y_values = pts[
            :, horizontal_axes[1]
        ]

        width = (
            x_values.max() -
            x_values.min()
        )

        depth = (
            y_values.max() -
            y_values.min()
        )

        area = width * depth

        widths.append(
            width
        )

        depths.append(
            depth
        )

        ratios.append(
            area
        )

    if len(ratios) < 4:

        return {
            "profile_area_cv": 1.0,
            "profile_width_cv": 1.0,
            "profile_depth_cv": 1.0,
            "top_bottom_area_ratio": 0.0,
            "profile_consistency": 0.0,
            "severe_taper": True,
        }

    ratios = np.asarray(
        ratios,
        dtype=float
    )

    widths = np.asarray(
        widths,
        dtype=float
    )

    depths = np.asarray(
        depths,
        dtype=float
    )

    area_mean = ratios.mean()

    width_mean = widths.mean()

    depth_mean = depths.mean()

    area_cv = safe_div(
        ratios.std(),
        area_mean,
    )

    width_cv = safe_div(
        widths.std(),
        width_mean,
    )

    depth_cv = safe_div(
        depths.std(),
        depth_mean,
    )

    bottom_area = np.mean(
        ratios[:max(1, len(ratios)//3)]
    )

    top_area = np.mean(
        ratios[-max(1, len(ratios)//3):]
    )

    top_bottom_ratio = safe_div(
        top_area,
        bottom_area,
    )

    # --------------------------------------------------------
    # Strong taper detection.
    #
    # Buildings generally maintain a reasonably consistent
    # footprint across floors.
    # Objects such as statues/towers can taper dramatically.
    # --------------------------------------------------------

    severe_taper = (
        top_bottom_ratio < 0.35
        or
        top_bottom_ratio > 1.80
    )

    # --------------------------------------------------------
    # Consistency score
    # --------------------------------------------------------

    inconsistency = (
        area_cv * 0.50
        +
        width_cv * 0.25
        +
        depth_cv * 0.25
    )

    consistency = (
        100 *
        max(
            0.0,
            min(
                1.0,
                1.0 -
                inconsistency,
            )
        )
    )

    return {
        "profile_area_cv": area_cv,
        "profile_width_cv": width_cv,
        "profile_depth_cv": depth_cv,
        "top_bottom_area_ratio":
            top_bottom_ratio,
        "profile_consistency":
            consistency,
        "severe_taper":
            severe_taper,
    }


# ============================================================
# FLOOR / HEIGHT ANALYSIS
# ============================================================

def calculate_height_metrics(
    height_ft,
    selected_floors,
    floor_height,
):

    expected_height = (
        selected_floors *
        floor_height
    )

    height_ratio = safe_div(
        height_ft,
        expected_height,
    )

    estimated_floors = (
        estimate_floor_count(
            height_ft,
            floor_height,
        )
    )

    detected_height = (
        estimated_floors *
        floor_height
    )

    detected_ratio = safe_div(
        height_ft,
        detected_height,
    )

    return {
        "expected_height":
            expected_height,

        "height_ratio":
            height_ratio,

        "estimated_floors":
            estimated_floors,

        "detected_expected_height":
            detected_height,

        "detected_ratio":
            detected_ratio,
    }


# ============================================================
# BUILDING CLASSIFICATION
# ============================================================

def classify_building_geometry(
    geometry,
):

    score = 100.0

    reasons = []

    # --------------------------------------------------------
    # Extract signals
    # --------------------------------------------------------

    fill = geometry[
        "volume_fill_ratio"
    ]

    components = geometry[
        "connected_components"
    ]

    profile = geometry[
        "vertical_profile"
    ]

    profile_consistency = profile[
        "profile_consistency"
    ]

    top_bottom_ratio = profile[
        "top_bottom_area_ratio"
    ]

    severe_taper = profile[
        "severe_taper"
    ]

    surface_volume = geometry[
        "surface_volume_ratio"
    ]

    height_ft = geometry[
        "height_ft"
    ]

    footprint_area = geometry[
        "footprint_area_sqft"
    ]

    # --------------------------------------------------------
    # Basic footprint sanity
    # --------------------------------------------------------

    if footprint_area < 50:

        score -= 25

        reasons.append(
            "Very small footprint"
        )

    # --------------------------------------------------------
    # Volume fill
    # --------------------------------------------------------

    if fill < 0.08:

        score -= 35

        reasons.append(
            "Very sparse geometry"
        )

    elif fill < 0.15:

        score -= 20

        reasons.append(
            "Low volume density"
        )

    elif fill < 0.22:

        score -= 8

        reasons.append(
            "Moderately sparse geometry"
        )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    if components >= 15:

        score -= 25

        reasons.append(
            "Many disconnected components"
        )

    elif components >= 8:

        score -= 15

        reasons.append(
            "Several disconnected components"
        )

    elif components >= 5:

        score -= 5

        reasons.append(
            "Multiple disconnected components"
        )

    # --------------------------------------------------------
    # Surface / volume
    # --------------------------------------------------------

    if surface_volume > 0.20:

        score -= 15

        reasons.append(
            "High surface-to-volume ratio"
        )

    elif surface_volume > 0.12:

        score -= 5

    # --------------------------------------------------------
    # Vertical profile
    # --------------------------------------------------------

    if profile_consistency < 30:

        score -= 35

        reasons.append(
            "Highly inconsistent vertical footprint"
        )

    elif profile_consistency < 50:

        score -= 20

        reasons.append(
            "Inconsistent vertical footprint"
        )

    elif profile_consistency < 70:

        score -= 8

        reasons.append(
            "Moderately variable vertical footprint"
        )

    else:

        reasons.append(
            "Consistent vertical footprint"
        )

    # --------------------------------------------------------
    # Tapering
    # --------------------------------------------------------

    if severe_taper:

        score -= 30

        reasons.append(
            "Severe vertical taper detected"
        )

    elif (
        top_bottom_ratio < 0.60
        or
        top_bottom_ratio > 1.50
    ):

        score -= 12

        reasons.append(
            "Strong vertical taper detected"
        )

    # --------------------------------------------------------
    # Extreme height/footprint ratio
    # --------------------------------------------------------

    footprint_width = math.sqrt(
        max(
            footprint_area,
            1.0,
        )
    )

    height_ratio = safe_div(
        height_ft,
        footprint_width,
    )

    if height_ratio > 8:

        score -= 20

        reasons.append(
            "Extreme height-to-footprint ratio"
        )

    elif height_ratio > 5:

        score -= 8

        reasons.append(
            "High height-to-footprint ratio"
        )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    score = max(
        0,
        min(
            100,
            score,
        )
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if score >= 70:

        classification = (
            "Likely Building"
        )

    elif score >= 45:

        classification = (
            "Uncertain Structure"
        )

    else:

        classification = (
            "Likely Non-Building Object"
        )

    return {
        "confidence":
            score,

        "classification":
            classification,

        "reasons":
            reasons,
    }


# ============================================================
# COMPLETE GEOMETRY ANALYSIS
# ============================================================

def analyze_geometry(
    mesh,
    selected_floors,
    floor_height,
):

    bounds = mesh.bounds

    dimensions = (
        bounds[1] -
        bounds[0]
    )

    dimensions = np.asarray(
        dimensions,
        dtype=float,
    )

    sorted_dims = np.sort(
        dimensions
    )

    footprint_x = float(
        sorted_dims[0]
    )

    footprint_y = float(
        sorted_dims[1]
    )

    height = float(
        sorted_dims[2]
    )

    footprint_area = (
        footprint_x *
        footprint_y
    )

    volume = float(
        abs(mesh.volume)
    )

    surface_area = float(
        mesh.area
    )

    bounding_volume = float(
        np.prod(dimensions)
    )

    volume_fill = safe_div(
        volume,
        bounding_volume,
    )

    surface_volume = safe_div(
        surface_area,
        max(
            volume,
            1e-6,
        ),
    )

    components = len(
        mesh.split(
            only_watertight=False
        )
    )

    height_metrics = (
        calculate_height_metrics(
            height,
            selected_floors,
            floor_height,
        )
    )

    geometry = {

        "dimensions":
            dimensions,

        "footprint_x_ft":
            footprint_x,

        "footprint_y_ft":
            footprint_y,

        "footprint_area_sqft":
            footprint_area,

        "height_ft":
            height,

        "volume":
            volume,

        "surface_area":
            surface_area,

        "bounding_volume":
            bounding_volume,

        "volume_fill_ratio":
            volume_fill,

        "surface_volume_ratio":
            surface_volume,

        "connected_components":
            components,

        "watertight":
            bool(mesh.is_watertight),

        "vertices":
            len(mesh.vertices),

        "faces":
            len(mesh.faces),

    }

    geometry.update(
        height_metrics
    )

    # --------------------------------------------------------
    # New V2.5.1 vertical profile intelligence
    # --------------------------------------------------------

    geometry[
        "vertical_profile"
    ] = analyze_vertical_profile(
        mesh
    )

    # --------------------------------------------------------
    # Building classification
    # --------------------------------------------------------

    classification = (
        classify_building_geometry(
            geometry
        )
    )

    geometry.update(
        classification
    )

    return geometry


# ============================================================
# FLOOR CONFIRMATION
# ============================================================

def handle_floor_confirmation(
    geometry,
    selected_floors,
):

    estimated = geometry[
        "estimated_floors"
    ]

    mismatch = (
        abs(
            estimated -
            selected_floors
        )
        >= 2
    )

    if not mismatch:

        st.session_state[
            "confirmed_floors"
        ] = selected_floors

        return True

    st.warning(
        f"⚠️ Geometry suggests approximately "
        f"**{estimated} floors**, while the project "
        f"configuration currently has "
        f"**{selected_floors} floors**."
    )

    c1, c2 = st.columns(2)

    with c1:

        if st.button(
            f"✅ Use detected {estimated} floors",
            type="primary",
            key=f"detected_{estimated}",
        ):

            st.session_state[
                "confirmed_floors"
            ] = estimated

            st.session_state[
                "floor_confirmation"
            ] = True

            st.rerun()

    with c2:

        if st.button(
            f"↩️ Keep {selected_floors} floors",
            key=f"keep_{selected_floors}",
        ):

            st.session_state[
                "confirmed_floors"
            ] = selected_floors

            st.session_state[
                "floor_confirmation"
            ] = False

            st.rerun()

    confirmed = st.session_state.get(
        "confirmed_floors"
    )

    return (
        confirmed is not None
        and
        confirmed == estimated
    )


# ============================================================
# BUILDING CLASSIFICATION DISPLAY
# ============================================================

def show_building_classification(
    geometry,
):

    st.subheader(
        "🏢 Building Classification"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Building Confidence",
            f"{geometry['confidence']:.0f}%",
        )

    with c2:

        st.metric(
            "Classification",
            geometry[
                "classification"
            ],
        )

    with c3:

        st.metric(
            "Estimated Floors",
            geometry[
                "estimated_floors"
            ],
        )

    classification = geometry[
        "classification"
    ]

    if classification == "Likely Building":

        st.success(
            "🏢 Geometry is structurally plausible "
            "as a building."
        )

    elif classification == "Uncertain Structure":

        st.warning(
            "⚠️ Geometry has mixed structural "
            "characteristics. Manual verification "
            "is recommended."
        )

    else:

        st.error(
            "🚫 Geometry is unlikely to represent "
            "a conventional building."
        )

    with st.expander(
        "🔎 Classification reasoning"
    ):

        for reason in geometry[
            "reasons"
        ]:

            st.write(
                f"• {reason}"
            )

        profile = geometry[
            "vertical_profile"
        ]

        st.write(
            f"Vertical profile consistency: "
            f"**{profile['profile_consistency']:.1f}%**"
        )

        st.write(
            f"Top / bottom footprint ratio: "
            f"**{profile['top_bottom_area_ratio']:.2f}×**"
        )


# ============================================================
# GEOMETRY METRICS DISPLAY
# ============================================================

def show_geometry_metrics(
    geometry,
):

    st.subheader(
        "📐 Geometry Analysis"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Footprint",
            f"{geometry['footprint_area_sqft']:,.0f} sqft",
        )

    with c2:

        st.metric(
            "Model Height",
            f"{geometry['height_ft']:.1f} ft",
        )

    with c3:

        st.metric(
            "Estimated Floors",
            geometry[
                "estimated_floors"
            ],
        )

    c4, c5, c6 = st.columns(3)

    with c4:

        st.metric(
            "Volume Fill",
            f"{geometry['volume_fill_ratio']:.2f}",
        )

    with c5:

        st.metric(
            "Components",
            geometry[
                "connected_components"
            ],
        )

    with c6:

        st.metric(
            "Faces",
            f"{geometry['faces']:,}",
        )

    st.divider()

    data = {
        "Metric": [
            "Footprint X",
            "Footprint Y",
            "Model Height",
            "Volume",
            "Surface Area",
            "Bounding Volume",
            "Volume Fill Ratio",
            "Surface / Volume",
            "Connected Components",
            "Watertight",
            "Vertices",
            "Faces",
            "Estimated Floors",
            "Profile Consistency",
            "Top / Bottom Ratio",
        ],

        "Value": [

            f"{geometry['footprint_x_ft']:.2f} ft",

            f"{geometry['footprint_y_ft']:.2f} ft",

            f"{geometry['height_ft']:.2f} ft",

            f"{geometry['volume']:,.2f}",

            f"{geometry['surface_area']:,.2f}",

            f"{geometry['bounding_volume']:,.2f}",

            f"{geometry['volume_fill_ratio']:.3f}",

            f"{geometry['surface_volume_ratio']:.4f}",

            geometry[
                "connected_components"
            ],

            "Yes"
            if geometry[
                "watertight"
            ]
            else "No",

            f"{geometry['vertices']:,}",

            f"{geometry['faces']:,}",

            geometry[
                "estimated_floors"
            ],

            f"{geometry['vertical_profile']['profile_consistency']:.1f}%",

            f"{geometry['vertical_profile']['top_bottom_area_ratio']:.2f}×",
        ],
    }

    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ============================================================
# V2.7 BLUEPRINT WALL RECONSTRUCTION
# ============================================================

def preprocess_blueprint(image):
    """
    V2.6 Step 1:
    Create several diagnostic representations instead of relying
    on one aggressive adaptive-threshold image.

    The goal is to preserve architectural wall lines while
    suppressing furniture/decorative noise as much as possible.
    """

    img = np.asarray(image.convert("RGB"))

    # Limit processing size for predictable Streamlit performance.
    max_dim = 1800
    h, w = img.shape[:2]

    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edges are more useful for architectural line drawings than
    # a raw threshold on this type of rendered blueprint.
    edges = cv2.Canny(gray, 50, 150)

    # Close small gaps in wall lines.
    line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )
    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        line_kernel,
        iterations=2,
    )

    # A second representation is retained for diagnostics.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    small_kernel = np.ones((3, 3), np.uint8)
    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        small_kernel,
        iterations=1,
    )

    # Combine edge and adaptive evidence.
    combined = cv2.bitwise_or(
        closed_edges,
        adaptive,
    )

    return img, gray, edges, closed_edges, adaptive, combined


def detect_floorplan_bounds(binary):
    """
    Detect the most plausible central architectural plan boundary.

    We deliberately avoid simply selecting contours[0], because a
    rendered blueprint often contains an outer page frame, compass,
    furniture and decorative objects.
    """

    h, w = binary.shape[:2]
    image_area = float(h * w)

    # Ignore a small page margin so the outer image border is less
    # likely to become the floor-plan contour.
    margin_x = max(5, int(w * 0.03))
    margin_y = max(5, int(h * 0.03))

    roi = binary[
        margin_y:h - margin_y,
        margin_x:w - margin_x,
    ]

    contours, _ = cv2.findContours(
        roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.02:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)

        if cw < w * 0.12 or ch < h * 0.12:
            continue

        rect_area = float(cw * ch)
        fill_ratio = safe_div(area, rect_area)

        aspect = safe_div(cw, ch)

        # Most architectural floor plans are not tiny extreme
        # aspect-ratio objects.
        if aspect < 0.12 or aspect > 8.0:
            continue

        # Prefer large, reasonably filled, central candidates.
        center_x = x + cw / 2
        center_y = y + ch / 2

        roi_cx = roi.shape[1] / 2
        roi_cy = roi.shape[0] / 2

        center_distance = (
            safe_div(
                abs(center_x - roi_cx),
                max(roi.shape[1], 1),
            )
            +
            safe_div(
                abs(center_y - roi_cy),
                max(roi.shape[0], 1),
            )
        )

        size_score = min(
            1.0,
            rect_area / (image_area * 0.75),
        )

        fill_score = min(
            1.0,
            max(0.0, fill_ratio),
        )

        central_score = max(
            0.0,
            1.0 - center_distance,
        )

        score = (
            size_score * 0.55
            +
            fill_score * 0.20
            +
            central_score * 0.25
        )

        candidates.append(
            (
                score,
                x + margin_x,
                y + margin_y,
                cw,
                ch,
                area,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    score, x, y, cw, ch, area = candidates[0]

    return {
        "x": int(x),
        "y": int(y),
        "width_px": int(cw),
        "height_px": int(ch),
        "area_px": float(area),
        "score": float(score),
    }


def detect_architectural_lines(edges):
    """
    Detect long horizontal/vertical lines.

    These are not yet room walls. They are V2.6 structural
    candidates used to build the next reconstruction stage.
    """

    h, w = edges.shape[:2]

    min_horizontal = max(30, int(w * 0.08))
    min_vertical = max(30, int(h * 0.08))

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, min_horizontal // 4), 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(3, min_vertical // 4)),
    )

    horizontal = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    combined = cv2.bitwise_or(
        horizontal,
        vertical,
    )

    # Count detected line pixels as a simple diagnostic signal.
    line_pixels = int(
        np.count_nonzero(combined)
    )

    return horizontal, vertical, combined, line_pixels


def crop_blueprint_region(image_array, bounds, padding=8):
    """
    Crop the detected plan for downstream processing.
    """

    if bounds is None:
        return image_array

    h, w = image_array.shape[:2]

    x1 = max(
        0,
        bounds["x"] - padding,
    )
    y1 = max(
        0,
        bounds["y"] - padding,
    )
    x2 = min(
        w,
        bounds["x"]
        + bounds["width_px"]
        + padding,
    )
    y2 = min(
        h,
        bounds["y"]
        + bounds["height_px"]
        + padding,
    )

    return image_array[
        y1:y2,
        x1:x2,
    ]



def detect_wall_segments(plan_gray, min_length_ratio=0.045):
    """V2.7: detect candidate horizontal/vertical wall centerlines.

    This is deliberately a geometry-first detector. It does not claim
    to semantically identify rooms, doors, furniture, or labels yet.
    Long collinear architectural strokes are merged into wall segments.
    """
    gray = np.asarray(plan_gray)
    h, w = gray.shape[:2]

    if h < 20 or w < 20:
        return [], np.zeros_like(gray)

    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    min_len = max(25, int(min(h, w) * min_length_ratio))
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(25, int(min(h, w) * 0.035)),
        minLineLength=min_len,
        maxLineGap=max(8, int(min(h, w) * 0.015)),
    )

    if raw is None:
        return [], edges

    # OpenCV may return HoughLinesP output as either (N, 1, 4) or (N, 4),
    # depending on the OpenCV build/input path. Normalize it before iterating.
    raw = np.asarray(raw)
    if raw.size == 0:
        return [], edges
    if raw.ndim == 3 and raw.shape[-1] == 4:
        raw_lines = raw.reshape(-1, 4)
    elif raw.ndim == 2 and raw.shape[-1] == 4:
        raw_lines = raw.reshape(-1, 4)
    else:
        # Defensive fallback for unexpected OpenCV output.
        try:
            raw_lines = raw.reshape(-1, 4)
        except ValueError:
            return [], edges

    horizontal = []
    vertical = []

    for line in raw_lines:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue

        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle

        # Keep near-axis-aligned architectural lines.
        if angle <= 7:
            y = (y1 + y2) / 2.0
            horizontal.append((y, min(x1, x2), max(x1, x2)))
        elif abs(angle - 90) <= 7:
            x = (x1 + x2) / 2.0
            vertical.append((x, min(y1, y2), max(y1, y2)))

    def merge_segments(items, axis_limit, coord_tol=10, gap_tol=18):
        if not items:
            return []

        # Cluster parallel strokes by their perpendicular coordinate.
        items = sorted(items, key=lambda v: (v[0], v[1]))
        clusters = []
        current = [items[0]]

        for item in items[1:]:
            if abs(item[0] - np.mean([v[0] for v in current])) <= coord_tol:
                current.append(item)
            else:
                clusters.append(current)
                current = [item]
        clusters.append(current)

        output = []
        for cluster in clusters:
            coord = float(np.median([v[0] for v in cluster]))
            intervals = sorted((v[1], v[2]) for v in cluster)
            merged = []
            start, end = intervals[0]

            for a, b in intervals[1:]:
                if a <= end + gap_tol:
                    end = max(end, b)
                else:
                    if end - start >= axis_limit * 0.035:
                        merged.append((start, end))
                    start, end = a, b

            if end - start >= axis_limit * 0.035:
                merged.append((start, end))

            for start, end in merged:
                output.append((coord, start, end))

        return output

    h_segments = merge_segments(horizontal, w)
    v_segments = merge_segments(vertical, h)

    # Convert to a common segment representation and remove very short noise.
    segments = []
    for y, x1, x2 in h_segments:
        length = float(x2 - x1)
        if length >= min_len:
            segments.append({
                "orientation": "horizontal",
                "x1": float(x1), "y1": float(y),
                "x2": float(x2), "y2": float(y),
                "length_px": length,
            })

    for x, y1, y2 in v_segments:
        length = float(y2 - y1)
        if length >= min_len:
            segments.append({
                "orientation": "vertical",
                "x1": float(x), "y1": float(y1),
                "x2": float(x), "y2": float(y2),
                "length_px": length,
            })

    # Prefer the longest candidates if a blueprint is unusually dense.
    segments.sort(key=lambda s: s["length_px"], reverse=True)
    segments = segments[:100]

    overlay = np.zeros_like(gray)
    for seg in segments:
        cv2.line(
            overlay,
            (int(seg["x1"]), int(seg["y1"])),
            (int(seg["x2"]), int(seg["y2"])),
            255,
            2,
        )

    return segments, overlay


def create_wall_reconstruction_mesh(
    segments,
    plan_width_ft,
    plan_depth_ft,
    num_floors,
    floor_height,
    wall_thickness_in,
):
    """Create a procedural 3D wall network from V2.7 line candidates."""
    if not segments:
        return None

    wall_t = max(0.15, wall_thickness_in / 12.0)
    total_height = max(1.0, num_floors * floor_height)

    # Pixel-to-foot mapping for the detected/cropped plan region.
    # x maps left→right, y maps top→bottom.
    parts = []
    used = 0

    for seg in segments:
        x1 = np.clip(seg["x1"], 0, max(1, seg.get("plan_width_px", 1)))
        x2 = np.clip(seg["x2"], 0, max(1, seg.get("plan_width_px", 1)))
        y1 = np.clip(seg["y1"], 0, max(1, seg.get("plan_height_px", 1)))
        y2 = np.clip(seg["y2"], 0, max(1, seg.get("plan_height_px", 1)))

        pw = float(seg.get("plan_width_px", 1))
        ph = float(seg.get("plan_height_px", 1))
        if pw <= 1 or ph <= 1:
            continue

        fx1 = (x1 / pw - 0.5) * plan_width_ft
        fx2 = (x2 / pw - 0.5) * plan_width_ft
        fy1 = (0.5 - y1 / ph) * plan_depth_ft
        fy2 = (0.5 - y2 / ph) * plan_depth_ft

        if seg["orientation"] == "horizontal":
            length = abs(fx2 - fx1)
            if length < 1.5:
                continue
            center = [(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0, 0]
            wall = trimesh.creation.box(
                extents=[length, wall_t, floor_height]
            )
        else:
            length = abs(fy2 - fy1)
            if length < 1.5:
                continue
            center = [(fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0, 0]
            wall = trimesh.creation.box(
                extents=[wall_t, length, floor_height]
            )

        # Repeat the detected floor-plan walls on each configured floor.
        for floor_idx in range(int(num_floors)):
            part = wall.copy()
            part.apply_translation(
                [center[0], center[1], floor_idx * floor_height + floor_height / 2.0]
            )
            parts.append(part)
            used += 1

        if used >= 250:
            break

    if not parts:
        return None

    # Add floor slabs so the reconstructed wall network has a usable building shell.
    slab_t = 0.5
    for floor_idx in range(int(num_floors)):
        slab = trimesh.creation.box(
            extents=[plan_width_ft, plan_depth_ft, slab_t]
        )
        slab.apply_translation(
            [0, 0, floor_idx * floor_height]
        )
        parts.append(slab)

    roof = trimesh.creation.box(
        extents=[plan_width_ft, plan_depth_ft, slab_t]
    )
    roof.apply_translation([0, 0, total_height])
    parts.append(roof)

    return trimesh.util.concatenate(parts)

# PROCEDURAL 3D BUILDING
# ============================================================

def create_building_mesh(
    width_ft,
    depth_ft,
    num_floors,
    floor_height,
    wall_thickness_in,
):

    wall_t = (
        wall_thickness_in /
        12.0
    )

    total_height = (
        num_floors *
        floor_height
    )

    parts = []

    # --------------------------------------------------------
    # Outer walls
    # --------------------------------------------------------

    front = trimesh.creation.box(
        extents=[
            width_ft,
            wall_t,
            total_height,
        ]
    )

    front.apply_translation(
        [
            0,
            -depth_ft / 2,
            total_height / 2,
        ]
    )

    parts.append(
        front
    )

    back = trimesh.creation.box(
        extents=[
            width_ft,
            wall_t,
            total_height,
        ]
    )

    back.apply_translation(
        [
            0,
            depth_ft / 2,
            total_height / 2,
        ]
    )

    parts.append(
        back
    )

    left = trimesh.creation.box(
        extents=[
            wall_t,
            depth_ft,
            total_height,
        ]
    )

    left.apply_translation(
        [
            -width_ft / 2,
            0,
            total_height / 2,
        ]
    )

    parts.append(
        left
    )

    right = trimesh.creation.box(
        extents=[
            wall_t,
            depth_ft,
            total_height,
        ]
    )

    right.apply_translation(
        [
            width_ft / 2,
            0,
            total_height / 2,
        ]
    )

    parts.append(
        right
    )

    # --------------------------------------------------------
    # Floor slabs
    # --------------------------------------------------------

    slab_thickness = 0.5

    for floor in range(
        1,
        num_floors + 1,
    ):

        slab = trimesh.creation.box(
            extents=[
                width_ft,
                depth_ft,
                slab_thickness,
            ]
        )

        slab.apply_translation(
            [
                0,
                0,
                floor *
                floor_height,
            ]
        )

        parts.append(
            slab
        )

    # --------------------------------------------------------
    # Roof
    # --------------------------------------------------------

    roof = trimesh.creation.box(
        extents=[
            width_ft,
            depth_ft,
            slab_thickness,
        ]
    )

    roof.apply_translation(
        [
            0,
            0,
            total_height,
        ]
    )

    parts.append(
        roof
    )

    # --------------------------------------------------------
    # Internal partition
    # --------------------------------------------------------

    if width_ft > 20:

        partition = trimesh.creation.box(
            extents=[
                wall_t,
                max(
                    1,
                    depth_ft -
                    2 * wall_t,
                ),
                total_height,
            ]
        )

        partition.apply_translation(
            [
                0,
                0,
                total_height / 2,
            ]
        )

        parts.append(
            partition
        )

    return trimesh.util.concatenate(
        parts
    )


def export_mesh_stl(mesh):

    buffer = io.BytesIO()

    mesh.export(
        buffer,
        file_type="stl",
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# ML FEATURE GENERATION
# ============================================================

def build_prediction_features(
    geometry,
    num_floors,
    floor_height,
    foundation_depth,
    wall_thickness,
    soil_condition,
    masonry_type,
    structural_intensity,
):

    floor_area = geometry[
        "footprint_area_sqft"
    ]

    built_up_area = (
        floor_area *
        num_floors
    )

    wall_length = (
        2 *
        (
            geometry[
                "footprint_x_ft"
            ]
            +
            geometry[
                "footprint_y_ft"
            ]
        )
        *
        num_floors
    )

    foundation_area = (
        floor_area
    )

    wall_height = (
        floor_height
    )

    slab_area = (
        floor_area
    )

    roof_area = (
        floor_area
    )

    num_rooms_per_floor = max(
        2,
        int(
            round(
                floor_area /
                250
            )
        ),
    )

    num_rooms = (
        num_rooms_per_floor *
        num_floors
    )

    door_area = (
        built_up_area *
        0.025
    )

    window_area = (
        built_up_area *
        0.12
    )

    # --------------------------------------------------------
    # Use geometry volume where possible.
    #
    # STL volume is assumed to be approximately m³ only
    # when the source model was constructed in those units.
    # This remains an engineering-prototyping approximation.
    # --------------------------------------------------------

    concrete_volume = (
        geometry["volume"]
        if geometry["volume"] > 0
        else
        built_up_area *
        0.18 /
        35.3147
    )

    features = {

        "floor_area_sqft":
            floor_area,

        "built_up_area_sqft":
            built_up_area,

        "num_floors":
            num_floors,

        "num_rooms":
            num_rooms,

        "wall_length_ft":
            wall_length,

        "wall_thickness_in":
            wall_thickness,

        "wall_height_ft":
            wall_height,

        "foundation_depth_ft":
            foundation_depth,

        "foundation_area_sqft":
            foundation_area,

        "door_area_sqft":
            door_area,

        "window_area_sqft":
            window_area,

        "slab_area_sqft":
            slab_area,

        "roof_area_sqft":
            roof_area,

        "concrete_volume_m3":
            concrete_volume,

        "slab_thickness_in":
            6.0,

        "soil_bearing_capacity_kpa":
            SOIL_CAPACITY[
                soil_condition
            ],

        "structural_intensity":
            structural_intensity,
    }

    return pd.DataFrame(
        [
            [
                features[f]
                for f in FEATURES
            ]
        ],
        columns=FEATURES,
    )


# ============================================================
# MATERIAL PREDICTION
# ============================================================

def predict_materials(
    feature_df
):

    predictions = {}

    for target, model in models.items():

        try:

            value = model.predict(
                feature_df
            )[0]

            predictions[
                target
            ] = max(
                0,
                float(value),
            )

        except Exception as e:

            predictions[
                target
            ] = None

            st.error(
                f"{target} prediction failed: {e}"
            )

    return predictions


# ============================================================
# ============================================================
# PAGE: BLUEPRINT → 3D
# ============================================================

if page == "Blueprint → 3D":

    st.header("🖼️ Blueprint → 3D Reconstruction")

    st.write(
        "V2.6 Step 1 analyzes the blueprint using computer vision "
        "to identify the primary plan region and long architectural "
        "line candidates before 3D reconstruction."
    )

    st.info(
        "🧱 V2.7 is a geometry-first reconstruction stage. "
        "It does not yet claim semantic room/door/furniture recognition; "
        "the detected wall network is an intermediate reconstruction."
    )

    blueprint = st.file_uploader(
        "Upload Blueprint / Floor Plan",
        type=["png", "jpg", "jpeg"],
        key="blueprint",
    )

    st.subheader("📏 Scale Calibration")

    calibration_mode = st.radio(
        "What real-world dimension do you know?",
        [
            "Overall Building Width",
            "Overall Building Depth",
            "No Overall Dimension Available",
        ],
        horizontal=True,
        help=(
            "Use the outside-to-outside dimension of the complete floor plan. "
            "Do not enter a room dimension such as 10 ft or 11 ft as the overall width."
        ),
    )

    reference_dimension = st.number_input(
        "Known Overall Dimension (ft)",
        min_value=5.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
        disabled=(calibration_mode == "No Overall Dimension Available"),
        help=(
            "Enter a dimension measured across the entire building footprint. "
            "For example, if the whole plan is 35 ft wide, enter 35—not the width of one room."
        ),
    )

    if calibration_mode != "No Overall Dimension Available":
        st.caption(
            "💡 Important: a room label like 11 ft × 15 ft is not the overall building width. "
            "Use the full exterior dimension of the floor plan."
        )

    if blueprint:

        image = Image.open(blueprint).convert("RGB")

        original, gray, edges, closed_edges, adaptive, combined = (
            preprocess_blueprint(image)
        )

        bounds = detect_floorplan_bounds(combined)

        st.subheader("1️⃣ Blueprint Analysis")

        st.image(
            image,
            caption="Uploaded Blueprint",
            use_container_width=True,
        )

        if bounds is None:

            st.error(
                "Could not detect a reliable central floor-plan "
                "region. Try a clearer blueprint with visible wall lines."
            )

        else:

            cropped = crop_blueprint_region(
                original,
                bounds,
                padding=10,
            )

            horizontal, vertical, line_mask, line_pixels = (
                detect_architectural_lines(edges)
            )

            cropped_pil = Image.fromarray(cropped)
            cropped_gray = cv2.cvtColor(
                np.asarray(cropped_pil),
                cv2.COLOR_RGB2GRAY,
            )
            wall_segments, wall_overlay = detect_wall_segments(
                cropped_gray
            )

            for segment in wall_segments:
                segment["plan_width_px"] = cropped_gray.shape[1]
                segment["plan_height_px"] = cropped_gray.shape[0]

            st.success(
                "✅ Primary blueprint region detected."
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Plan Width",
                    f"{bounds['width_px']:,} px",
                )

            with c2:
                st.metric(
                    "Plan Height",
                    f"{bounds['height_px']:,} px",
                )

            with c3:
                st.metric(
                    "Boundary Score",
                    f"{bounds['score']:.2f}",
                )

            with c4:
                st.metric(
                    "Line Pixels",
                    f"{line_pixels:,}",
                )

            st.subheader("2️⃣ Computer Vision Diagnostics")

            d1, d2, d3, d4 = st.columns(4)

            with d1:
                st.image(
                    cv2.cvtColor(
                        edges,
                        cv2.COLOR_GRAY2RGB,
                    ),
                    caption="Edge Map",
                    use_container_width=True,
                )

            with d2:
                st.image(
                    cv2.cvtColor(
                        line_mask,
                        cv2.COLOR_GRAY2RGB,
                    ),
                    caption="Long Architectural Line Candidates",
                    use_container_width=True,
                )

            with d3:
                st.image(
                    cropped,
                    caption="Detected Plan Region",
                    use_container_width=True,
                )

            with d4:
                st.image(
                    cv2.cvtColor(
                        wall_overlay,
                        cv2.COLOR_GRAY2RGB,
                    ),
                    caption=f"V2.7 Wall Candidates ({len(wall_segments)})",
                    use_container_width=True,
                )

            if wall_segments:
                st.success(
                    f"🧱 V2.7 detected {len(wall_segments)} candidate wall segments. "
                    "These are geometric candidates, not yet semantic room labels."
                )
            else:
                st.warning(
                    "⚠️ No stable wall segments were detected. Try a clearer plan "
                    "or a blueprint with stronger architectural wall lines."
                )

            st.subheader("3️⃣ Calibrated Scale Estimate")

            width_px = float(bounds["width_px"])
            height_px = float(bounds["height_px"])
            aspect_ratio = safe_div(height_px, width_px)

            if calibration_mode == "Overall Building Width":
                calibrated_width = float(reference_dimension)
                calibrated_depth = height_px * safe_div(calibrated_width, width_px)
                scale_basis = "overall width"

            elif calibration_mode == "Overall Building Depth":
                calibrated_depth = float(reference_dimension)
                calibrated_width = width_px * safe_div(calibrated_depth, height_px)
                scale_basis = "overall depth"

            else:
                calibrated_width = width_px
                calibrated_depth = height_px
                scale_basis = "pixel geometry (no real-world scale)"

            calibrated_footprint = calibrated_width * calibrated_depth

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Calibrated Width",
                    f"{calibrated_width:.1f} ft" if calibration_mode != "No Overall Dimension Available" else "Unscaled",
                )

            with c2:
                st.metric(
                    "Calibrated Depth",
                    f"{calibrated_depth:.1f} ft" if calibration_mode != "No Overall Dimension Available" else "Unscaled",
                )

            with c3:
                st.metric(
                    "Plan Aspect Ratio",
                    f"1 : {aspect_ratio:.2f}",
                )

            with c4:
                st.metric(
                    "Estimated Footprint",
                    f"{calibrated_footprint:,.0f} sqft" if calibration_mode != "No Overall Dimension Available" else "Unavailable",
                )

            if calibration_mode == "No Overall Dimension Available":
                st.warning(
                    "⚠️ No real-world dimension was supplied. ArchMind can analyze the "
                    "pixel geometry, but it will not claim a real-world footprint or "
                    "use this unscaled geometry for material estimation."
                )
            else:
                st.success(
                    f"✅ Scale calibrated using the {scale_basis}. "
                    "The reference dimension is applied to the complete detected plan boundary."
                )

            st.caption(
                "Automatic dimension/OCR extraction is still a future reconstruction stage. "
                "For now, use an actual overall exterior dimension when available."
            )

            st.subheader("4️⃣ V2.7 Wall-Network 3D Reconstruction")

            st.caption(
                "V2.7 converts detected horizontal/vertical architectural line candidates "
                "into wall meshes. This is the first step beyond the simple rectangular baseline."
            )

            if st.button(
                "🏗️ Reconstruct 3D Wall Network",
                type="primary",
            ):

                if calibration_mode == "No Overall Dimension Available":
                    st.error(
                        "❌ Provide an overall building width or depth before generating "
                        "a real-world 3D reconstruction."
                    )
                    st.stop()

                reconstructed_segments = [dict(seg) for seg in wall_segments]
                generated_mesh = create_wall_reconstruction_mesh(
                    reconstructed_segments,
                    calibrated_width,
                    calibrated_depth,
                    num_floors_input,
                    floor_height,
                    wall_thickness,
                )

                if generated_mesh is None:
                    st.error(
                        "❌ Wall reconstruction failed because no usable wall segments "
                        "were detected. The rectangular baseline is still available below."
                    )
                else:
                    st.session_state["generated_mesh"] = generated_mesh
                    st.success(
                        f"✅ V2.7 wall-network geometry generated from {len(reconstructed_segments)} "
                        "candidate segments."
                    )

            if "generated_mesh" in st.session_state:

                generated_mesh = (
                    st.session_state[
                        "generated_mesh"
                    ]
                )

                st.warning(
                    "V2.7 is an intermediate wall-network reconstruction. "
                    "It uses detected architectural line candidates; room semantics, "
                    "doors/windows, and exact wall topology are the next stage."
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Generated Width",
                        f"{generated_mesh.extents[0]:.1f} ft",
                    )

                with c2:
                    st.metric(
                        "Generated Depth",
                        f"{generated_mesh.extents[1]:.1f} ft",
                    )

                with c3:
                    st.metric(
                        "Generated Height",
                        f"{generated_mesh.extents[2]:.1f} ft",
                    )

                stl_bytes = export_mesh_stl(
                    generated_mesh
                )

                st.download_button(
                    "⬇️ Download Generated STL",
                    data=stl_bytes,
                    file_name="archmind_generated_building.stl",
                    mime="model/stl",
                )


# PAGE: STL MATERIAL PREDICTION
# ============================================================

elif page == "STL Material Prediction":

    st.header(
        "🧱 STL Material Prediction"
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL building model "
            "from the sidebar."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_stl
            )

            geometry = analyze_geometry(
                mesh,
                num_floors_input,
                floor_height,
            )

            show_building_classification(
                geometry
            )

            st.divider()

            # ------------------------------------------------
            # Floor estimation
            # ------------------------------------------------

            estimated = geometry[
                "estimated_floors"
            ]

            selected = (
                num_floors_input
            )

            st.subheader(
                "🏢 Automatic Floor Estimation"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Model Height",
                    f"{geometry['height_ft']:.1f} ft",
                )

            with c2:

                st.metric(
                    "Selected Floors",
                    selected,
                )

            with c3:

                st.metric(
                    "Detected Floors",
                    estimated,
                )

            with c4:

                st.metric(
                    "Height Ratio",
                    f"{geometry['height_ratio']:.2f}×",
                )

            floor_ok = (
                handle_floor_confirmation(
                    geometry,
                    selected,
                )
            )

            st.divider()

            show_geometry_metrics(
                geometry
            )

            st.divider()

            # ------------------------------------------------
            # Final confirmed floors
            # ------------------------------------------------

            confirmed_floors = (
                st.session_state.get(
                    "confirmed_floors"
                )
            )

            if confirmed_floors is None:

                confirmed_floors = selected

            # ------------------------------------------------
            # Final height validation
            # ------------------------------------------------

            final_expected_height = (
                confirmed_floors *
                floor_height
            )

            final_ratio = safe_div(
                geometry["height_ft"],
                final_expected_height,
            )

            final_height_valid = (
                0.60 <=
                final_ratio <=
                1.80
            )

            building_ok = (
                geometry[
                    "classification"
                ]
                ==
                "Likely Building"
            )

            uncertain_ok = (
                geometry[
                    "classification"
                ]
                ==
                "Uncertain Structure"
            )

            # ------------------------------------------------
            # Prediction gate
            # ------------------------------------------------

            prediction_ready = (
                building_ok
                and
                final_height_valid
            )

            if building_ok:

                st.success(
                    "🏢 Building geometry accepted."
                )

            elif uncertain_ok:

                st.warning(
                    "⚠️ Geometry is uncertain. "
                    "Material prediction is paused "
                    "until the structure is manually "
                    "verified."
                )

            else:

                st.error(
                    "🚫 Geometry classified as "
                    "likely non-building."
                )

            if not final_height_valid:

                st.warning(
                    f"⚠️ Confirmed floor configuration "
                    f"expects approximately "
                    f"**{final_expected_height:.1f} ft**, "
                    f"while the model is "
                    f"**{geometry['height_ft']:.1f} ft**."
                )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if prediction_ready:

                st.subheader(
                    "🤖 Material Intelligence"
                )

                st.info(
                    f"Using **{confirmed_floors} floors** "
                    "for material estimation."
                )

                feature_df = (
                    build_prediction_features(
                        geometry,
                        confirmed_floors,
                        floor_height,
                        foundation_depth,
                        wall_thickness,
                        soil_condition,
                        masonry_type,
                        structural_intensity,
                    )
                )

                with st.expander(
                    "🔎 ML Feature Preview"
                ):

                    st.dataframe(
                        feature_df,
                        use_container_width=True,
                    )

                if st.button(
                    "🚀 Run V2.5.1 Material Prediction",
                    type="primary",
                ):

                    if model_errors:

                        st.warning(
                            "Some ML models could not "
                            "be loaded."
                        )

                        for error in model_errors:

                            st.write(
                                error
                            )

                    else:

                        predictions = (
                            predict_materials(
                                feature_df
                            )
                        )

                        st.subheader(
                            "📊 Predicted Materials"
                        )

                        c1, c2 = st.columns(2)

                        with c1:

                            cement = predictions[
                                "cement_bags"
                            ]

                            steel = predictions[
                                "steel_tonnes"
                            ]

                            st.metric(
                                "Cement",
                                (
                                    f"{cement:,.0f} bags"
                                    if cement is not None
                                    else "Unavailable"
                                ),
                            )

                            st.metric(
                                "Steel",
                                (
                                    f"{steel:,.2f} tonnes"
                                    if steel is not None
                                    else "Unavailable"
                                ),
                            )

                        with c2:

                            bricks = predictions[
                                "brick_count"
                            ]

                            aac = predictions[
                                "aac_block_count"
                            ]

                            st.metric(
                                "Bricks",
                                (
                                    f"{bricks:,.0f}"
                                    if bricks is not None
                                    else "Unavailable"
                                ),
                            )

                            st.metric(
                                "AAC Blocks",
                                (
                                    f"{aac:,.0f}"
                                    if aac is not None
                                    else "Unavailable"
                                ),
                            )

                        st.info(
                            "These are predictions from "
                            "the ArchMind V2.2 synthetic "
                            "engineering-informed dataset "
                            "and are intended for ML "
                            "prototyping, not professional "
                            "structural estimation."
                        )

            else:

                st.info(
                    "🔒 Material prediction is currently "
                    "paused until building classification "
                    "and geometry configuration are valid."
                )

        except Exception as e:

            st.error(
                f"STL processing failed: {e}"
            )


# ============================================================
# PAGE: GEOMETRY ANALYSIS
# ============================================================

elif page == "Geometry Analysis":

    st.header(
        "📐 Geometry Intelligence"
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL model from the sidebar."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_stl
            )

            geometry = analyze_geometry(
                mesh,
                num_floors_input,
                floor_height,
            )

            show_building_classification(
                geometry
            )

            st.divider()

            show_geometry_metrics(
                geometry
            )

            st.divider()

            st.subheader(
                "🏢 Floor Estimation"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Model Height",
                    f"{geometry['height_ft']:.1f} ft",
                )

            with c2:

                st.metric(
                    "Selected Floors",
                    num_floors_input,
                )

            with c3:

                st.metric(
                    "Detected Floors",
                    geometry[
                        "estimated_floors"
                    ],
                )

            if (
                geometry[
                    "estimated_floors"
                ]
                !=
                num_floors_input
            ):

                st.warning(
                    "⚠️ The geometry and selected "
                    "floor configuration do not match."
                )

            else:

                st.success(
                    "✅ Floor configuration is "
                    "consistent with geometry."
                )

        except Exception as e:

            st.error(
                f"Geometry analysis failed: {e}"
            )


# ============================================================
# PAGE: FEATURE PREVIEW
# ============================================================

elif page == "Feature Preview":

    st.header(
        "🔎 ML Feature Preview"
    )

    st.write(
        "Features currently supplied to the "
        "material prediction models."
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL model."
        )

    else:

        try:

            mesh = load_stl_mesh(
                uploaded_stl
            )

            geometry = analyze_geometry(
                mesh,
                num_floors_input,
                floor_height,
            )

            confirmed_floors = (
                st.session_state.get(
                    "confirmed_floors"
                )
            )

            if confirmed_floors is None:
                confirmed_floors = num_floors_input

            feature_df = (
                build_prediction_features(
                    geometry,
                    confirmed_floors,
                    floor_height,
                    foundation_depth,
                    wall_thickness,
                    soil_condition,
                    masonry_type,
                    structural_intensity,
                )
            )

            st.dataframe(
                feature_df.T.rename(
                    columns={
                        0: "Value"
                    }
                ),
                use_container_width=True,
            )

        except Exception as e:

            st.error(
                f"Feature generation failed: {e}"
            )


# ============================================================
# PAGE: ABOUT
# ============================================================

elif page == "About":

    st.header(
        "ℹ️ About ArchMind Pro"
    )

    st.markdown(
        """
## ArchMind Pro V2.6.0

ArchMind Pro is an AI-powered construction
material intelligence prototype.

### Current Pipeline

**Blueprint**
→ Computer Vision
→ Structural Footprint
→ Procedural 3D Geometry
→ Geometry Intelligence
→ Building Classification
→ Automatic Floor Estimation
→ User Confirmation
→ Custom ML Models
→ Material Prediction

### V2.6.0 Improvements

- Automatic floor estimation
- Building plausibility classification
- Vertical geometry profile analysis
- Taper detection
- Floor configuration confirmation
- Geometry validation
- STL processing
- Blueprint → 3D MVP
[L26 - Step 1] Multi-stage blueprint preprocessing and architectural line diagnostics
- Existing material prediction models
- Hugging Face model loading

### Important Design Principle

Automatic floor estimation is treated as
a **geometric hypothesis**, not proof that
the object is a building.

Building classification considers:

- footprint
- volume density
- disconnected components
- surface-to-volume ratio
- vertical footprint consistency
- top-to-bottom footprint ratio
- geometric taper
- height-to-footprint relationship

### Limitations

This is an AI/geometry prototype.

It does not provide professional structural
engineering certification or final construction
material quantities.

The material prediction models were trained
using a synthetic engineering-informed dataset
for ML prototyping.

### Technology

Python  
Streamlit  
OpenCV  
Trimesh  
NumPy  
Pandas  
scikit-learn  
XGBoost  
Joblib  
Hugging Face Hub

### Roadmap

**V2.6.0**
Intelligent blueprint preprocessing + building geometry validation

**V3**
Better structural feature extraction

**V3.x**
Algorithm benchmarking

**Future**
Safety monitoring, manpower planning,
progress analysis and explainable AI.
"""
    )
