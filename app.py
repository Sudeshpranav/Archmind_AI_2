import io
import math
import pickle
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
# ARCHMIND PRO V2.5
# AI-POWERED CONSTRUCTION MATERIAL INTELLIGENCE
# ============================================================

APP_VERSION = "V2.5"

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

SOIL_SCORE = {
    "weak": 1,
    "normal": 2,
    "good": 3,
}

MASONRY_TYPES = ["brick", "aac", "mixed"]


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
    "AI-powered construction material intelligence from 3D structural geometry"
)

st.success(f"🟢 System Online — {APP_VERSION}")


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

            models[target] = joblib.load(model_path)

        except Exception as e:

            errors.append(
                f"{filename}: {str(e)}"
            )

    return models, errors


models, model_errors = load_models_from_huggingface()


# ============================================================
# STL LOADING
# ============================================================

def load_stl_mesh(uploaded_file):

    if uploaded_file is None:
        raise ValueError("No STL file provided.")

    uploaded_file.seek(0)

    mesh = trimesh.load(
        uploaded_file,
        file_type="stl",
        force="mesh",
    )

    if isinstance(mesh, trimesh.Scene):

        if len(mesh.geometry) == 0:
            raise ValueError("STL scene contains no geometry.")

        mesh = trimesh.util.concatenate(
            tuple(mesh.geometry.values())
        )

    if mesh is None:
        raise ValueError("Unable to load STL.")

    if len(mesh.vertices) == 0:
        raise ValueError("STL contains no vertices.")

    if len(mesh.faces) == 0:
        raise ValueError("STL contains no faces.")

    return mesh


# ============================================================
# UNIT / GEOMETRY HELPERS
# ============================================================

def safe_div(a, b):

    if b is None or abs(b) < 1e-9:
        return 0.0

    return a / b


def estimate_floor_count(
    height_ft,
    floor_height,
    max_floors=100,
):

    if height_ft <= 0 or floor_height <= 0:
        return 1

    estimated = int(round(height_ft / floor_height))

    estimated = max(1, estimated)

    estimated = min(
        estimated,
        max_floors,
    )

    return estimated


def calculate_height_metrics(
    height_ft,
    selected_floors,
    floor_height,
):

    expected_height = (
        selected_floors * floor_height
    )

    ratio = safe_div(
        height_ft,
        expected_height,
    )

    estimated_floors = estimate_floor_count(
        height_ft,
        floor_height,
    )

    detected_expected_height = (
        estimated_floors * floor_height
    )

    detected_ratio = safe_div(
        height_ft,
        detected_expected_height,
    )

    return {
        "expected_height": expected_height,
        "height_ratio": ratio,
        "estimated_floors": estimated_floors,
        "detected_expected_height": detected_expected_height,
        "detected_ratio": detected_ratio,
    }


# ============================================================
# BUILDING CLASSIFICATION
# ============================================================

def classify_building_geometry(
    geometry,
    selected_floors,
):

    height_ft = geometry["height_ft"]
    footprint_area = geometry["footprint_area_sqft"]
    volume_fill = geometry["volume_fill_ratio"]
    components = geometry["connected_components"]
    surface_volume_ratio = geometry[
        "surface_volume_ratio"
    ]

    estimated_floors = geometry[
        "estimated_floors"
    ]

    floor_mismatch = (
        abs(
            estimated_floors -
            selected_floors
        )
        >= 2
    )

    score = 100

    reasons = []

    # --------------------------------------------------------
    # Height / footprint
    # --------------------------------------------------------

    aspect_ratio = safe_div(
        height_ft,
        math.sqrt(max(footprint_area, 1)),
    )

    if aspect_ratio > 8:
        score -= 30
        reasons.append(
            "Extreme height-to-footprint ratio"
        )

    elif aspect_ratio > 5:
        score -= 10
        reasons.append(
            "High height-to-footprint ratio"
        )

    # --------------------------------------------------------
    # Volume fill
    # --------------------------------------------------------

    if volume_fill < 0.05:

        score -= 30

        reasons.append(
            "Very sparse geometry"
        )

    elif volume_fill < 0.15:

        score -= 15

        reasons.append(
            "Sparse geometry"
        )

    elif volume_fill >= 0.20:

        score += 5

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    if components >= 15:

        score -= 20

        reasons.append(
            "Large number of disconnected components"
        )

    elif components >= 8:

        score -= 10

    # --------------------------------------------------------
    # Surface-volume behavior
    # --------------------------------------------------------

    if surface_volume_ratio > 0.20:

        score -= 10

        reasons.append(
            "High surface-to-volume ratio"
        )

    # --------------------------------------------------------
    # Important:
    # FLOOR MISMATCH DOES NOT AUTOMATICALLY MEAN
    # NON-BUILDING.
    # --------------------------------------------------------

    if floor_mismatch:

        reasons.append(
            f"Selected {selected_floors} floor(s), "
            f"geometry suggests ~{estimated_floors}"
        )

    score = max(
        0,
        min(100, score),
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if score >= 70:

        classification = "Likely Building"

    elif score >= 50:

        classification = "Uncertain Structure"

    else:

        classification = "Likely Non-Building Object"

    return {
        "confidence": score,
        "classification": classification,
        "reasons": reasons,
        "floor_mismatch": floor_mismatch,
    }


# ============================================================
# GEOMETRY ANALYSIS
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

    # --------------------------------------------------------
    # We don't assume STL axis orientation.
    # Largest dimension is treated as candidate vertical.
    # --------------------------------------------------------

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

    footprint_area_model_units = (
        footprint_x *
        footprint_y
    )

    # Current STL workflow assumes
    # model units correspond approximately to feet.
    footprint_area_sqft = float(
        footprint_area_model_units
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

    volume_fill_ratio = safe_div(
        volume,
        bounding_volume,
    )

    surface_volume_ratio = safe_div(
        surface_area,
        max(volume, 1e-6),
    )

    components = len(
        mesh.split(
            only_watertight=False
        )
    )

    height_metrics = calculate_height_metrics(
        height,
        selected_floors,
        floor_height,
    )

    geometry = {
        "dimensions": dimensions,

        "footprint_x_ft": footprint_x,
        "footprint_y_ft": footprint_y,

        "height_ft": height,

        "footprint_area_sqft":
            footprint_area_sqft,

        "volume":
            volume,

        "surface_area":
            surface_area,

        "bounding_volume":
            bounding_volume,

        "volume_fill_ratio":
            volume_fill_ratio,

        "surface_volume_ratio":
            surface_volume_ratio,

        "connected_components":
            components,

        "watertight":
            bool(mesh.is_watertight),

        "vertices":
            len(mesh.vertices),

        "faces":
            len(mesh.faces),

        "estimated_floors":
            height_metrics["estimated_floors"],

        "expected_height":
            height_metrics["expected_height"],

        "height_ratio":
            height_metrics["height_ratio"],

        "detected_expected_height":
            height_metrics[
                "detected_expected_height"
            ],

        "detected_ratio":
            height_metrics[
                "detected_ratio"
            ],
    }

    classification = classify_building_geometry(
        geometry,
        selected_floors,
    )

    geometry.update(
        classification
    )

    return geometry


# ============================================================
# BLUEPRINT PROCESSING
# ============================================================

def preprocess_blueprint(image):

    img = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5,
    )

    kernel = np.ones(
        (3, 3),
        np.uint8,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return img, gray, binary


def detect_floorplan_bounds(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    largest = contours[0]

    x, y, w, h = cv2.boundingRect(
        largest
    )

    image_area = (
        binary.shape[0] *
        binary.shape[1]
    )

    contour_area = cv2.contourArea(
        largest
    )

    if contour_area < image_area * 0.02:
        return None

    return {
        "x": x,
        "y": y,
        "width_px": w,
        "height_px": h,
        "area_px": contour_area,
    }


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

    # --------------------------------------------------------
    # Main outer walls
    # --------------------------------------------------------

    wall_meshes = []

    wall_meshes.append(
        trimesh.creation.box(
            extents=[
                width_ft,
                wall_t,
                total_height,
            ]
        )
    )

    wall_meshes[-1].apply_translation(
        [
            0,
            -depth_ft / 2,
            total_height / 2,
        ]
    )

    wall_meshes.append(
        trimesh.creation.box(
            extents=[
                width_ft,
                wall_t,
                total_height,
            ]
        )
    )

    wall_meshes[-1].apply_translation(
        [
            0,
            depth_ft / 2,
            total_height / 2,
        ]
    )

    wall_meshes.append(
        trimesh.creation.box(
            extents=[
                wall_t,
                depth_ft,
                total_height,
            ]
        )
    )

    wall_meshes[-1].apply_translation(
        [
            -width_ft / 2,
            0,
            total_height / 2,
        ]
    )

    wall_meshes.append(
        trimesh.creation.box(
            extents=[
                wall_t,
                depth_ft,
                total_height,
            ]
        )
    )

    wall_meshes[-1].apply_translation(
        [
            width_ft / 2,
            0,
            total_height / 2,
        ]
    )

    # --------------------------------------------------------
    # Floor slabs
    # --------------------------------------------------------

    slab_thickness = 0.5

    for floor in range(
        1,
        num_floors + 1,
    ):

        z = (
            floor *
            floor_height
        )

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
                z,
            ]
        )

        wall_meshes.append(
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

    wall_meshes.append(
        roof
    )

    # --------------------------------------------------------
    # Simple internal partition
    # --------------------------------------------------------

    if width_ft > 20:

        partition = trimesh.creation.box(
            extents=[
                wall_t,
                depth_ft - 2 * wall_t,
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

        wall_meshes.append(
            partition
        )

    return trimesh.util.concatenate(
        wall_meshes
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
# FEATURE GENERATION
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
            geometry["footprint_x_ft"] +
            geometry["footprint_y_ft"]
        )
        *
        num_floors
    )

    foundation_area = floor_area

    wall_height = (
        floor_height
    )

    slab_area = floor_area

    roof_area = floor_area

    # --------------------------------------------------------
    # Approximate opening areas
    # --------------------------------------------------------

    door_area = (
        built_up_area *
        0.025
    )

    window_area = (
        built_up_area *
        0.12
    )

    # --------------------------------------------------------
    # Approximate room count
    # --------------------------------------------------------

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
            {
                "weak": 100,
                "normal": 180,
                "good": 300,
            }[soil_condition],

        "structural_intensity":
            structural_intensity,
    }

    return pd.DataFrame(
        [[features[f] for f in FEATURES]],
        columns=FEATURES,
    )


# ============================================================
# MATERIAL PREDICTION
# ============================================================

def predict_materials(
    feature_df,
):

    predictions = {}

    for target, model in models.items():

        try:

            value = model.predict(
                feature_df
            )[0]

            predictions[target] = max(
                0,
                float(value),
            )

        except Exception as e:

            predictions[target] = None

            st.error(
                f"{target} prediction failed: {e}"
            )

    return predictions


# ============================================================
# BUILDING CLASSIFICATION UI
# ============================================================

def show_building_classification(
    geometry,
):

    st.subheader(
        "🏢 Building Classification"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Building Confidence",
            f"{geometry['confidence']:.0f}%",
        )

    with col2:

        st.metric(
            "Classification",
            geometry["classification"],
        )

    with col3:

        st.metric(
            "Estimated Floors",
            geometry["estimated_floors"],
        )

    if geometry["classification"] == "Likely Building":

        st.success(
            "🏢 Geometry is structurally plausible "
            "as a building."
        )

    elif geometry["classification"] == "Uncertain Structure":

        st.warning(
            "⚠️ Geometry has mixed building/non-building "
            "characteristics."
        )

    else:

        st.error(
            "🚫 Geometry is unlikely to represent "
            "a conventional building."
        )

    if geometry["reasons"]:

        with st.expander(
            "🔎 Classification reasoning"
        ):

            for reason in geometry["reasons"]:

                st.write(
                    f"• {reason}"
                )


# ============================================================
# FLOOR ESTIMATION UI
# ============================================================

def show_floor_estimation(
    geometry,
    selected_floors,
    floor_height,
):

    estimated = geometry[
        "estimated_floors"
    ]

    height = geometry[
        "height_ft"
    ]

    expected_selected = geometry[
        "expected_height"
    ]

    ratio = geometry[
        "height_ratio"
    ]

    difference = abs(
        height -
        expected_selected
    )

    st.subheader(
        "🏢 Automatic Floor Estimation"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Model Height",
            f"{height:.1f} ft",
        )

    with c2:

        st.metric(
            "Selected Floors",
            selected_floors,
        )

    with c3:

        st.metric(
            "Detected Floors",
            estimated,
        )

    with c4:

        st.metric(
            "Height Ratio",
            f"{ratio:.2f}×",
        )

    # --------------------------------------------------------
    # Mismatch threshold
    # --------------------------------------------------------

    mismatch = (
        abs(
            estimated -
            selected_floors
        )
        >= 2
    )

    if mismatch:

        st.warning(
            f"⚠️ The geometry suggests approximately "
            f"**{estimated} floors**, but the project "
            f"configuration currently has **{selected_floors} floors**."
        )

        st.info(
            f"Model height: **{height:.1f} ft**  \n"
            f"Selected configuration height: "
            f"**{expected_selected:.1f} ft**  \n"
            f"Difference: **{difference:.1f} ft**"
        )

        st.write(
            "ArchMind will not silently change your project "
            "configuration. Confirm the detected floor count "
            "before continuing to material prediction."
        )

        confirm_key = (
            f"floor_confirmed_{estimated}"
        )

        if confirm_key not in st.session_state:

            st.session_state[
                confirm_key
            ] = False

        if st.button(
            f"✅ Use detected {estimated} floors",
            key=f"use_detected_{estimated}",
        ):

            st.session_state[
                "confirmed_floors"
            ] = estimated

            st.session_state[
                "floor_confirmation_active"
            ] = True

            st.rerun()

        if st.button(
            f"↩️ Keep selected {selected_floors} floors",
            key=f"keep_selected_{selected_floors}",
        ):

            st.session_state[
                "confirmed_floors"
            ] = selected_floors

            st.session_state[
                "floor_confirmation_active"
            ] = False

            st.session_state[
                "keep_selected_override"
            ] = True

            st.rerun()

        return False

    # --------------------------------------------------------
    # Close enough
    # --------------------------------------------------------

    height_valid = (
        0.60 <= ratio <= 1.80
    )

    if height_valid:

        st.success(
            f"✅ Geometry height is consistent "
            f"with the selected {selected_floors} floor(s)."
        )

        st.session_state[
            "confirmed_floors"
        ] = selected_floors

        return True

    else:

        st.warning(
            "⚠️ Height is outside the normal "
            "configuration range."
        )

        return False


# ============================================================
# GEOMETRY TABLE
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
            geometry["estimated_floors"],
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
            geometry["connected_components"],
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
            geometry["connected_components"],
            "Yes"
            if geometry["watertight"]
            else "No",
            f"{geometry['vertices']:,}",
            f"{geometry['faces']:,}",
            geometry["estimated_floors"],
        ],
    }

    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# BLUEPRINT → 3D
# ============================================================

if page == "Blueprint → 3D":

    st.header(
        "🖼️ Blueprint → 3D Reconstruction"
    )

    st.write(
        "Upload a 2D floor plan and ArchMind will "
        "extract the primary footprint and generate "
        "a procedural 3D building model."
    )

    blueprint = st.file_uploader(
        "Upload Blueprint / Floor Plan",
        type=[
            "png",
            "jpg",
            "jpeg",
        ],
        key="blueprint",
    )

    known_width = st.number_input(
        "Known Building Width (ft)",
        min_value=5.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
    )

    if blueprint:

        image = Image.open(
            blueprint
        )

        st.image(
            image,
            caption="Uploaded Blueprint",
            use_container_width=True,
        )

        original, gray, binary = (
            preprocess_blueprint(image)
        )

        bounds = detect_floorplan_bounds(
            binary
        )

        if bounds is None:

            st.error(
                "Could not detect a clear floor-plan boundary."
            )

        else:

            detected_width_px = bounds[
                "width_px"
            ]

            detected_height_px = bounds[
                "height_px"
            ]

            scale = (
                known_width /
                detected_width_px
            )

            detected_depth = (
                detected_height_px *
                scale
            )

            st.success(
                "Blueprint boundary detected."
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Detected Width",
                    f"{known_width:.1f} ft",
                )

            with c2:

                st.metric(
                    "Estimated Depth",
                    f"{detected_depth:.1f} ft",
                )

            with c3:

                st.metric(
                    "Floors",
                    num_floors_input,
                )

            if st.button(
                "🏗️ Generate 3D Building",
                type="primary",
            ):

                building_mesh = create_building_mesh(
                    known_width,
                    detected_depth,
                    num_floors_input,
                    floor_height,
                    wall_thickness,
                )

                st.session_state[
                    "generated_mesh"
                ] = building_mesh

                st.success(
                    "3D building geometry generated."
                )

        if (
            "generated_mesh"
            in st.session_state
        ):

            mesh = st.session_state[
                "generated_mesh"
            ]

            st.info(
                "The generated geometry is procedural. "
                "It is intended for MVP reconstruction "
                "and material-intelligence workflows."
            )

            st.metric(
                "Generated Height",
                f"{mesh.extents[2]:.1f} ft",
            )

            stl_bytes = export_mesh_stl(
                mesh
            )

            st.download_button(
                "⬇️ Download Generated STL",
                data=stl_bytes,
                file_name="archmind_generated_building.stl",
                mime="model/stl",
            )


# ============================================================
# STL MATERIAL PREDICTION
# ============================================================

elif page == "STL Material Prediction":

    st.header(
        "🧱 STL Material Prediction"
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL building model from the sidebar."
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

            floor_configuration_ok = (
                show_floor_estimation(
                    geometry,
                    num_floors_input,
                    floor_height,
                )
            )

            st.divider()

            show_geometry_metrics(
                geometry
            )

            st.divider()

            # ------------------------------------------------
            # Prediction gate
            # ------------------------------------------------

            selected_for_prediction = (
                st.session_state.get(
                    "confirmed_floors",
                    num_floors_input,
                )
            )

            classification_ok = (
                geometry["classification"]
                !=
                "Likely Non-Building Object"
            )

            floor_difference = abs(
                geometry["estimated_floors"]
                -
                selected_for_prediction
            )

            final_height_ratio = safe_div(
                geometry["height_ft"],
                (
                    selected_for_prediction *
                    floor_height
                ),
            )

            final_height_valid = (
                0.60 <= final_height_ratio <= 1.80
            )

            prediction_ready = (
                classification_ok
                and
                floor_difference < 2
                and
                final_height_valid
            )

            if not classification_ok:

                st.error(
                    "🚫 Material prediction paused because "
                    "the geometry is currently classified "
                    "as likely non-building."
                )

            elif not prediction_ready:

                st.warning(
                    "⚠️ Material prediction is waiting for "
                    "a valid floor configuration."
                )

            else:

                st.success(
                    f"✅ Geometry validated for approximately "
                    f"{selected_for_prediction} floor(s). "
                    "Material prediction is ready."
                )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if prediction_ready:

                st.subheader(
                    "🤖 Material Intelligence"
                )

                feature_df = build_prediction_features(
                    geometry,
                    selected_for_prediction,
                    floor_height,
                    foundation_depth,
                    wall_thickness,
                    soil_condition,
                    masonry_type,
                    structural_intensity,
                )

                with st.expander(
                    "🔎 Feature Preview"
                ):

                    st.dataframe(
                        feature_df,
                        use_container_width=True,
                    )

                if st.button(
                    "🚀 Run V2.5 Material Prediction",
                    type="primary",
                ):

                    if model_errors:

                        st.warning(
                            "Some models could not be loaded."
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

                        p1, p2 = st.columns(2)

                        with p1:

                            cement = predictions[
                                "cement_bags"
                            ]

                            steel = predictions[
                                "steel_tonnes"
                            ]

                            st.metric(
                                "Cement",
                                f"{cement:,.0f} bags"
                                if cement is not None
                                else "Unavailable",
                            )

                            st.metric(
                                "Steel",
                                f"{steel:,.2f} tonnes"
                                if steel is not None
                                else "Unavailable",
                            )

                        with p2:

                            bricks = predictions[
                                "brick_count"
                            ]

                            aac = predictions[
                                "aac_block_count"
                            ]

                            st.metric(
                                "Bricks",
                                f"{bricks:,.0f}"
                                if bricks is not None
                                else "Unavailable",
                            )

                            st.metric(
                                "AAC Blocks",
                                f"{aac:,.0f}"
                                if aac is not None
                                else "Unavailable",
                            )

                        st.info(
                            "These outputs are ML predictions "
                            "from the ArchMind V2.2 synthetic "
                            "engineering-informed training dataset. "
                            "They are intended for prototyping and "
                            "should not replace professional "
                            "structural/material estimation."
                        )

        except Exception as e:

            st.error(
                f"STL processing failed: {e}"
            )


# ============================================================
# GEOMETRY ANALYSIS PAGE
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

            show_floor_estimation(
                geometry,
                num_floors_input,
                floor_height,
            )

            st.divider()

            show_geometry_metrics(
                geometry
            )

            st.divider()

            st.subheader(
                "🧠 Geometry Observations"
            )

            if geometry[
                "floor_mismatch"
            ]:

                st.warning(
                    f"Selected floor configuration differs "
                    f"from geometry. Estimated floors: "
                    f"**{geometry['estimated_floors']}**."
                )

            if geometry[
                "watertight"
            ]:

                st.success(
                    "✓ Mesh is watertight."
                )

            else:

                st.warning(
                    "⚠ Mesh is not watertight."
                )

            if geometry[
                "volume_fill_ratio"
            ] < 0.15:

                st.warning(
                    "⚠ Geometry has a relatively low "
                    "volume-fill ratio."
                )

            else:

                st.success(
                    "✓ Geometry has reasonable volume density."
                )

        except Exception as e:

            st.error(
                f"Geometry analysis failed: {e}"
            )


# ============================================================
# FEATURE PREVIEW
# ============================================================

elif page == "Feature Preview":

    st.header(
        "🔎 ML Feature Preview"
    )

    st.write(
        "These are the features currently supplied "
        "to the V2.2 material prediction models."
    )

    if uploaded_stl is None:

        st.info(
            "Upload an STL model to generate features."
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

            confirmed_floors = st.session_state.get(
                "confirmed_floors",
                geometry["estimated_floors"],
            )

            feature_df = build_prediction_features(
                geometry,
                confirmed_floors,
                floor_height,
                foundation_depth,
                wall_thickness,
                soil_condition,
                masonry_type,
                structural_intensity,
            )

            st.dataframe(
                feature_df.T.rename(
                    columns={0: "Value"}
                ),
                use_container_width=True,
            )

        except Exception as e:

            st.error(
                f"Feature generation failed: {e}"
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.header(
        "ℹ️ About ArchMind Pro"
    )

    st.markdown(
        """
## ArchMind Pro V2.5

ArchMind Pro is an AI-powered construction
material intelligence prototype.

### Current pipeline

**2D Blueprint**
→ Computer Vision
→ Structural Footprint
→ Procedural 3D Geometry
→ Geometry Intelligence
→ Building Classification
→ Automatic Floor Estimation
→ Custom ML Models
→ Material Prediction

### V2.5 improvements

- Automatic floor estimation
- Building plausibility classification
- Floor configuration confirmation
- Geometry validation
- STL processing
- Blueprint-to-3D MVP
- Material prediction using custom-trained models
- Hugging Face model deployment

### Important limitation

ArchMind Pro currently performs
**geometry-based building plausibility checks**.

It does not claim to understand every
architectural STL or blueprint automatically.

The material models were trained on a
**synthetic engineering-informed dataset**
for ML prototyping and should not be treated
as professional structural engineering output.

### Technology

- Python
- Streamlit
- OpenCV
- Trimesh
- NumPy
- Pandas
- scikit-learn
- XGBoost
- Joblib
- Hugging Face Hub

### Roadmap

**V2.5**
Blueprint → 3D + intelligent floor estimation

**V3**
Better structural feature extraction

**V3.x**
Algorithm benchmarking and model competition

**Future**
Safety monitoring, manpower planning,
progress analysis and explainable AI.
"""
    )
