import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
from datetime import time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CGMacros Personalized Glycemic Response",
    page_icon="🩸",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"


# ============================================================
# LOAD PYTHON MODULES
# ============================================================

def load_module(module_name, filename):

    module_path = SRC_DIR / filename

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# Original iAUC prediction engine
prediction_engine = load_module(
    "prediction_engine",
    "06_prediction_engine.py"
)


# Food database
food_database = load_module(
    "food_database",
    "07_food_database.py"
)


# Glucose trajectory prediction engine
trajectory_module = load_module(
    "glucose_trajectory_prediction",
    "13_glucose_trajectory_prediction.py"
)

predict_glucose_trajectory = (
    trajectory_module.predict_glucose_trajectory
)


# ============================================================
# LOAD FOOD DATABASE
# ============================================================

@st.cache_data
def load_foods():

    return food_database.load_food_database()


food_df = load_foods()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_bmi(weight, height):
    """
    Calculate BMI from weight in kg and height in cm.
    """

    if height <= 0:
        return 0.0

    height_m = height / 100

    return weight / (height_m ** 2)


def calculate_meal_nutrients(food_items):
    """
    Calculate total carbohydrates and protein
    for all foods in the meal.
    """

    total_carbs = 0.0
    total_protein = 0.0

    rows = []

    for item in food_items:

        food_name = item["food"]
        quantity = item["quantity"]

        nutrients = food_database.calculate_nutrients(
            food_name,
            quantity,
            food_df
        )

        carbs = float(nutrients["Carbs_g"])
        protein = float(nutrients["Protein_g"])

        total_carbs += carbs
        total_protein += protein

        rows.append({
            "Food": food_name,
            "Quantity (g)": quantity,
            "Carbohydrates (g)": round(carbs, 2),
            "Protein (g)": round(protein, 2)
        })

    return total_carbs, total_protein, rows


def predict_meal_response(
    hba1c,
    fasting_glucose,
    carbs,
    protein,
    meal_time,
    meal_type
):
    """
    Send the meal information to the trained ANN.
    """

    return prediction_engine.predict_glycemic_response(
        hba1c=hba1c,
        fasting_glucose=fasting_glucose,
        carbs=carbs,
        protein=protein,
        meal_time=meal_time,
        meal_type=meal_type
    )


def plot_glucose_trajectory(
    trajectory,
    title="Predicted Postprandial Glucose Response",
    show_what_if=False,
    what_if_trajectory=None
):
    """
    Plot original glucose trajectory and optionally
    the What-If trajectory on the same graph.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    # Original trajectory
    ax.plot(
        trajectory["Time_After_Meal"],
        trajectory["Predicted_Glucose"],
        marker="o",
        linewidth=2.5,
        label="Original"
    )

    # What-If trajectory
    if (
        show_what_if
        and what_if_trajectory is not None
    ):

        ax.plot(
            what_if_trajectory["Time_After_Meal"],
            what_if_trajectory["Predicted_Glucose"],
            marker="o",
            linewidth=2.5,
            label="What-If"
        )

    ax.set_xlabel(
        "Time after meal (minutes)"
    )

    ax.set_ylabel(
        "Predicted glucose (mg/dL)"
    )

    ax.set_title(title)

    ax.set_xticks(
        [0, 30, 60, 90, 120]
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🩸 CGMacros Personalized Glycemic Response"
)

st.write(
    "ANN-based personalized prediction of postprandial "
    "glycemic response from meal composition and health data."
)

st.info(
    "This is a research and educational prototype. "
    "The prediction is a model estimate and is not a medical measurement."
)


# ============================================================
# USER INFORMATION
# ============================================================

st.header("👤 User Information")

col1, col2, col3 = st.columns(3)


with col1:

    name = st.text_input(
        "Name"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male",
            "Other",
            "Prefer not to say"
        ]
    )


with col2:

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=60.0
    )

    bmi = calculate_bmi(
        weight,
        height
    )

    st.metric(
        "Calculated BMI",
        f"{bmi:.2f}"
    )


with col3:

    hba1c = st.number_input(
        "HbA1c (%)",
        min_value=2.0,
        max_value=20.0,
        value=5.4,
        step=0.1
    )

    fasting_glucose = st.number_input(
        "Fasting Glucose (mg/dL)",
        min_value=40.0,
        max_value=500.0,
        value=91.0,
        step=1.0
    )


# ============================================================
# MEAL INFORMATION
# ============================================================

st.header("🍽️ Meal Information")

col1, col2 = st.columns(2)


with col1:

    meal_type = st.selectbox(
        "Meal Type",
        [
            "breakfast",
            "lunch",
            "dinner",
            "snack"
        ]
    )


with col2:

    meal_time = st.time_input(
        "Meal Time",
        value=time(12, 30)
    )


# ============================================================
# FOOD ITEMS
# ============================================================

st.header("🥗 Food Items")

st.write(
    "Add all foods included in the meal. "
    "For example: rice + chicken + curry + curd."
)


num_foods = st.number_input(
    "Number of food items",
    min_value=1,
    max_value=10,
    value=1,
    step=1
)


food_items = []

food_names = (
    food_df["name"]
    .dropna()
    .tolist()
)


for i in range(int(num_foods)):

    col1, col2 = st.columns([3, 1])

    with col1:

        selected_food = st.selectbox(
            f"Food item {i + 1}",
            food_names,
            key=f"food_{i}"
        )

    with col2:

        quantity = st.number_input(
            "Quantity (g)",
            min_value=1.0,
            max_value=2000.0,
            value=100.0,
            step=10.0,
            key=f"quantity_{i}"
        )

    food_items.append({
        "food": selected_food,
        "quantity": quantity
    })


# ============================================================
# CALCULATE MEAL NUTRIENTS
# ============================================================

if st.button(
    "Calculate Meal Nutrients",
    type="secondary"
):

    try:

        total_carbs, total_protein, nutrient_rows = (
            calculate_meal_nutrients(
                food_items
            )
        )

        st.session_state["nutrient_rows"] = (
            nutrient_rows
        )

        st.session_state["total_carbs"] = (
            total_carbs
        )

        st.session_state["total_protein"] = (
            total_protein
        )

    except Exception as e:

        st.error(
            f"Could not calculate nutrients: {e}"
        )


# ============================================================
# DISPLAY NUTRIENTS
# ============================================================

if "total_carbs" in st.session_state:

    st.subheader(
        "📊 Meal Nutrient Summary"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Carbohydrates",
            f"{st.session_state['total_carbs']:.2f} g"
        )

    with col2:

        st.metric(
            "Total Protein",
            f"{st.session_state['total_protein']:.2f} g"
        )

    nutrient_table = pd.DataFrame(
        st.session_state["nutrient_rows"]
    )

    st.dataframe(
        nutrient_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ANN PREDICTION
# ============================================================

st.header("🧠 ANN Prediction")


if st.button(
    "Predict Glycemic Response",
    type="primary"
):

    try:

        # ----------------------------------------------------
        # CALCULATE TOTAL MEAL NUTRIENTS
        # ----------------------------------------------------

        total_carbs, total_protein, nutrient_rows = (
            calculate_meal_nutrients(
                food_items
            )
        )


        # ----------------------------------------------------
        # ORIGINAL iAUC PREDICTION
        # ----------------------------------------------------

        prediction = predict_meal_response(
            hba1c=hba1c,
            fasting_glucose=fasting_glucose,
            carbs=total_carbs,
            protein=total_protein,
            meal_time=meal_time,
            meal_type=meal_type
        )

        prediction = float(
            prediction
        )


        # ----------------------------------------------------
        # ORIGINAL GLUCOSE TRAJECTORY
        # ----------------------------------------------------

        original_trajectory = (
            predict_glucose_trajectory(
                hba1c=hba1c,
                fasting_glucose=fasting_glucose,
                carbs=total_carbs,
                protein=total_protein,
                meal_time=meal_time,
                meal_type=meal_type
            )
        )


        # ----------------------------------------------------
        # SAVE ORIGINAL RESULTS
        # ----------------------------------------------------

        st.session_state["original_prediction"] = (
            prediction
        )

        st.session_state["original_food_items"] = (
            food_items.copy()
        )

        st.session_state["original_carbs"] = (
            total_carbs
        )

        st.session_state["original_protein"] = (
            total_protein
        )

        st.session_state["original_trajectory"] = (
            original_trajectory
        )

        st.session_state["original_meal_time"] = (
            meal_time
        )

        st.session_state["original_meal_type"] = (
            meal_type
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.success(
            "Prediction calculated successfully."
        )


        st.metric(
            "Predicted Meal-level iAUC",
            f"{prediction:,.2f} mg/dL × min"
        )


        st.caption(
            "iAUC = incremental area under the glucose-response curve."
        )


        # ----------------------------------------------------
        # DISPLAY MEAL COMPOSITION
        # ----------------------------------------------------

        st.subheader(
            "🍽️ Meal Composition"
        )

        st.dataframe(
            pd.DataFrame(nutrient_rows),
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # TOTAL NUTRIENTS
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Carbohydrates",
                f"{total_carbs:.2f} g"
            )

        with col2:

            st.metric(
                "Total Protein",
                f"{total_protein:.2f} g"
            )


        # ----------------------------------------------------
        # ORIGINAL GLUCOSE GRAPH
        # ----------------------------------------------------

        st.subheader(
            "📈 Predicted Postprandial Glucose Response"
        )

        plot_glucose_trajectory(
            original_trajectory,
            title="Predicted Postprandial Glucose Response"
        )

        st.caption(
            "The graph shows the glucose trajectory predicted "
            "by the separate CGM-trained trajectory ANN."
        )


        # ----------------------------------------------------
        # SCIENTIFIC NOTE
        # ----------------------------------------------------

        st.info(
            "Two related ANN models are used: the primary model "
            "predicts meal-level iAUC, while the separate trajectory "
            "model predicts glucose at 0, 30, 60, 90, and 120 minutes."
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

if "original_prediction" in st.session_state:

    st.divider()

    st.header("🔄 What-If Simulator")

    st.write(
        "Create an alternative version of your meal by changing "
        "the quantity of one or more foods, replacing foods, "
        "and/or changing the meal time."
    )

    use_what_if = st.checkbox(
        "Use What-If Simulator"
    )

    if use_what_if:

        st.subheader("🥗 Modify Your Meal")

        st.info(
            "You can change multiple food items at the same time. "
            "Leave a food unchanged if you do not want to modify it."
        )

        # ----------------------------------------------------
        # COPY ORIGINAL MEAL
        # ----------------------------------------------------

        what_if_food_items = []

        for i, original_item in enumerate(food_items):

            st.markdown(
                f"### Food {i + 1}: {original_item['food']}"
            )

            col1, col2, col3 = st.columns([2.5, 1.2, 2.5])

            # ------------------------------------------------
            # FOOD SELECTION
            # ------------------------------------------------

            with col1:

                current_food = st.selectbox(
                    "Food",
                    food_names,
                    index=food_names.index(
                        original_item["food"]
                    ),
                    key=f"what_if_food_{i}"
                )

            # ------------------------------------------------
            # QUANTITY
            # ------------------------------------------------

            with col2:

                current_quantity = st.number_input(
                    "Quantity (g)",
                    min_value=1.0,
                    max_value=2000.0,
                    value=float(
                        original_item["quantity"]
                    ),
                    step=10.0,
                    key=f"what_if_quantity_{i}"
                )

            # ------------------------------------------------
            # ORIGINAL QUANTITY DISPLAY
            # ------------------------------------------------

            with col3:

                st.metric(
                    "Original quantity",
                    f"{original_item['quantity']:.0f} g"
                )

            what_if_food_items.append({
                "food": current_food,
                "quantity": current_quantity
            })

        # ----------------------------------------------------
        # MEAL TIME
        # ----------------------------------------------------

        st.subheader("🕐 Meal Time")

        change_meal_time = st.checkbox(
            "Change meal time"
        )

        if change_meal_time:

            what_if_meal_time = st.time_input(
                "What-If Meal Time",
                value=meal_time
            )

        else:

            what_if_meal_time = meal_time

            st.write(
                f"Original meal time: "
                f"**{meal_time.strftime('%H:%M')}**"
            )

        # ----------------------------------------------------
        # RUN WHAT-IF
        # ----------------------------------------------------

        if st.button(
            "🚀 Run What-If Prediction",
            type="primary"
        ):

            try:

                # =================================================
                # CALCULATE WHAT-IF NUTRIENTS
                # =================================================

                what_if_carbs, what_if_protein, what_if_rows = (
                    calculate_meal_nutrients(
                        what_if_food_items
                    )
                )

                # =================================================
                # WHAT-IF iAUC PREDICTION
                # =================================================

                what_if_prediction = predict_meal_response(
                    hba1c=hba1c,
                    fasting_glucose=fasting_glucose,
                    carbs=what_if_carbs,
                    protein=what_if_protein,
                    meal_time=what_if_meal_time,
                    meal_type=meal_type
                )

                what_if_prediction = float(
                    what_if_prediction
                )

                # =================================================
                # WHAT-IF GLUCOSE TRAJECTORY
                # =================================================

                what_if_trajectory = (
                    predict_glucose_trajectory(
                        hba1c=hba1c,
                        fasting_glucose=fasting_glucose,
                        carbs=what_if_carbs,
                        protein=what_if_protein,
                        meal_time=what_if_meal_time,
                        meal_type=meal_type
                    )
                )

                # =================================================
                # ORIGINAL RESULTS
                # =================================================

                original_prediction = float(
                    st.session_state[
                        "original_prediction"
                    ]
                )

                original_carbs = float(
                    st.session_state[
                        "original_carbs"
                    ]
                )

                original_protein = float(
                    st.session_state[
                        "original_protein"
                    ]
                )

                original_trajectory = (
                    st.session_state[
                        "original_trajectory"
                    ]
                )

                # =================================================
                # CALCULATE CHANGES
                # =================================================

                iauc_change = (
                    what_if_prediction
                    - original_prediction
                )

                if original_prediction != 0:

                    iauc_percentage_change = (
                        iauc_change
                        / original_prediction
                    ) * 100

                else:

                    iauc_percentage_change = 0.0


                carbs_change = (
                    what_if_carbs
                    - original_carbs
                )

                protein_change = (
                    what_if_protein
                    - original_protein
                )

                # =================================================
                # SAVE WHAT-IF RESULTS
                # =================================================

                st.session_state[
                    "what_if_prediction"
                ] = what_if_prediction

                st.session_state[
                    "what_if_trajectory"
                ] = what_if_trajectory

                st.session_state[
                    "what_if_food_items"
                ] = what_if_food_items.copy()

                st.session_state[
                    "what_if_carbs"
                ] = what_if_carbs

                st.session_state[
                    "what_if_protein"
                ] = what_if_protein

                st.session_state[
                    "what_if_meal_time"
                ] = what_if_meal_time

                # =================================================
                # DISPLAY iAUC COMPARISON
                # =================================================

                st.subheader(
                    "📊 Original vs What-If Prediction"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Original iAUC",
                        f"{original_prediction:,.2f}"
                    )

                with col2:

                    st.metric(
                        "What-If iAUC",
                        f"{what_if_prediction:,.2f}"
                    )

                with col3:

                    st.metric(
                        "Change",
                        f"{iauc_change:+,.2f}",
                        delta=(
                            f"{iauc_percentage_change:+.2f}%"
                        )
                    )

                # =================================================
                # NUTRIENT COMPARISON
                # =================================================

                st.subheader(
                    "🍽️ Nutrient Comparison"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Carbohydrates",
                        f"{what_if_carbs:.2f} g",
                        delta=(
                            f"{carbs_change:+.2f} g"
                        )
                    )

                with col2:

                    st.metric(
                        "Protein",
                        f"{what_if_protein:.2f} g",
                        delta=(
                            f"{protein_change:+.2f} g"
                        )
                    )

                # =================================================
                # INTERPRETATION
                # =================================================

                if iauc_change > 0:

                    st.warning(
                        "The model predicts a higher "
                        "post-meal glycemic response "
                        "for the What-If meal."
                    )

                elif iauc_change < 0:

                    st.success(
                        "The model predicts a lower "
                        "post-meal glycemic response "
                        "for the What-If meal."
                    )

                else:

                    st.info(
                        "The model prediction did not change."
                    )

                # =================================================
                # WHAT-IF MEAL TABLE
                # =================================================

                st.subheader(
                    "🥗 What-If Meal Composition"
                )

                st.dataframe(
                    pd.DataFrame(
                        what_if_rows
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                # =================================================
                # ORIGINAL VS WHAT-IF GRAPH
                # =================================================

                st.subheader(
                    "📈 Original vs What-If Glucose Response"
                )

                plot_glucose_trajectory(
                    original_trajectory,
                    title=(
                        "Original vs What-If "
                        "Predicted Glucose Response"
                    ),
                    show_what_if=True,
                    what_if_trajectory=(
                        what_if_trajectory
                    )
                )

                st.caption(
                    "The original and What-If predictions are "
                    "shown on the same time axis (0–120 minutes) "
                    "for direct comparison."
                )

                # =================================================
                # WHAT CHANGED
                # =================================================

                st.subheader(
                    "🔎 What Changed?"
                )

                changes_found = False

                for i in range(
                    len(food_items)
                ):

                    original_item = (
                        food_items[i]
                    )

                    new_item = (
                        what_if_food_items[i]
                    )

                    if (
                        original_item["food"]
                        != new_item["food"]
                    ):

                        st.write(
                            f"**Food {i + 1}:** "
                            f"{original_item['food']} "
                            f"→ "
                            f"{new_item['food']}"
                        )

                        changes_found = True

                    if (
                        original_item["quantity"]
                        != new_item["quantity"]
                    ):

                        st.write(
                            f"**{new_item['food']} quantity:** "
                            f"{original_item['quantity']:.0f} g "
                            f"→ "
                            f"{new_item['quantity']:.0f} g"
                        )

                        changes_found = True

                if change_meal_time:

                    st.write(
                        f"**Meal time:** "
                        f"{meal_time.strftime('%H:%M')} "
                        f"→ "
                        f"{what_if_meal_time.strftime('%H:%M')}"
                    )

                    changes_found = True

                if not changes_found:

                    st.info(
                        "No meal changes were made."
                    )

                # =================================================
                # SCIENTIFIC NOTE
                # =================================================

                st.info(
                    "The primary ANN predicts meal-level iAUC. "
                    "The separate CGM-trained trajectory ANN "
                    "predicts glucose at 0, 30, 60, 90, and "
                    "120 minutes. Both models use the updated "
                    "meal carbohydrate and protein totals."
                )

            except Exception as e:
             st.error(f"Prediction failed: {type(e).__name__}: {e}")



# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CGMacros Personalized Glycemic Response | "
    "ANN Research Prototype"
)

st.caption(
    "Disclaimer: This system provides model-based estimates "
    "for research and educational purposes only. "
    "It is not a medical diagnostic or treatment tool."
)