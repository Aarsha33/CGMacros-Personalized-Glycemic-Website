import streamlit as st
from pathlib import Path
import sys
from importlib.util import spec_from_file_location, module_from_spec
import matplotlib.pyplot as plt


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

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# MODULE LOADER
# ============================================================

def load_module(module_name, filename):
    """Load a Python module whose filename starts with a number."""

    module_path = SRC_DIR / filename

    spec = spec_from_file_location(
        module_name,
        module_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {module_path}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD PREDICTION MODULE
# ============================================================

meal_prediction_module = load_module(
    "cgmacros_meal_prediction",
    "08_meal_prediction.py"
)

predict_meal = meal_prediction_module.predict_meal


# ============================================================
# LOAD FOOD DATABASE
# ============================================================

food_database_module = load_module(
    "cgmacros_food_database",
    "07_food_database.py"
)

load_food_database = food_database_module.load_food_database


# ============================================================
# LOAD WHAT-IF MODULE
# ============================================================

what_if_module = load_module(
    "cgmacros_what_if",
    "09_what_if.py"
)

run_what_if = what_if_module.run_what_if


# ============================================================
# LOAD FOOD DATABASE ONCE
# ============================================================

@st.cache_data
def get_food_database():
    return load_food_database()


food_database = get_food_database()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# HEADER
# ============================================================

st.title("🩸 CGMacros Personalized Glycemic Response")

st.write(
    "Estimate your predicted post-meal glycemic response "
    "using your personal glucose information and meal details."
)

st.info(
    "This is a model prediction, not a medical measurement."
)


# ============================================================
# PERSONAL INFORMATION
# ============================================================

st.header("👤 Your Information")

col1, col2, col3 = st.columns(3)

with col1:

    name = st.text_input(
        "Name",
        placeholder="Enter your name"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        [
            "Prefer not to say",
            "Female",
            "Male",
            "Other"
        ]
    )


with col2:

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=165.0,
        step=0.5
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=60.0,
        step=0.5
    )


with col3:

    if height > 0:
        bmi = weight / ((height / 100) ** 2)
    else:
        bmi = 0.0

    st.metric(
        "BMI",
        f"{bmi:.1f}"
    )

    hba1c = st.number_input(
        "HbA1c (%)",
        min_value=3.0,
        max_value=20.0,
        value=5.4,
        step=0.1
    )

    fasting_glucose = st.number_input(
        "Fasting glucose (mg/dL)",
        min_value=40.0,
        max_value=500.0,
        value=91.0,
        step=1.0
    )


# ============================================================
# MEAL INFORMATION
# ============================================================

st.header("🍽️ Your Meal")

col1, col2 = st.columns(2)

with col1:

    meal_type = st.selectbox(
        "Meal type",
        [
            "Breakfast",
            "Lunch",
            "Dinner",
            "Snack"
        ]
    )


with col2:

    meal_time = st.time_input(
        "Meal time"
    )


# ============================================================
# FOOD SELECTION
# ============================================================

st.subheader("🥗 Food")

food_names = food_database["name"].tolist()

selected_food = st.selectbox(
    "Select a food",
    food_names
)

quantity = st.number_input(
    "Quantity (grams)",
    min_value=1.0,
    max_value=2000.0,
    value=100.0,
    step=5.0
)


# ============================================================
# SHOW NUTRIENTS
# ============================================================

selected_food_row = food_database[
    food_database["name"] == selected_food
].iloc[0]

carbs_preview = (
    float(selected_food_row["carbs_g"])
    * quantity
    / 100
)

protein_preview = (
    float(selected_food_row["protein_g"])
    * quantity
    / 100
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Estimated carbohydrates",
        f"{carbs_preview:.2f} g"
    )

with col2:

    st.metric(
        "Estimated protein",
        f"{protein_preview:.2f} g"
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict My Post-Meal Response",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_button:

    try:

        result = predict_meal(
            food_name=selected_food,
            quantity_grams=quantity,
            hba1c=hba1c,
            fasting_glucose=fasting_glucose,
            meal_time=meal_time.strftime("%H:%M"),
            meal_type=meal_type,
            food_df=food_database
        )

        st.session_state.prediction_result = result

    except Exception as error:

        st.error(
            f"Prediction could not be completed: {error}"
        )


# ============================================================
# DISPLAY RESULT
# ============================================================

if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    st.divider()

    st.header("📊 Your Predicted Response")

    st.metric(
        "Predicted post-meal glycemic response",
        f"{result['Predicted_iAUC']:.2f}"
    )

    st.write(
        f"Based on **{result['Quantity_g']:.0f} g "
        f"of {result['Food']}**, the model predicts "
        f"a post-meal iAUC of "
        f"**{result['Predicted_iAUC']:.2f}**."
    )

    st.caption(
        "iAUC represents the incremental area under the "
        "glucose curve used as the prediction target."
    )


    # ========================================================
    # MEAL SUMMARY
    # ========================================================

    st.subheader("Meal Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.write("**Food**")
        st.write(result["Food"])

    with summary_col2:

        st.write("**Quantity**")
        st.write(f"{result['Quantity_g']:.0f} g")

    with summary_col3:

        st.write("**Meal type**")
        st.write(result["Meal_Type"].title())


    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.write("**Carbohydrates**")
        st.write(f"{result['Carbs_g']:.2f} g")

    with summary_col2:

        st.write("**Protein**")
        st.write(f"{result['Protein_g']:.2f} g")

    with summary_col3:

        st.write("**Meal time**")
        st.write(result["Meal_Time"])


    # ========================================================
    # WHAT-IF SIMULATOR
    # ========================================================

    st.divider()

    st.header("🔄 What-if Simulator")

    st.write(
        "Explore how changing your meal could change the "
        "model's predicted post-meal response."
    )

    st.caption(
        "Your HbA1c, fasting glucose and meal type remain "
        "the same while you explore different meal options."
    )


    # ========================================================
    # WHAT-IF CHANGE TYPE
    # ========================================================

    change_type = st.radio(
        "What would you like to change?",
        [
            "🍽️ Change amount",
            "🥗 Change food",
            "🕐 Change meal time"
        ],
        horizontal=True
    )


    # ========================================================
    # DEFAULT WHAT-IF VALUES
    # ========================================================

    new_food = selected_food
    new_quantity = float(quantity)
    new_meal_time = meal_time


    # ========================================================
    # CHANGE AMOUNT
    # ========================================================

    if change_type == "🍽️ Change amount":

        st.subheader("Change food amount")

        new_quantity = st.slider(
            "New quantity (grams)",
            min_value=1.0,
            max_value=2000.0,
            value=float(quantity),
            step=5.0
        )

        if new_quantity == quantity:

            st.info(
                "Move the slider to try a different quantity."
            )


    # ========================================================
    # CHANGE FOOD
    # ========================================================

    elif change_type == "🥗 Change food":

        st.subheader("Change food")

        new_food = st.selectbox(
            "Choose another food",
            food_names,
            index=(
                food_names.index(selected_food)
                if selected_food in food_names
                else 0
            )
        )

        st.write(
            f"Keeping the quantity at **{quantity:.0f} g**."
        )


    # ========================================================
    # CHANGE MEAL TIME
    # ========================================================

    elif change_type == "🕐 Change meal time":

        st.subheader("Change meal time")

        new_meal_time = st.time_input(
            "New meal time",
            value=meal_time
        )

        if new_meal_time == meal_time:

            st.info(
                "Choose a different time to explore a "
                "what-if scenario."
            )


    # ========================================================
    # RUN WHAT-IF BUTTON
    # ========================================================

    st.write("")

    what_if_button = st.button(
        "🔄 Compare Before vs After",
        type="secondary",
        use_container_width=True
    )


    # ========================================================
    # RUN WHAT-IF SIMULATION
    # ========================================================

    if what_if_button:

        try:

            what_if_result = run_what_if(
                original_food=selected_food,
                original_quantity=float(quantity),
                original_hba1c=float(hba1c),
                original_fasting_glucose=float(
                    fasting_glucose
                ),
                original_meal_time=(
                    meal_time.strftime("%H:%M")
                ),
                original_meal_type=meal_type,
                new_food=new_food,
                new_quantity=float(new_quantity),
                new_meal_time=(
                    new_meal_time.strftime("%H:%M")
                )
            )


            # ==================================================
            # BEFORE VS AFTER VALUES
            # ==================================================

            st.subheader("📊 Your Meal: Before vs After")

            before_value = what_if_result["before"]
            after_value = what_if_result["after"]


            # ==================================================
            # VALUE CARDS
            # ==================================================

            comparison_col1, comparison_col2 = st.columns(2)

            with comparison_col1:

                st.metric(
                    "Before",
                    f"{before_value:.2f}"
                )

            with comparison_col2:

                st.metric(
                    "After",
                    f"{after_value:.2f}",
                    delta=f"{after_value - before_value:+.2f}"
                )


            # ==================================================
            # PERCENTAGE CHANGE
            # ==================================================

            percentage_change = (
                what_if_result["percentage_change"]
            )

            if percentage_change is not None:

                if percentage_change > 0:

                    st.warning(
                        f"📈 The model predicts a "
                        f"**{percentage_change:.2f}% higher** "
                        f"post-meal response after this change."
                    )

                elif percentage_change < 0:

                    st.success(
                        f"📉 The model predicts a "
                        f"**{abs(percentage_change):.2f}% lower** "
                        f"post-meal response after this change."
                    )

                else:

                    st.info(
                        "The model predicts no change in the "
                        "post-meal response."
                    )


            # ==================================================
            # WHAT CHANGED
            # ==================================================

            st.subheader("🔎 What Changed?")

            change_col1, change_col2 = st.columns(2)

            with change_col1:

                st.write("### Before")

                st.write(
                    f"🍽️ **Food:** "
                    f"{what_if_result['original_food']}"
                )

                st.write(
                    f"⚖️ **Amount:** "
                    f"{what_if_result['original_quantity']:.0f} g"
                )

                st.write(
                    f"🕐 **Meal time:** "
                    f"{what_if_result['original_meal_time']}"
                )


            with change_col2:

                st.write("### After")

                st.write(
                    f"🍽️ **Food:** "
                    f"{what_if_result['new_food']}"
                )

                st.write(
                    f"⚖️ **Amount:** "
                    f"{what_if_result['new_quantity']:.0f} g"
                )

                st.write(
                    f"🕐 **Meal time:** "
                    f"{what_if_result['new_meal_time']}"
                )


            # ==================================================
            # GRAPH
            # ==================================================

            st.subheader("📊 Before vs After")

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            labels = [
                "Before",
                "After"
            ]

            values = [
                before_value,
                after_value
            ]

            bars = ax.bar(
                labels,
                values,
                width=0.55
            )

            ax.set_ylabel(
                "Predicted post-meal response (iAUC)"
            )

            ax.set_title(
                "Your Meal: Before vs After",
                fontsize=15,
                fontweight="bold"
            )

            ax.grid(
                axis="y",
                alpha=0.25
            )

            # Display values above bars.
            for bar, value in zip(
                bars,
                values
            ):

                ax.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold"
                )

            # Add space above tallest bar.
            maximum_value = max(values)

            if maximum_value > 0:

                ax.set_ylim(
                    0,
                    maximum_value * 1.20
                )

            # Remove unnecessary borders.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


            # ==================================================
            # INTERPRETATION
            # ==================================================

            st.subheader("💡 What Does This Mean?")

            st.write(
                what_if_result["interpretation"]
            )

            st.caption(
                "This comparison shows model predictions. "
                "It is not a medical measurement and does "
                "not establish a cause-and-effect relationship."
            )


        except Exception as error:

            st.error(
                f"What-if simulation could not be completed: "
                f"{error}"
            )