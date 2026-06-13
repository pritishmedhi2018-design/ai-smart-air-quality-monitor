# ==========================================================
# MODEL PERFORMANCE METRICS
# ==========================================================

def get_metrics():

    metrics={

        "Accuracy":"99.35%",

        "Precision":"99.40%",

        "Recall":"99.35%",

        "F1 Score":"99.36%",

        "CV Accuracy":"97.56%",

        "Silhouette Score":"0.2251"

    }

    return metrics


# ==========================================================
# DISPLAY METRICS
# ==========================================================

def display_metrics(st):

    metrics=get_metrics()

    c1,c2,c3=st.columns(3)

    with c1:
        st.metric(
            "Accuracy",
            metrics["Accuracy"]
        )

    with c2:
        st.metric(
            "Precision",
            metrics["Precision"]
        )

    with c3:
        st.metric(
            "Recall",
            metrics["Recall"]
        )

    c4,c5,c6=st.columns(3)

    with c4:
        st.metric(
            "F1 Score",
            metrics["F1 Score"]
        )

    with c5:
        st.metric(
            "CV Accuracy",
            metrics["CV Accuracy"]
        )

    with c6:
        st.metric(
            "Silhouette Score",
            metrics["Silhouette Score"]
        )