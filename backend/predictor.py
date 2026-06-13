from backend.feature_engineering import create_features

def predict_air_quality(
        temp,
        humidity,
        air_quality,
        dust,
        scaler,
        iso_forest,
        feature_cols,
        source="indoor"
):

    input_df=create_features(
        temp,
        humidity,
        air_quality,
        dust
    )

    X_scaled=scaler.transform(
        input_df[feature_cols]
    )

    anomaly=iso_forest.predict(
        X_scaled
    )[0]

    norm_air=air_quality/2500

    norm_dust=dust/250

    norm_product=(
        air_quality*dust
    )/(2500*250)

    kitchen_penalty=0.25 if source=="kitchen" else 0.0

    weighted=(

        norm_air*0.48+

        norm_dust*0.22+

        norm_product*0.15+

        kitchen_penalty+

        (1 if anomaly==-1 else 0)*0.08

    )

    ai_score=min(
        max(
            weighted**1.15*500,
            0
        ),
        500
    )

    if ai_score<=100:

        category="GOOD"
        color="#22c55e"

    elif ai_score<=220:

        category="MODERATE"
        color="#06b6d4"

    elif ai_score<=380:

        category="POOR"
        color="#f97316"

    else:

        category="HAZARDOUS"
        color="#ef4444"

    return ai_score,category,color,anomaly