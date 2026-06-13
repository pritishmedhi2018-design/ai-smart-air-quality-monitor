import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ==========================================================
# GAUGE CHART
# ==========================================================

def gauge_chart(score,color):

    fig=go.Figure(go.Indicator(

        mode="gauge+number",

        value=score,

        title={
            'text':"Air Quality Score"
        },

        gauge={

            'axis':{
                'range':[0,500]
            },

            'bar':{
                'color':color,
                'thickness':0.35
            },

            'steps':[

                {
                    'range':[0,100],
                    'color':'#22c55e'
                },

                {
                    'range':[100,220],
                    'color':'#06b6d4'
                },

                {
                    'range':[220,380],
                    'color':'#f97316'
                },

                {
                    'range':[380,500],
                    'color':'#ef4444'
                }

            ]

        }

    ))

    fig.update_layout(

        paper_bgcolor="#0B1120",

        font={
            'color':'white'
        },

        height=400

    )

    return fig


# ==========================================================
# AI SCORE TREND
# ==========================================================

def trend_chart(history_df):

    fig=px.line(

        history_df,

        x="Time",

        y="AI_Score",

        markers=True,

        template="plotly_dark"

    )

    fig.update_layout(

        paper_bgcolor="#0B1120",

        plot_bgcolor="#0B1120"

    )

    return fig


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance_chart():

    import pandas as pd

    feature_df=pd.DataFrame({

        "Feature":[

            "Air_Dust_Ratio",

            "Air_Dust_Product",

            "Temp_Hum_Index",

            "AirQuality_Log",

            "Dust_Log"

        ],

        "Importance":[

            0.42,

            0.33,

            0.12,

            0.08,

            0.05

        ]

    })

    fig=px.bar(

        feature_df,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        template="plotly_dark"

    )

    fig.update_layout(

        paper_bgcolor="#0B1120",

        plot_bgcolor="#0B1120"

    )

    return fig


# ==========================================================
# CATEGORY PIE CHART
# ==========================================================

def category_distribution_chart():

    import pandas as pd

    category_df=pd.DataFrame({

        "Category":[

            "Good",

            "Moderate",

            "Poor",

            "Hazardous"

        ],

        "Count":[

            320,

            145,

            78,

            22

        ]

    })

    fig=px.pie(

        category_df,

        values="Count",

        names="Category",

        hole=0.5,

        template="plotly_dark"

    )

    fig.update_layout(

        paper_bgcolor="#0B1120"

    )

    return fig


# ==========================================================
# AI SCORE HISTOGRAM
# ==========================================================

def score_distribution_chart(scores):

    fig=px.histogram(

        x=scores,

        nbins=30,

        template="plotly_dark"

    )

    fig.update_layout(

        xaxis_title="AI Score",

        yaxis_title="Frequency",

        paper_bgcolor="#0B1120",

        plot_bgcolor="#0B1120"

    )

    return fig


# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def correlation_chart(df):

    corr=df.corr(numeric_only=True)

    fig=px.imshow(

        corr,

        text_auto=True,

        color_continuous_scale="Blues"

    )

    fig.update_layout(

        paper_bgcolor="#0B1120"

    )

    return fig


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def confusion_matrix_chart(cm,labels):

    fig=ff.create_annotated_heatmap(

        z=cm,

        x=labels,

        y=labels,

        colorscale='Blues'

    )

    fig.update_layout(

        xaxis_title="Predicted",

        yaxis_title="Actual",

        paper_bgcolor="#0B1120",

        font_color="white"

    )

    return fig


# ==========================================================
# CATEGORY BAR CHART
# ==========================================================

def category_count_chart(category_count):

    fig=px.bar(

        category_count,

        x="Category",

        y="Count",

        color="Category",

        template="plotly_dark"

    )

    fig.update_layout(

        paper_bgcolor="#0B1120",

        plot_bgcolor="#0B1120"

    )

    return fig


# ==========================================================
# FEATURE CONTRIBUTION
# ==========================================================

def feature_contribution_chart(df):

    fig=px.bar(

        df,

        x="Feature",

        y="Contribution",

        color="Feature",

        template="plotly_dark"

    )

    fig.update_layout(

        paper_bgcolor="#0B1120",

        plot_bgcolor="#0B1120"

    )

    return fig
